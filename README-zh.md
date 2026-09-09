# miniEvoAgent

[English](./README.md) | 简体中文

miniEvoAgent 是一个受 [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) 启发的轻量级、可测试学习项目。它保留了智能体构建、工具调用、提示词评估和自进化这几类核心机制，同时去掉大型框架的复杂外围，让每个模块都能被直接阅读、测试和继续扩展。

当前仓库里的 Python 包名是 `evo_repro`。

## miniEvoAgent 是什么？

miniEvoAgent 是一个用于构建和改进 LLM 智能体的最小框架。它把智能体建模为一个清晰的小型执行单元：

- `PromptTemplate` 负责把任务输入渲染成提示词。
- `Action` 负责调用 LLM，可选地执行工具，并解析最终输出。
- `Agent` 封装一个 action，并返回结构化消息。
- `Evaluator` 系列函数用于给 QA 类型预测打分。
- `Evaluator` 类可直接运行 EvoAgentX 风格 benchmark，并可选写入 PostgreSQL。
- `PromptRewriter` 根据反馈样本生成改进后的候选提示词。
- `PromptSelector` 根据开发集表现保留更好的提示词。
- `evolve` 负责运行反馈、重写、评估和选择构成的自进化循环。

相比完整的 EvoAgentX，本项目更偏向研究和教学用途：目标是先把 agent evolution loop 做得透明、可验证，再逐步加入工作流图、记忆、benchmark 适配器、可视化编辑器或更丰富的工具库。

## 核心特性

- 最小智能体抽象：单 action agent，便于理解和调试。
- OpenAI 兼容 LLM 适配器：可通过环境变量配置。
- 工具调用循环：支持 OpenAI 风格 function tools 和多轮工具执行。
- 提示词重写：使用 optimizer LLM 从具体失败案例中改写提示词。
- 开发集选择：只有候选提示词在开发集上更好时才接受。
- 可追踪自进化：记录每一代的训练执行、重写输入、开发集分数和选择决策。
- 统一组件基类：核心数据模型继承 `BaseModel`，支持 `class_name`、`version`、字典/JSON 往返和文件保存。
- 配置/运行时分离：带 LLM、函数等运行时依赖的组件提供 `to_config()`，只保存可复现配置引用。
- 正式 benchmark 适配：不重造 benchmark 家族，直接接收 EvoAgentX 风格的 `get_*_data/get_id/get_label/evaluate` 接口。
- 聚焦测试套件：通过 deterministic fake LLM 测试框架逻辑，不依赖远程模型调用。

## 项目结构

```text
miniEvoAgent/
├── evo_repro/
│   ├── actions.py              # Action 执行与工具调用循环
│   ├── agents.py               # 最小 Agent 封装
│   ├── evaluation.py           # EvoAgentX 风格 benchmark 的薄 Evaluator 适配层
│   ├── evaluators.py           # QA 指标：EM、F1、accuracy
│   ├── llms.py                 # BaseLLM 与 OpenAILLM 适配器
│   ├── messages.py             # 结构化智能体消息
│   ├── parsers.py              # 文本输出解析器
│   ├── prompts.py              # PromptTemplate
│   ├── tools/                  # Tool 封装与示例工具
│   └── optimization/
│       ├── evolution.py        # evolve_once 与 evolve 循环
│       ├── prompt_rewriter.py  # 基于反馈的提示词重写
│       └── selector.py         # 父提示词/候选提示词选择
├── examples/
│   ├── simple_agent.py         # 基础 LLM QA agent
│   ├── tool_agent.py           # 带工具调用能力的 agent
│   ├── prompt_rewriter_demo.py # 使用 fake optimizer LLM 的提示词重写 demo
│   └── evolution_audit.py      # 完整 deterministic evolution trace
├── tests/                      # Agent、工具和自进化逻辑的单元测试
└── pyproject.toml
```

## 安装

需要 Python 3.10 或更新版本。

```powershell
cd D:\Projects\RSIProject\student-project\Evo-repro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

## LLM 配置

默认的 `OpenAILLM` 会从环境变量或本地 `.env` 文件读取配置。

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

如果使用 OpenAI 兼容接口，也可以配置自定义 endpoint：

```powershell
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

也可以复制 `.env.example` 为 `.env`，并在其中填写相同配置。

## 快速开始

创建一个简单问答 agent：

```python
from evo_repro import Action, Agent, OpenAILLM, PromptTemplate, TextOutputParser

llm = OpenAILLM()
action = Action(
    name="answer_question",
    prompt_template=PromptTemplate(
        template="Answer the following question:\n\n{question}"
    ),
    llm=llm,
    output_parser=TextOutputParser(),
)
agent = Agent(
    name="qa_agent",
    description="Answers a single user question.",
    action=action,
)

message = agent.execute({"question": "What is Retrieval-Augmented Generation?"})
print(message.content)
```

运行内置示例：

```powershell
python examples\simple_agent.py
```

## 工具增强 Agent

miniEvoAgent 的工具是 Python 函数外面的一层轻量封装。每个工具向 LLM 暴露 JSON schema，并返回结构化的 `ToolResult`。工具返回值必须是有限的 JSON 可序列化值；框架会在再次调用 LLM 前验证这一契约。

```python
from evo_repro import Action, Agent, OpenAILLM, PromptTemplate, TextOutputParser
from evo_repro import get_word_length_tool

agent = Agent(
    name="tool_agent",
    description="Answers a question and can use tools.",
    action=Action(
        name="answer_with_tools",
        prompt_template=PromptTemplate(
            template="Answer the following question. Use tools when useful:\n\n{question}"
        ),
        llm=OpenAILLM(),
        output_parser=TextOutputParser(),
        tools=[get_word_length_tool],
    ),
)

message = agent.execute({"question": 'How many characters are in "retrieval"?'})
print(message.content)
print(message.metadata["tool_results"])
```

运行：

```powershell
python examples\tool_agent.py
```

## 提示词自进化

自进化循环沿用了 EvoAgentX 优化模块的高层思想，但实现得更小：

1. 用当前提示词运行训练样本。
2. 把 prediction、label 和 metrics 转成反馈样本。
3. 让 `PromptRewriter` 生成一个候选提示词。
4. 在同一份开发集上评估父提示词和候选提示词。
5. 用 `PromptSelector` 选择保留下来的提示词。
6. 按设定代数重复以上流程。

主要入口：

- `evolve_once(...)`：运行一代，并返回 `EvolutionRoundTrace`。
- `evolve(...)`：运行多代，并返回最终提示词和完整历史。

运行一个不依赖远程 LLM 的 deterministic audit：

```powershell
python examples\evolution_audit.py
```

该 audit 会打印每一代的父提示词、训练记录、重写输入、候选提示词、开发集对比和 accept/reject 决策。

## 评估

内置 QA evaluator 会输出：

- Exact match
- Token-level F1
- Accuracy

这些指标故意保持简单，主要用于验证 evolution loop。后续可以替换成面向具体 benchmark 的评估器。

如果要评估正式 benchmark，使用 `Evaluator.evaluate_agent(...)` 直接接入 EvoAgentX 风格对象。miniEvoAgent 不重新实现 NQ、HotPotQA、GSM8K 等 benchmark，只要求传入对象提供：

- `get_train_data()`、`get_dev_data()`、`get_test_data()`
- `get_id(example)`
- `get_label(example)`
- `evaluate(prediction, label)`

`collate_func` 负责把不同 benchmark 的原始样本转换成 agent 输入，`output_postprocess_func` 负责从 `Message` 中抽取最终 prediction。

如需抽样，传入正整数 `sample_k`；`seed` 用于保证兼容访问器的本地抽样可复现。

示例：

```powershell
$env:PYTHONPATH="D:\Projects\RSIProject\reference\EvoAgentX"
python examples\evaluate_evoagentx_benchmark.py
```

## 运行测试

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy evo_repro tests
python -m pytest --cov=evo_repro --cov-report=term-missing --cov-fail-under=85
```

测试中尽量使用 fake LLM，因此大部分框架逻辑无需 API key 也能检查。GitHub Actions 会在 Python 3.10–3.13 上执行相同的 lint、类型检查和测试门禁。

## 多任务演化

演化流程已支持 NQ/GSM8K 任务适配器与版本化审计 trace。
详见[任务适配与可复现信息](docs/evolution-v1.md)；离线验证可运行
`python examples/task_evolution.py`。

## PostgreSQL 实验存储

Schema v2 已加入整次评测事务、不可变输入快照和独立 prompt 血缘记录。
升级前请阅读[迁移、兼容性与集成测试说明](docs/storage-v2.md)。

miniEvoAgent 使用 PostgreSQL 作为后续 benchmark、evaluation 和 evolution trace 的结构化存储。先在 `.env` 中配置连接：

```powershell
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/minievoagent
```

初始化数据表：

```powershell
python scripts\init_postgres_storage.py
```

运行真实数据库写读 smoke check：

```powershell
python scripts\run_postgres_storage_checks.py
```

当前 schema 覆盖：

- `benchmarks`：benchmark 元信息。
- `examples`：train/dev/test 样本和标签。
- `prompts`：prompt 内容、hash 和父 prompt。
- `agent_configs`：可复现 agent 配置。
- `runs`：一次实验运行的配置、状态和最终分数。
- `evaluations`：某个 prompt 在某个 split 上的聚合指标。
- `evaluation_items`：逐样本 prediction、label、metrics 和 rendered prompt。
- `evolution_rounds`：每代 parent/candidate/selected prompt 及选择结果。

`PostgresConfig.to_config()` 和 `PostgreSQLStorage.to_config()` 会脱敏数据库密码，只保存可复现配置，不保存密钥。

## 与 EvoAgentX 的关系

EvoAgentX 是完整的 agentic workflow 框架，包含工作流生成、多智能体协作、工具库、记忆、HITL、benchmark 集成和多种优化算法。

miniEvoAgent 是一个更小的继续开发项目，把其中几类核心机制抽取成更容易阅读的代码：

| EvoAgentX 思路 | miniEvoAgent 实现 |
| --- | --- |
| Agent/action 模块 | `Agent`、`Action`、`PromptTemplate`、`TextOutputParser` |
| 内置工具生态 | 最小 `Tool` 封装和示例工具 |
| 评估 | `evaluators.py` 中的轻量 QA 指标；`evaluation.py` 直接适配 EvoAgentX 风格 benchmark |
| 提示词/工作流优化 | prompt-level rewrite 与 selection loop |
| 实验可追踪性 | `EvolutionRoundTrace` 及相关 record |

## Roadmap

- 加入比单 action agent 更丰富的工作流组合能力。
- 通过统一适配器支持更多 LLM provider。
- 增加小型 QA 和代码生成 benchmark loader。
- 把示例工具扩展成可复用 toolkit。
- 支持将 agent、prompt 和 evolution trace 保存/加载为 JSON。
- 补充架构说明和扩展模式文档。

## License

当前 mini 项目还没有包含 license 文件。正式发布或再分发前，应先补充 license。
