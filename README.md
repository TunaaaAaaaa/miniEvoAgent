# miniEvoAgent

English | [简体中文](./README-zh.md)

miniEvoAgent is a compact, testable learning project inspired by
[EvoAgentX](https://github.com/EvoAgentX/EvoAgentX). It keeps the core ideas of
agent construction, tool calling, prompt evaluation, and self-evolution, while
removing the large-framework complexity so that each mechanism can be read,
tested, and extended directly.

The Python package in this repository is currently named `evo_repro`.

## What is miniEvoAgent?

miniEvoAgent is a minimal framework for building and improving LLM-based agents.
It models an agent as a small execution unit:

- `PromptTemplate` renders task inputs into prompts.
- `Action` calls an LLM, optionally executes tools, and parses the final output.
- `Agent` wraps an action and returns a structured message.
- `Evaluator` functions score QA-style predictions.
- `Evaluator` can run EvoAgentX-style benchmarks and optionally persist results
  to PostgreSQL.
- `PromptRewriter` proposes improved prompts from feedback examples.
- `PromptSelector` keeps the better prompt based on dev-set performance.
- `evolve` runs the feedback, rewrite, evaluation, and selection loop.

Compared with EvoAgentX, this project focuses on a smaller research and teaching
surface: the goal is to make the agent evolution loop transparent before adding
workflow graphs, memory, benchmark adapters, visual editors, or richer toolkits.

## Key Features

- Minimal agent abstraction: a single-action agent that is easy to inspect.
- OpenAI-compatible LLM adapter: configurable with environment variables.
- Tool calling loop: supports OpenAI-style function tools and multi-round tool
  execution.
- Prompt rewriting: uses an optimizer LLM to rewrite prompts from concrete
  failure cases.
- Dev-set selection: accepts a candidate prompt only when it improves the score.
- Traceable evolution: records train executions, rewrite inputs, dev scores, and
  selection decisions for every generation.
- Unified component base: core data models inherit from `BaseModel` with
  `class_name`, `version`, dict/JSON round trips, and file persistence.
- Config/runtime separation: components with LLMs or Python functions expose
  `to_config()` with persistence-friendly references.
- Official benchmark adapter: miniEvoAgent does not reimplement benchmark
  families; it accepts EvoAgentX-style `get_*_data/get_id/get_label/evaluate`
  objects directly.
- Focused test suite: deterministic fake LLMs make the evolution logic testable
  without remote model calls.

## Project Structure

```text
miniEvoAgent/
├── evo_repro/
│   ├── actions.py              # Action execution and tool-call loop
│   ├── agents.py               # Minimal Agent wrapper
│   ├── evaluation.py           # Thin Evaluator adapter for EvoAgentX-style benchmarks
│   ├── evaluators.py           # QA metrics: EM, F1, accuracy
│   ├── llms.py                 # BaseLLM and OpenAILLM adapter
│   ├── messages.py             # Structured agent message
│   ├── parsers.py              # Text output parser
│   ├── prompts.py              # PromptTemplate
│   ├── tools/                  # Tool wrapper and example tool
│   └── optimization/
│       ├── evolution.py        # evolve_once and evolve loops
│       ├── prompt_rewriter.py  # Feedback-driven prompt rewriting
│       └── selector.py         # Parent/candidate prompt selection
├── examples/
│   ├── simple_agent.py         # Basic LLM-backed QA agent
│   ├── tool_agent.py           # Agent with a callable tool
│   ├── prompt_rewriter_demo.py # Prompt rewriting with a fake optimizer LLM
│   └── evolution_audit.py      # Full deterministic evolution trace
├── tests/                      # Unit tests for agents, tools, and evolution
└── pyproject.toml
```

## Installation

Use Python 3.10 or newer.

```powershell
cd D:\Projects\RSIProject\student-project\Evo-repro
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

## LLM Configuration

The default `OpenAILLM` reads configuration from environment variables or a
local `.env` file.

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Optional custom endpoint:

```powershell
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

You can also copy `.env.example` to `.env` and fill in the same values.

## Quick Start

Create a simple question-answering agent:

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

Run the bundled example:

```powershell
python examples\simple_agent.py
```

## Tool-Enabled Agent

miniEvoAgent tools are thin wrappers around Python functions. A tool exposes a
JSON schema to the LLM and returns a structured `ToolResult`. Tool results must
be finite JSON-serializable values; the framework validates this contract before
calling the LLM again.

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

Run:

```powershell
python examples\tool_agent.py
```

## Prompt Self-Evolution

The self-evolution loop follows the same high-level idea as EvoAgentX
optimization modules, but in a smaller form:

1. Run the current prompt on training examples.
2. Convert predictions, labels, and metrics into feedback examples.
3. Ask a `PromptRewriter` to generate one candidate prompt.
4. Evaluate both parent and candidate prompts on the same dev set.
5. Use `PromptSelector` to choose the survivor.
6. Repeat for the configured number of generations.

The main entry points are:

- `evolve_once(...)`: run one generation and return an `EvolutionRoundTrace`.
- `evolve(...)`: run multiple generations and return the final prompt plus the
  full history.

Run a deterministic audit without calling a remote LLM:

```powershell
python examples\evolution_audit.py
```

The audit prints each generation's parent prompt, training records, rewrite
input, candidate prompt, dev-set comparison, and accept/reject decision.

## Evaluation

The built-in QA evaluator reports:

- Exact match
- Token-level F1
- Accuracy

These metrics are intentionally simple. They are useful for verifying the
evolution loop and can be replaced later with benchmark-specific evaluators.

For official benchmarks, use `Evaluator.evaluate_agent(...)` with an
EvoAgentX-style object. miniEvoAgent does not reimplement NQ, HotPotQA, GSM8K,
or similar benchmark families; it expects the object to provide:

- `get_train_data()`, `get_dev_data()`, `get_test_data()`
- `get_id(example)`
- `get_label(example)`
- `evaluate(prediction, label)`

`collate_func` converts each raw benchmark example into agent inputs, and
`output_postprocess_func` extracts the final prediction from the returned
`Message`.

For sampling, pass a positive `sample_k`; `seed` makes local sampling for legacy
accessors reproducible.

Example:

```powershell
$env:PYTHONPATH="D:\Projects\RSIProject\reference\EvoAgentX"
python examples\evaluate_evoagentx_benchmark.py
```

## Running Tests

```powershell
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy evo_repro tests
python -m pytest --cov=evo_repro --cov-report=term-missing --cov-fail-under=85
```

The tests use fake LLM implementations where possible, so most framework logic
can be checked without API keys. GitHub Actions runs the same lint, type-check,
and test gates on Python 3.10 through 3.13.

## PostgreSQL Experiment Storage

Schema v2 adds atomic evaluation saves, input snapshots and prompt lineage.
See [storage v2 migration and compatibility notes](docs/storage-v2.md) before upgrading.

miniEvoAgent uses PostgreSQL as the structured storage layer for upcoming
benchmark, evaluation, and evolution-trace workflows. Configure the connection
in `.env`:

```powershell
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/minievoagent
```

Initialize the tables:

```powershell
python scripts\init_postgres_storage.py
```

Run the live database write/read smoke check:

```powershell
python scripts\run_postgres_storage_checks.py
```

The current schema covers:

- `benchmarks`: benchmark metadata.
- `examples`: train/dev/test examples and labels.
- `prompts`: prompt content, hashes, and parent prompts.
- `agent_configs`: reproducible agent configurations.
- `runs`: experiment run configuration, status, and final scores.
- `evaluations`: aggregate metrics for a prompt on a split.
- `evaluation_items`: per-example prediction, label, metrics, and rendered prompt.
- `evolution_rounds`: parent/candidate/selected prompts and selection decisions.

`PostgresConfig.to_config()` and `PostgreSQLStorage.to_config()` mask database
passwords so persisted configs do not store secrets.

## Relationship to EvoAgentX

EvoAgentX is a full agentic workflow framework with workflow generation,
multi-agent orchestration, toolkits, memory, HITL, benchmark integration, and
multiple optimization algorithms.

miniEvoAgent is a smaller continuation project that extracts a few core
mechanisms into a readable codebase:

| EvoAgentX idea | miniEvoAgent implementation |
| --- | --- |
| Agent/action modules | `Agent`, `Action`, `PromptTemplate`, `TextOutputParser` |
| Built-in tool ecosystem | Minimal `Tool` wrapper plus example tools |
| Evaluation | Lightweight QA metrics in `evaluators.py`; `evaluation.py` directly adapts EvoAgentX-style benchmarks |
| Prompt/workflow optimization | Prompt-level rewrite and selection loop |
| Experiment traceability | `EvolutionRoundTrace` and related records |

## Roadmap

- Add richer workflow composition beyond a single-action agent.
- Support more LLM providers through a shared adapter interface.
- Add benchmark loaders for small QA and code-generation datasets.
- Extend tool examples into reusable toolkits.
- Save and load agent, prompt, and evolution traces as JSON.
- Add documentation pages for architecture and extension patterns.

## License

No license file is currently included in this mini project. Add one before
publishing or redistributing the code.
