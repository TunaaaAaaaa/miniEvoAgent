## PR3: task adapters and trace provenance

`evolve_once()` and `evolve()` accept `task` and `provenance`. Existing callers
keep the single-answer QA behavior (`question` input and mean F1). The new
adapters map inputs, extract predictions, evaluate labels and aggregate the
selection score. All built-in scores are maximized; custom adapters must also
return a higher-is-better score. Missing or nonfinite metrics fail explicitly.

| Adapter | Label | Selection metric |
| --- | --- | --- |
| `QATask` (default) | Single answer string | Mean `f1` |
| `NQTask` | Nonempty list of answer aliases | Mean best-alias `f1` |
| `GSM8KTask` | Worked solution or numeric-answer string | Mean `solve_rate` |

NQ also reports EM and Unicode-token answer containment. GSM8K compares the last
number in each response and label, removes thousands separators, and uses an
absolute tolerance of `1e-6`. These adapters follow the EvoAgentX short-answer
[NQ](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/benchmark/nq.py)
and [GSM8K](https://github.com/EvoAgentX/EvoAgentX/blob/main/evoagentx/benchmark/gsm8k.py)
metric conventions, inspected on 2026-09-09. NQ here is not the official
long-answer/span evaluation. Missing numeric GSM8K labels raise an error.

## Running and extending

```python
from evo_repro import GSM8KTask, RunProvenance, evolve_once

trace = evolve_once(
    parent_prompt=parent_prompt,
    train_examples=train_examples,
    dev_examples=dev_examples,
    agent_factory=agent_factory,
    rewriter=rewriter,
    task=GSM8KTask(input_key="problem"),
    provenance=RunProvenance.capture(
        seed=42,
        dataset_name="gsm8k",
        dataset_revision="your-exact-data-release-or-source-commit",
    ),
)
trace.save_json("gsm8k-trace.json")
```

Examples use `sample_id`, `input`, and `label`. Map EvoAgentX NQ's `question` and
`answers` into `input` and `label`; map GSM8K's `question` and `answer` likewise.
Give IDs split-qualified names and use a held-out development split. Raw GSM8K
has no supplied dev split; create and record a deterministic split from training
data without using the test set to choose prompts.

For structured inputs/labels, subclass `TaskAdapter`, implement `collate()` and
`evaluate()`, and set `name` and `score_key`. Optionally override `prediction()`
and `aggregate()`. Feedback values may be JSON objects/lists; unsupported objects
fail instead of being silently stringified. Empty sets, duplicate IDs or IDs
shared between train/dev are rejected before model calls. Only training examples
are sent to the optimizer. ID checks cannot detect duplicate content with new IDs.

## Trace contents

New `EvolutionRoundTrace` objects have version 1. Old serialized traces still
load with unknown/default provenance fields. New traces retain:

- Exact optimizer prompt, original response text, safe optimizer configuration
  captured before generation, and provider response metadata. This comes directly
  from `PromptRewriteResult`, with no dependency on a fake model's `last_prompt`.
- Actual collated inputs, rendered prompts, agent configurations and final LLM
  response metadata for each training, parent-dev and candidate-dev record.
- Task configuration and score key; ordered train/dev fingerprints containing
  sample IDs, inputs and labels; Git SHA and dirty flag; caller-declared dataset
  name/revision and seed.

Git capture defaults to the source checkout of this package. Use
`RunProvenance.capture(repo_path=...)` for a separate experiment repository.
Unavailable Git or dataset information stays unknown. A dirty flag does not save
the working-tree diff, so archive or commit source changes when reproducibility
matters. Fingerprints identify supplied subsets, not an upstream data release.

The provenance seed records the caller's data/preprocessing seed; it does not
change global Python/NumPy RNGs or silently configure an LLM. Pass LLM settings
explicitly, e.g. `OpenAILLM(temperature=0, seed=42, max_completion_tokens=256)`.
Only explicitly set generation parameters are sent, and they are retained in
configuration and response metadata. Seed/temperature support depends on the
chosen model. As documented in the [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create),
seed is best-effort and does not guarantee identical responses; returned model,
response ID, system fingerprint, finish reason and usage are recorded when supplied.
Custom LLM `to_config()` implementations must return safe, JSON-serializable data.

`storage.save_evolution_round(..., metadata=trace.to_dict(mode="json"))` can save
the complete trace using the existing PostgreSQL schema; also pass the exact
optimizer prompt to its `optimizer_prompt` argument. This does not make an entire
multi-generation run atomic. Automatic run orchestration is not added here.

## Verification scope

Run `python examples/task_evolution.py` for both deterministic fixture tasks, or
add `--task gsm8k --output-dir outputs/traces` to save a trace. Fixture answers
are synthetic and scores validate plumbing only, not model improvement. Tests
cover aliases, numeric extraction, structured tasks, metadata round-trips,
invalid splits, stable/changing fingerprints, and mocked OpenAI requests without
network or paid model calls. Full NQ/GSM8K evaluation requires actual datasets,
an explicit data revision and configured model access.
