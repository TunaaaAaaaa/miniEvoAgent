## miniEvoAgent improvement plan

Verified on 2026-09-09: the checkout moved to `D:/Projects/miniEvoAgent`;
local main and fetched origin/main both point to `adda212`. The working tree
was clean before PR2 work. The baseline suite passed all 63 tests.

## PR1 — completed, committed and pushed

Contract validation, TypeError handling, JSON tool results, URL secret redaction,
template variables, typing and CI quality gates. Commit: `adda212`.

## PR2 — completed and locally verified

Commit: `af718fd`.

Versioned transactional schema upgrades; immutable sample versions and evaluation
input snapshots; evaluated-subset fingerprints; separate prompt text identity and
occurrence lineage; atomic evaluation persistence; disposable PostgreSQL tests
and CI service. See `storage-v2.md` for migration and adapter compatibility.

Local validation on 2026-09-09: 76 tests passed, coverage 93.22%; all 12
PostgreSQL integration tests also passed against PostgreSQL 18 in addition to
the PostgreSQL 16 full-suite run. Ruff, mypy and git diff whitespace checks
passed. CI configuration is updated but its remote run has not been verified.

## PR3 — completed and locally verified

Return optimizer prompt, model configuration and response metadata directly from
PromptRewriteResult. Record Git SHA, seed, model parameters and dataset release.
Extract task adapters from hardcoded question/evaluate_qa/f1 behavior. Validate
both NQ and GSM8K metric systems.

Implemented optional task adapters with backward-compatible QA defaults,
structured inputs/labels, direct rewrite provenance, per-record model/input
snapshots, Git status and dataset fingerprints. See `evolution-v1.md`.
Validation: 99 tests passed, coverage 95.68%, including PostgreSQL trace
round-trip and mocked OpenAI request/response tests. NQ/GSM8K offline fixture
examples passed; no paid inference or full-dataset benchmark was run.

## PR4 — next

Client injection and reuse, timeouts, retries and structured errors; token,
latency and cost accounting; batch/asynchronous evaluation; repeated paired
parent/candidate evaluations and minimum improvement thresholds.
