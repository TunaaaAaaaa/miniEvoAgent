## Experiment storage v2

PR2 adds versioned migrations, immutable input snapshots, prompt occurrences,
and atomic evaluation persistence. Run `python scripts/init_postgres_storage.py`
before writing with the new code. `PostgreSQLStorage.initialize()` upgrades both
fresh databases and the original unversioned schema. It takes a transaction-level
advisory lock, records completed versions in `schema_migrations`, and rolls back
the entire upgrade on failure. Repeated initialization is safe. Back up valuable
databases before upgrading; automatic downgrade is not provided.

## Historical data

`examples` now identifies a version by benchmark, split, sample ID and snapshot
hash. Reusing a sample ID with different input, label or metadata creates another
row instead of changing an existing one. Each new `evaluation_items` row also
stores its own `sample_id`, `raw_example_json` and `input_json`, alongside its
label and prediction. Raw examples are copied before collation, and inputs are
copied before agent execution so in-place mutations do not change the snapshots.

`evaluations.dataset_fingerprint` is SHA-256 of canonical, sorted-key JSON of the
benchmark name, split, and ordered evaluated records (sample ID, raw example,
collated input and label). It identifies the evaluated subset, not the entire
upstream dataset or its release version. Changes to selection, order or content
change the fingerprint. JSON persistence rejects unsupported objects and NaN or
infinity rather than silently converting them to strings.

Migration retains old IDs and relationships. Historical collated input is copied
from the example available at migration time and marked
`snapshot_provenance=legacy_at_migration`. Previously overwritten inputs cannot
be recovered. Old raw examples and dataset fingerprints stay SQL NULL because
the old schema did not record them.

## Prompt identity

`prompt_contents` deduplicates text by SHA-256. `prompts` now represents individual
occurrences: every `save_prompt()` call returns a new ID with its own source,
parent occurrence and metadata, even for identical text. Existing run, evaluation
and evolution-round foreign keys continue to reference `prompts` IDs. Legacy
content columns remain on `prompts` for existing readers; new content identity is
`content_id`. Callers that relied on repeated saves returning the same prompt ID
must instead compare content hashes or retain and reuse the returned occurrence ID.

## Transaction contract

`Evaluator` finishes inference and aggregation before calling
`storage.save_evaluation_result(...)` once. This method writes the benchmark,
evaluation, example versions and items on one connection in one transaction.
Any error rolls back all those writes, including benchmark updates. No database
transaction is held open during model calls. Duplicate sample IDs do not collapse
the per-record example links.

Custom `EvaluationStorage` adapters must implement `save_evaluation_result`
instead of the former four per-row methods and provide equivalent atomicity.
The low-level PostgreSQL save methods remain available for scripts, but separate
calls are separate transactions. This change does not make an entire multi-round
evolution run atomic.

## Integration tests

Set `TEST_DATABASE_URL` to a disposable PostgreSQL database, then run:

```powershell
$env:TEST_DATABASE_URL = 'postgresql://minievo_test:minievo_test@127.0.0.1:55439/minievo_test'
python -m pytest --cov=evo_repro --cov-report=term-missing --cov-fail-under=85 -q
```

Each integration test creates and drops its own randomly named schema. Without
`TEST_DATABASE_URL` these tests are explicitly skipped; they never use
`DATABASE_URL`. CI supplies a PostgreSQL 16 service for all supported Python
versions. Tests cover legacy and repeated upgrades, transactional migration
failure, historical snapshots, fingerprints, prompt lineage, duplicate sample
IDs, inference failure and a database failure on the second saved item.
