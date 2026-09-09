"""Real PostgreSQL tests; use only an explicitly supplied disposable test database."""

import os
from uuid import uuid4

import pytest
from psycopg import sql

from evo_repro import Evaluator
from evo_repro.storage import PostgreSQLStorage, PostgresConfig, SCHEMA_SQL
from test_evaluation import FakeBenchmark, build_agent


@pytest.fixture
def storage():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")
    schema = "test_minievo_" + uuid4().hex

    class IsolatedStorage(PostgreSQLStorage):
        def connect(self):
            conn = super().connect()
            conn.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(schema)))
            return conn

    adapter = IsolatedStorage(config=PostgresConfig(database_url=url))
    with adapter.connect() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield adapter
    finally:
        with adapter.connect() as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def rows(storage, query):
    with storage.connect() as conn:
        return conn.execute(query).fetchall()


def evaluate(storage, benchmark=None):
    return Evaluator().evaluate_agent(
        agent=build_agent(), benchmark=benchmark or FakeBenchmark(),
        collate_func=lambda e: {"question": e["question"]}, storage=storage,
    )


def test_fresh_schema_and_repeated_migration(storage):
    storage.initialize()
    storage.initialize()
    assert [r["version"] for r in rows(storage, "SELECT version FROM schema_migrations "
                                     "ORDER BY version")] == [1, 2]
    evaluate(storage)
    assert len(rows(storage, "SELECT * FROM evaluation_items")) == 2


def test_legacy_upgrade_preserves_ids_and_marks_partial_snapshots(storage):
    with storage.connect() as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute("INSERT INTO benchmarks(id,name,task_type) VALUES (1,'legacy','qa')")
        conn.execute("""INSERT INTO examples(id,benchmark_id,split,sample_id,input_json,
                     label_json) VALUES (1,1,'dev','1','{"q":"old"}','"answer"')""")
        conn.execute("""INSERT INTO prompts(id,content,content_hash,source)
                     VALUES (1,'old prompt','legacy-hash','manual')""")
        conn.execute("""INSERT INTO evaluations(id,prompt_id,split,aggregate_metrics_json)
                     VALUES (1,1,'dev','{}')""")
        conn.execute("""INSERT INTO evaluation_items(evaluation_id,example_id,prediction_json,
                     label_json,metrics_json) VALUES (1,1,'"answer"','"answer"','{}')""")
    storage.initialize()
    storage.initialize()
    item = rows(storage, "SELECT * FROM evaluation_items")[0]
    assert item["input_json"] == {"q": "old"}
    assert item["raw_example_json"] is None
    assert item["metadata_json"]["snapshot_provenance"] == "legacy_at_migration"
    assert rows(storage, "SELECT prompt_id FROM evaluations")[0]["prompt_id"] == 1
    assert rows(storage, "SELECT content FROM prompt_contents")[0]["content"] == "old prompt"


def test_historical_inputs_and_fingerprints_survive_sample_changes(storage):
    storage.initialize()
    benchmark = FakeBenchmark()
    first = evaluate(storage, benchmark)
    repeated = evaluate(storage, benchmark)
    assert first.dataset_fingerprint == repeated.dataset_fingerprint
    benchmark.dev[0]["question"] = "Changed question?"
    changed = evaluate(storage, benchmark)
    assert changed.dataset_fingerprint != first.dataset_fingerprint
    items = rows(storage, "SELECT * FROM evaluation_items ORDER BY id")
    assert items[0]["input_json"] == {"question": "Capital of France?"}
    assert items[0]["raw_example_json"]["answer"] == "Paris"
    assert items[4]["input_json"] == {"question": "Changed question?"}
    assert items[0]["example_id"] != items[4]["example_id"]
    assert len(rows(storage, "SELECT * FROM examples")) == 3


def test_same_text_has_distinct_prompt_occurrences_and_lineage(storage):
    storage.initialize()
    parent = storage.save_prompt(content="same", source="manual")
    child = storage.save_prompt(content="same", source="optimizer",
                                parent_prompt_id=parent, metadata={"generation": 2})
    assert parent != child
    assert len(rows(storage, "SELECT * FROM prompt_contents")) == 1
    prompts = rows(storage, "SELECT * FROM prompts ORDER BY id")
    assert prompts[1]["parent_prompt_id"] == parent
    assert prompts[1]["source"] == "optimizer"
    assert prompts[1]["metadata_json"] == {"generation": 2}
    run = storage.create_run(initial_prompt_id=parent)
    storage.finish_run(run_id=run, final_prompt_id=child)
    assert rows(storage, "SELECT final_prompt_id FROM runs")[0]["final_prompt_id"] == child


def test_mid_save_failure_rolls_back_all_evaluation_writes(storage):
    storage.initialize()
    # A real database constraint rejects only the second item, after earlier writes.
    with storage.connect() as conn:
        conn.execute("ALTER TABLE evaluation_items ADD CHECK (sample_id <> 'dev-2')")
    with pytest.raises(Exception, match="check constraint"):
        evaluate(storage)
    for table in ("benchmarks", "examples", "evaluations", "evaluation_items"):
        assert rows(storage, f"SELECT * FROM {table}") == []


def test_failed_evaluation_never_starts_persistence(storage):
    storage.initialize()

    class BrokenBenchmark(FakeBenchmark):
        def evaluate(self, prediction, label):
            if label == "Jupiter":
                raise RuntimeError("second example failed")
            return {"f1": 1.0}

    with pytest.raises(RuntimeError, match="second example failed"):
        evaluate(storage, BrokenBenchmark())
    assert rows(storage, "SELECT * FROM benchmarks") == []


def test_duplicate_sample_ids_keep_their_own_input_links(storage):
    storage.initialize()
    benchmark = FakeBenchmark()
    benchmark.dev[1]["id"] = benchmark.dev[0]["id"]
    evaluate(storage, benchmark)
    items = rows(storage, "SELECT * FROM evaluation_items ORDER BY id")
    assert items[0]["example_id"] != items[1]["example_id"]


def test_migration_failure_is_atomic(storage, monkeypatch):
    import evo_repro.storage as module

    monkeypatch.setattr(module, "MIGRATIONS", ((2, "CREATE TABLE partial(id INT); "
                                                  "SELECT missing_column"),))
    with pytest.raises(Exception, match="missing_column"):
        storage.initialize()
    assert rows(storage, "SELECT to_regclass('partial') AS name")[0]["name"] is None
    assert rows(storage, "SELECT to_regclass('schema_migrations') AS name")[0]["name"] is None


def test_failure_preserves_previous_evaluation_and_benchmark(storage):
    storage.initialize()
    evaluate(storage)
    before = rows(storage, "SELECT * FROM benchmarks")
    with storage.connect() as conn:
        conn.execute("ALTER TABLE evaluation_items ADD CHECK "
                     "(sample_id <> 'new-failure')")
    benchmark = FakeBenchmark()
    benchmark.description = "should roll back"
    benchmark.dev[1]["id"] = "new-failure"
    with pytest.raises(Exception, match="check constraint"):
        evaluate(storage, benchmark)
    assert rows(storage, "SELECT * FROM benchmarks") == before
    assert len(rows(storage, "SELECT * FROM evaluations")) == 1
    assert len(rows(storage, "SELECT * FROM evaluation_items")) == 2


def test_low_level_storage_operations(storage):
    storage.initialize()
    benchmark_id = storage.upsert_benchmark(name="manual", task_type="qa")
    assert storage.get_benchmark_by_name("manual")["id"] == benchmark_id
    config_id = storage.save_agent_config({"name": "test"})
    assert storage.save_agent_config({"name": "test"}) == config_id
    parent = storage.save_prompt(content="parent")
    child = storage.save_prompt(content="child", parent_prompt_id=parent)
    run = storage.create_run(benchmark_id=benchmark_id, initial_prompt_id=parent)
    round_id = storage.save_evolution_round(
        run_id=run, generation=0, parent_prompt_id=parent, candidate_prompt_id=child,
        selected_prompt_id=child, parent_score=0.0, candidate_score=1.0, accepted=True,
    )
    assert rows(storage, "SELECT id FROM evolution_rounds")[0]["id"] == round_id
    storage.finish_run(run_id=run, final_prompt_id=child, final_score=1.0)
    assert rows(storage, "SELECT status FROM runs")[0]["status"] == "completed"


def test_newer_schema_is_rejected(storage):
    storage.initialize()
    with storage.connect() as conn:
        conn.execute("INSERT INTO schema_migrations(version) VALUES (999)")
    with pytest.raises(RuntimeError, match="newer"):
        storage.initialize()


def test_serialization_failure_rolls_back(storage):
    storage.initialize()
    result = Evaluator().evaluate_agent(
        agent=build_agent(), benchmark=FakeBenchmark(),
        collate_func=lambda e: {"question": e["question"]},
    )
    result.records[1].prediction = object()
    with pytest.raises(TypeError, match="JSON serializable"):
        storage.save_evaluation_result(result=result, task_type="qa")
    assert rows(storage, "SELECT * FROM benchmarks") == []
    assert rows(storage, "SELECT * FROM evaluation_items") == []


def test_evolution_trace_roundtrips_in_postgres_metadata(storage):
    from evo_repro import EvolutionRoundTrace, GSM8KTask
    from test_evolution_trace import run

    storage.initialize()
    trace = run(GSM8KTask(input_key="problem"), "#### 42")
    parent = storage.save_prompt(content=trace.parent_prompt)
    child = storage.save_prompt(content=trace.candidate_prompt, parent_prompt_id=parent)
    run_id = storage.create_run(initial_prompt_id=parent, score_key=trace.score_key)
    storage.save_evolution_round(
        run_id=run_id, generation=trace.generation, parent_prompt_id=parent,
        candidate_prompt_id=child, selected_prompt_id=child,
        parent_score=trace.parent_dev_score, candidate_score=trace.candidate_dev_score,
        accepted=trace.selection.accepted,
        optimizer_prompt=trace.rewrite_input.optimizer_meta_prompt,
        metadata=trace.to_dict(mode="json"),
    )
    saved = rows(storage, "SELECT metadata_json, optimizer_prompt FROM evolution_rounds")[0]
    assert EvolutionRoundTrace.from_dict(saved["metadata_json"]) == trace
    assert saved["optimizer_prompt"] == trace.rewrite_input.optimizer_meta_prompt
