"""Run live PostgreSQL smoke checks for miniEvoAgent storage."""

import json
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evo_repro import PostgreSQLStorage  # noqa: E402


def main() -> None:
    load_dotenv()
    storage = PostgreSQLStorage()
    suffix = uuid4().hex[:12]

    benchmark_id = storage.upsert_benchmark(
        name=f"postgres_smoke_{suffix}",
        task_type="qa",
        description="Live PostgreSQL smoke-test benchmark.",
        metadata={"source": "run_postgres_storage_checks.py"},
    )
    example_id = storage.add_example(
        benchmark_id=benchmark_id,
        split="dev",
        sample_id=f"sample_{suffix}",
        input_data={"question": "What is the capital of France?"},
        label="Paris",
        metadata={"purpose": "smoke"},
    )
    parent_prompt_id = storage.save_prompt(
        content=f"Answer concisely. smoke={suffix}",
        source="smoke",
    )
    candidate_prompt_id = storage.save_prompt(
        content=f"Answer with the shortest correct span. smoke={suffix}",
        source="smoke",
        parent_prompt_id=parent_prompt_id,
    )
    agent_config_id = storage.save_agent_config(
        {
            "name": "qa_agent",
            "action": "answer_question",
            "prompt_id": parent_prompt_id,
        }
    )
    run_id = storage.create_run(
        benchmark_id=benchmark_id,
        run_name=f"postgres_smoke_{suffix}",
        initial_prompt_id=parent_prompt_id,
        score_key="f1",
        config={"agent_config_id": agent_config_id},
    )
    evaluation_id = storage.save_evaluation(
        run_id=run_id,
        prompt_id=parent_prompt_id,
        split="dev",
        aggregate_metrics={"em": 1.0, "f1": 1.0, "acc": 1.0},
    )
    item_id = storage.save_evaluation_item(
        evaluation_id=evaluation_id,
        example_id=example_id,
        prediction="Paris",
        label="Paris",
        metrics={"em": 1.0, "f1": 1.0, "acc": 1.0},
        rendered_prompt="Question: What is the capital of France?",
        metadata={"purpose": "smoke"},
    )
    round_id = storage.save_evolution_round(
        run_id=run_id,
        generation=0,
        parent_prompt_id=parent_prompt_id,
        candidate_prompt_id=candidate_prompt_id,
        selected_prompt_id=candidate_prompt_id,
        parent_score=0.8,
        candidate_score=1.0,
        accepted=True,
        optimizer_prompt="Improve the prompt without memorizing answers.",
        metadata={"purpose": "smoke"},
    )
    storage.finish_run(
        run_id=run_id,
        final_prompt_id=candidate_prompt_id,
        initial_score=0.8,
        final_score=1.0,
    )

    with storage.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    r.status,
                    r.final_score,
                    e.aggregate_metrics_json,
                    ei.metrics_json,
                    er.accepted
                FROM runs r
                JOIN evaluations e ON e.run_id = r.id
                JOIN evaluation_items ei ON ei.evaluation_id = e.id
                JOIN evolution_rounds er ON er.run_id = r.id
                WHERE r.id = %s AND e.id = %s AND ei.id = %s AND er.id = %s;
                """,
                (run_id, evaluation_id, item_id, round_id),
            )
            row = cur.fetchone()

    assert row is not None
    assert _text(row["status"]) == "completed"
    assert row["final_score"] == 1.0
    assert row["aggregate_metrics_json"]["f1"] == 1.0
    assert row["metrics_json"]["em"] == 1.0
    assert row["accepted"] is True

    print(
        json.dumps(
            {
                "status": "passed",
                "benchmark_id": benchmark_id,
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "evolution_round_id": round_id,
            },
            ensure_ascii=False,
        )
    )


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


if __name__ == "__main__":
    main()
