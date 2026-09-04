import json

import pytest

from evo_repro.storage import (
    SCHEMA_SQL,
    PostgresConfig,
    PostgreSQLStorage,
    mask_database_url,
    stable_hash,
)


def test_postgres_config_reads_database_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://user:secret@localhost:5432/minievoagent",
    )

    config = PostgresConfig.from_env()

    assert config.database_url == "postgresql://user:secret@localhost:5432/minievoagent"
    assert config.resolved_database_url == config.database_url


def test_postgres_config_requires_database_url() -> None:
    config = PostgresConfig(database_url=None)

    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        _ = config.resolved_database_url


def test_postgres_config_masks_password_in_config() -> None:
    config = PostgresConfig(
        database_url="postgresql://user:secret@localhost:5432/minievoagent"
    )

    persisted_config = config.to_config()

    assert persisted_config["database_url"] == (
        "postgresql://user:***@localhost:5432/minievoagent"
    )
    assert "secret" not in json.dumps(persisted_config)


def test_mask_database_url_handles_url_without_password() -> None:
    database_url = "postgresql://localhost:5432/minievoagent"

    assert mask_database_url(database_url) == database_url


def test_mask_database_url_redacts_secret_query_parameters() -> None:
    database_url = (
        "postgresql://user:secret@[::1]:5432/minievoagent"
        "?sslpassword=query-secret&application_name=tests"
    )

    masked_url = mask_database_url(database_url)

    assert masked_url == (
        "postgresql://user:***@[::1]:5432/minievoagent"
        "?sslpassword=***&application_name=tests"
    )
    assert "secret" not in masked_url


def test_schema_contains_experiment_storage_tables_and_indexes() -> None:
    required_names = [
        "benchmarks",
        "examples",
        "prompts",
        "agent_configs",
        "runs",
        "evaluations",
        "evaluation_items",
        "evolution_rounds",
        "idx_examples_benchmark_split",
        "idx_runs_benchmark_created",
    ]

    for name in required_names:
        assert name in SCHEMA_SQL


def test_stable_hash_is_deterministic_sha256() -> None:
    value = "Answer questions concisely."

    assert stable_hash(value) == stable_hash(value)
    assert len(stable_hash(value)) == 64


def test_storage_config_is_safe_to_persist() -> None:
    storage = PostgreSQLStorage(
        config=PostgresConfig(
            database_url="postgresql://user:secret@localhost:5432/minievoagent"
        )
    )

    config = storage.to_config()

    assert config["config"]["database_url"] == (
        "postgresql://user:***@localhost:5432/minievoagent"
    )
    assert "secret" not in json.dumps(config)
