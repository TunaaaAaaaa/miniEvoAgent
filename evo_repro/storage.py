import hashlib
import json
import os
from typing import Any

from pydantic import Field

from .base import BaseModel
from .urls import redact_url_secrets


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS benchmarks (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    task_type TEXT NOT NULL,
    description TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS examples (
    id BIGSERIAL PRIMARY KEY,
    benchmark_id BIGINT NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
    split TEXT NOT NULL CHECK (split IN ('train', 'dev', 'test')),
    sample_id TEXT NOT NULL,
    input_json JSONB NOT NULL,
    label_json JSONB NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (benchmark_id, split, sample_id)
);

CREATE TABLE IF NOT EXISTS prompts (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    parent_prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_configs (
    id BIGSERIAL PRIMARY KEY,
    config_json JSONB NOT NULL,
    config_hash TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
    id BIGSERIAL PRIMARY KEY,
    benchmark_id BIGINT REFERENCES benchmarks(id) ON DELETE SET NULL,
    run_name TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    initial_prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    final_prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    score_key TEXT NOT NULL DEFAULT 'f1',
    initial_score DOUBLE PRECISION,
    final_score DOUBLE PRECISION,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS evaluations (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT REFERENCES runs(id) ON DELETE CASCADE,
    prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'dev', 'test')),
    aggregate_metrics_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evaluation_items (
    id BIGSERIAL PRIMARY KEY,
    evaluation_id BIGINT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    example_id BIGINT REFERENCES examples(id) ON DELETE SET NULL,
    prediction_json JSONB NOT NULL,
    label_json JSONB NOT NULL,
    metrics_json JSONB NOT NULL,
    rendered_prompt TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evolution_rounds (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    generation INTEGER NOT NULL,
    parent_prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    candidate_prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    selected_prompt_id BIGINT REFERENCES prompts(id) ON DELETE SET NULL,
    parent_score DOUBLE PRECISION NOT NULL,
    candidate_score DOUBLE PRECISION NOT NULL,
    accepted BOOLEAN NOT NULL,
    optimizer_prompt TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, generation)
);

CREATE INDEX IF NOT EXISTS idx_examples_benchmark_split
    ON examples (benchmark_id, split);
CREATE INDEX IF NOT EXISTS idx_runs_benchmark_created
    ON runs (benchmark_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evaluation_items_example
    ON evaluation_items (example_id);
CREATE INDEX IF NOT EXISTS idx_evolution_rounds_run_generation
    ON evolution_rounds (run_id, generation);
"""


class PostgresConfig(BaseModel):
    """Connection settings for PostgreSQL-backed experiment storage."""

    database_url: str | None = None
    connect_timeout: int = 10
    application_name: str = "miniEvoAgent"

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        return cls(database_url=os.getenv("DATABASE_URL"))

    @property
    def resolved_database_url(self) -> str:
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for PostgreSQL storage.")
        return self.database_url

    def to_config(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "version": self.version,
            "database_url": mask_database_url(self.database_url),
            "connect_timeout": self.connect_timeout,
            "application_name": self.application_name,
        }


class PostgreSQLStorage(BaseModel):
    """Small PostgreSQL adapter for benchmarks, runs, prompts, and evaluations."""

    config: PostgresConfig = Field(default_factory=PostgresConfig.from_env)

    def to_config(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "version": self.version,
            "config": self.config.to_config(),
        }

    def connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ImportError(
                "Install psycopg[binary] to use PostgreSQLStorage."
            ) from exc

        return psycopg.connect(
            self.config.resolved_database_url,
            connect_timeout=self.config.connect_timeout,
            application_name=self.config.application_name,
            row_factory=dict_row,
        )

    def initialize(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
            conn.commit()

    def upsert_benchmark(
        self,
        *,
        name: str,
        task_type: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
        INSERT INTO benchmarks (name, task_type, description, metadata_json)
        VALUES (%s, %s, %s, %s::jsonb)
        ON CONFLICT (name) DO UPDATE SET
            task_type = EXCLUDED.task_type,
            description = EXCLUDED.description,
            metadata_json = EXCLUDED.metadata_json
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (name, task_type, description, _json(metadata or {})),
        )

    def get_benchmark_by_name(self, name: str) -> dict[str, Any] | None:
        query = """
        SELECT id, name, task_type, description, metadata_json, created_at
        FROM benchmarks
        WHERE name = %s;
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (name,))
                return cur.fetchone()

    def add_example(
        self,
        *,
        benchmark_id: int,
        split: str,
        sample_id: str,
        input_data: Any,
        label: Any,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
        INSERT INTO examples (
            benchmark_id, split, sample_id, input_json, label_json, metadata_json
        )
        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
        ON CONFLICT (benchmark_id, split, sample_id) DO UPDATE SET
            input_json = EXCLUDED.input_json,
            label_json = EXCLUDED.label_json,
            metadata_json = EXCLUDED.metadata_json
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (
                benchmark_id,
                split,
                sample_id,
                _json(input_data),
                _json(label),
                _json(metadata or {}),
            ),
        )

    def save_prompt(
        self,
        *,
        content: str,
        source: str = "manual",
        parent_prompt_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        content_hash = stable_hash(content)
        query = """
        INSERT INTO prompts (
            content, content_hash, source, parent_prompt_id, metadata_json
        )
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (content_hash) DO UPDATE SET
            content = EXCLUDED.content
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (content, content_hash, source, parent_prompt_id, _json(metadata or {})),
        )

    def save_agent_config(self, config: dict[str, Any]) -> int:
        config_json = _json(config)
        config_hash = stable_hash(config_json)
        query = """
        INSERT INTO agent_configs (config_json, config_hash)
        VALUES (%s::jsonb, %s)
        ON CONFLICT (config_hash) DO UPDATE SET
            config_json = EXCLUDED.config_json
        RETURNING id;
        """
        return self._fetch_id(query, (config_json, config_hash))

    def create_run(
        self,
        *,
        benchmark_id: int | None = None,
        run_name: str | None = None,
        initial_prompt_id: int | None = None,
        score_key: str = "f1",
        config: dict[str, Any] | None = None,
    ) -> int:
        query = """
        INSERT INTO runs (
            benchmark_id, run_name, initial_prompt_id, score_key, config_json
        )
        VALUES (%s, %s, %s, %s, %s::jsonb)
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (
                benchmark_id,
                run_name,
                initial_prompt_id,
                score_key,
                _json(config or {}),
            ),
        )

    def finish_run(
        self,
        *,
        run_id: int,
        final_prompt_id: int | None = None,
        initial_score: float | None = None,
        final_score: float | None = None,
        status: str = "completed",
    ) -> None:
        query = """
        UPDATE runs
        SET status = %s,
            final_prompt_id = %s,
            initial_score = %s,
            final_score = %s,
            finished_at = NOW()
        WHERE id = %s;
        """
        self._execute(query, (status, final_prompt_id, initial_score, final_score, run_id))

    def save_evaluation(
        self,
        *,
        run_id: int | None,
        prompt_id: int | None,
        split: str,
        aggregate_metrics: dict[str, float],
    ) -> int:
        query = """
        INSERT INTO evaluations (
            run_id, prompt_id, split, aggregate_metrics_json
        )
        VALUES (%s, %s, %s, %s::jsonb)
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (run_id, prompt_id, split, _json(aggregate_metrics)),
        )

    def save_evaluation_item(
        self,
        *,
        evaluation_id: int,
        example_id: int | None,
        prediction: Any,
        label: Any,
        metrics: dict[str, float],
        rendered_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
        INSERT INTO evaluation_items (
            evaluation_id,
            example_id,
            prediction_json,
            label_json,
            metrics_json,
            rendered_prompt,
            metadata_json
        )
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s::jsonb)
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (
                evaluation_id,
                example_id,
                _json(prediction),
                _json(label),
                _json(metrics),
                rendered_prompt,
                _json(metadata or {}),
            ),
        )

    def save_evolution_round(
        self,
        *,
        run_id: int,
        generation: int,
        parent_prompt_id: int | None,
        candidate_prompt_id: int | None,
        selected_prompt_id: int | None,
        parent_score: float,
        candidate_score: float,
        accepted: bool,
        optimizer_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        query = """
        INSERT INTO evolution_rounds (
            run_id,
            generation,
            parent_prompt_id,
            candidate_prompt_id,
            selected_prompt_id,
            parent_score,
            candidate_score,
            accepted,
            optimizer_prompt,
            metadata_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (run_id, generation) DO UPDATE SET
            parent_prompt_id = EXCLUDED.parent_prompt_id,
            candidate_prompt_id = EXCLUDED.candidate_prompt_id,
            selected_prompt_id = EXCLUDED.selected_prompt_id,
            parent_score = EXCLUDED.parent_score,
            candidate_score = EXCLUDED.candidate_score,
            accepted = EXCLUDED.accepted,
            optimizer_prompt = EXCLUDED.optimizer_prompt,
            metadata_json = EXCLUDED.metadata_json
        RETURNING id;
        """
        return self._fetch_id(
            query,
            (
                run_id,
                generation,
                parent_prompt_id,
                candidate_prompt_id,
                selected_prompt_id,
                parent_score,
                candidate_score,
                accepted,
                optimizer_prompt,
                _json(metadata or {}),
            ),
        )

    def _fetch_id(self, query: str, params: tuple[Any, ...]) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
            conn.commit()
        if row is None:
            raise RuntimeError("PostgreSQL query did not return an id.")
        return int(row["id"])

    def _execute(self, query: str, params: tuple[Any, ...]) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
            conn.commit()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mask_database_url(database_url: str | None) -> str | None:
    if not database_url:
        return None
    return redact_url_secrets(database_url)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
