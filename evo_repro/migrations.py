"""Ordered, transactional upgrades from the original experiment schema."""

MIGRATIONS = (
    (2, """
CREATE TABLE prompt_contents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE
);
INSERT INTO prompt_contents (content, content_hash)
SELECT content, content_hash FROM prompts;
ALTER TABLE prompts ADD COLUMN content_id BIGINT REFERENCES prompt_contents(id);
UPDATE prompts p SET content_id = c.id FROM prompt_contents c
WHERE p.content_hash = c.content_hash;
ALTER TABLE prompts ALTER COLUMN content_id SET NOT NULL;
ALTER TABLE prompts DROP CONSTRAINT prompts_content_hash_key;
CREATE INDEX idx_prompts_content ON prompts(content_id);

ALTER TABLE examples DROP CONSTRAINT examples_benchmark_id_split_sample_id_key;
ALTER TABLE examples ADD COLUMN snapshot_hash TEXT;
CREATE UNIQUE INDEX idx_examples_snapshot
ON examples(benchmark_id, split, sample_id, snapshot_hash);

ALTER TABLE evaluations ADD COLUMN dataset_fingerprint TEXT;
ALTER TABLE evaluation_items ADD COLUMN raw_example_json JSONB;
ALTER TABLE evaluation_items ADD COLUMN input_json JSONB;
ALTER TABLE evaluation_items ADD COLUMN sample_id TEXT;
-- Old raw examples cannot be reconstructed. Preserve only what is actually known.
UPDATE evaluation_items i SET input_json = e.input_json, sample_id = e.sample_id,
    metadata_json = i.metadata_json || '{"snapshot_provenance":"legacy_at_migration"}'::jsonb
FROM examples e WHERE i.example_id = e.id;
"""),
)
