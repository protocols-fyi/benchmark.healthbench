PRAGMA foreign_keys = ON;

BEGIN;

CREATE TABLE IF NOT EXISTS cases (
    prompt_id TEXT PRIMARY KEY,
    line_number INTEGER NOT NULL UNIQUE CHECK (line_number > 0),
    question TEXT NOT NULL,
    raw_json TEXT NOT NULL CHECK (json_valid(raw_json)),
    ideal_completions_json TEXT CHECK (
        ideal_completions_json IS NULL OR json_valid(ideal_completions_json)
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    backend TEXT NOT NULL,
    checkpoint_path TEXT,
    request_parameters_json TEXT CHECK (
        request_parameters_json IS NULL OR json_valid(request_parameters_json)
    ),
    serving_config_json TEXT CHECK (
        serving_config_json IS NULL OR json_valid(serving_config_json)
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    prompt_id TEXT NOT NULL REFERENCES cases(prompt_id),
    rollout_index INTEGER NOT NULL CHECK (rollout_index >= 0),
    model_response TEXT NOT NULL,
    grader_response_json TEXT NOT NULL CHECK (json_valid(grader_response_json)),
    grader_explanation TEXT,
    criteria_met INTEGER NOT NULL CHECK (criteria_met IN (0, 1)),
    score REAL NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
    input_token_count INTEGER CHECK (
        input_token_count IS NULL OR input_token_count >= 0
    ),
    cached_input_token_count INTEGER CHECK (
        cached_input_token_count IS NULL OR cached_input_token_count >= 0
    ),
    output_token_count INTEGER CHECK (
        output_token_count IS NULL OR output_token_count >= 0
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sessions_model
    ON sessions (model_id, prompt_id, rollout_index);
CREATE INDEX IF NOT EXISTS idx_sessions_prompt
    ON sessions (prompt_id, model_id, rollout_index);
CREATE INDEX IF NOT EXISTS idx_sessions_score
    ON sessions (model_id, score);

DROP VIEW IF EXISTS v_case_rollup;
CREATE VIEW v_case_rollup AS
SELECT
    model_id,
    prompt_id,
    COUNT(*) AS sample_count,
    AVG(score) AS mean_score,
    MIN(score) AS worst_score,
    MAX(score) AS best_score
FROM sessions
GROUP BY model_id, prompt_id;

DROP VIEW IF EXISTS v_model_rollup;
CREATE VIEW v_model_rollup AS
SELECT
    model_id,
    COUNT(*) AS sample_count,
    AVG(score) AS mean_score,
    MIN(score) AS worst_score,
    MAX(score) AS best_score
FROM sessions
GROUP BY model_id;

COMMIT;
