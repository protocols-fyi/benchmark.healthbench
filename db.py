"""SQLite storage for benchmark inputs and sessions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from entities import StrictModel

SCHEMA_PATH = Path(__file__).resolve().with_name("db.sql")


class StoredCase(StrictModel):
    prompt_id: str
    line_number: int
    question: str
    raw_json: str
    ideal_completions_json: str | None

    @field_validator("prompt_id", "question", "raw_json")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        assert value, "stored case string fields must be non-empty."
        return value

    @field_validator("line_number")
    @classmethod
    def validate_line_number(cls, value: int) -> int:
        assert value > 0, "line_number must be > 0."
        return value


class StoredModel(StrictModel):
    model_id: str
    display_name: str
    provider: Literal["anthropic", "azure-openai", "local"]
    backend: Literal["aws-bedrock", "azure-openai", "vllm"]
    checkpoint_path: str | None
    request_parameters_json: str | None
    serving_config_json: str | None

    @field_validator("model_id", "display_name")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        value = value.strip()
        assert value, "stored model string fields must be non-empty."
        return value

    @field_validator(
        "checkpoint_path",
        "request_parameters_json",
        "serving_config_json",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        assert value, "stored model optional string fields must be non-empty when set."
        return value


class StoredSession(StrictModel):
    model_id: str
    prompt_id: str
    rollout_index: int
    model_response: str
    grader_response_json: str
    grader_explanation: str
    criteria_met: int
    score: float
    input_token_count: int | None
    cached_input_token_count: int | None
    output_token_count: int | None

    @field_validator(
        "model_id",
        "prompt_id",
        "model_response",
        "grader_response_json",
    )
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        assert value, "stored session string fields must be non-empty."
        return value

    @field_validator("grader_explanation")
    @classmethod
    def validate_grader_explanation(cls, value: str) -> str:
        return value.strip()

    @field_validator("rollout_index")
    @classmethod
    def validate_rollout_index(cls, value: int) -> int:
        assert value >= 0, "rollout_index must be >= 0."
        return value

    @field_validator("criteria_met")
    @classmethod
    def validate_criteria_met(cls, value: int) -> int:
        assert value in {0, 1}, "criteria_met must be 0 or 1."
        return value

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        assert 0.0 <= value <= 1.0, "score must be in [0, 1]."
        return value

    @field_validator(
        "input_token_count",
        "cached_input_token_count",
        "output_token_count",
    )
    @classmethod
    def validate_optional_token_counts(cls, value: int | None) -> int | None:
        if value is None:
            return None
        assert value >= 0, "token counts must be non-negative when set."
        return value


class BenchmarkStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._db_path)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._connection.commit()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        self._connection.close()

    def upsert_case(self, case: StoredCase) -> None:
        self._connection.execute(
            """
            INSERT INTO cases (
                prompt_id,
                line_number,
                question,
                raw_json,
                ideal_completions_json
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(prompt_id) DO UPDATE SET
                line_number = excluded.line_number,
                question = excluded.question,
                raw_json = excluded.raw_json,
                ideal_completions_json = excluded.ideal_completions_json
            """,
            tuple(case.model_dump().values()),
        )

    def upsert_model(self, model: StoredModel) -> None:
        self._connection.execute(
            """
            INSERT INTO models (
                model_id,
                display_name,
                provider,
                backend,
                checkpoint_path,
                request_parameters_json,
                serving_config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id) DO UPDATE SET
                display_name = excluded.display_name,
                provider = excluded.provider,
                backend = excluded.backend,
                checkpoint_path = excluded.checkpoint_path,
                request_parameters_json = excluded.request_parameters_json,
                serving_config_json = excluded.serving_config_json
            """,
            tuple(model.model_dump().values()),
        )

    def insert_session(self, session: StoredSession) -> None:
        self._connection.execute(
            """
            INSERT INTO sessions (
                model_id,
                prompt_id,
                rollout_index,
                model_response,
                grader_response_json,
                grader_explanation,
                criteria_met,
                score,
                input_token_count,
                cached_input_token_count,
                output_token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(session.model_dump().values()),
        )

    def commit(self) -> None:
        self._connection.commit()
