"""SQLite storage for benchmark inputs and sessions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import field_validator

from entities import StrictModel
from validation import (
    require_float_in_range,
    require_int_choice,
    require_non_negative_int,
    require_positive_int,
    strip_and_require_non_empty,
    strip_optional_non_empty,
    strip_string,
)

SCHEMA_PATH = Path(__file__).resolve().with_name("db.sql")


class StoredCase(StrictModel):
    prompt_id: str
    line_number: int
    question: str
    raw_json: str
    ideal_completions_json: str | None

    @field_validator("prompt_id", "question", "raw_json")
    @classmethod
    def validate_stored_case_strings(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="stored case string fields must be non-empty.",
        )

    @field_validator("line_number")
    @classmethod
    def validate_stored_case_line_number(cls, value: int) -> int:
        return require_positive_int(
            value,
            message="line_number must be > 0.",
        )


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
    def validate_stored_model_strings(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="stored model string fields must be non-empty.",
        )

    @field_validator(
        "checkpoint_path",
        "request_parameters_json",
        "serving_config_json",
    )
    @classmethod
    def validate_stored_model_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        return strip_optional_non_empty(
            value,
            message="stored model optional string fields must be non-empty when set.",
        )


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
    def validate_stored_session_strings(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="stored session string fields must be non-empty.",
        )

    @field_validator("grader_explanation")
    @classmethod
    def normalize_grader_explanation(cls, value: str) -> str:
        return strip_string(value)

    @field_validator("rollout_index")
    @classmethod
    def validate_session_rollout_index(cls, value: int) -> int:
        return require_non_negative_int(
            value,
            message="rollout_index must be >= 0.",
        )

    @field_validator("criteria_met")
    @classmethod
    def validate_session_criteria_met(cls, value: int) -> int:
        return require_int_choice(
            value,
            choices=frozenset({0, 1}),
            message="criteria_met must be 0 or 1.",
        )

    @field_validator("score")
    @classmethod
    def validate_session_score(cls, value: float) -> float:
        return require_float_in_range(
            value,
            minimum=0.0,
            maximum=1.0,
            message="score must be in [0, 1].",
        )

    @field_validator(
        "input_token_count",
        "cached_input_token_count",
        "output_token_count",
    )
    @classmethod
    def validate_session_token_counts(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return require_non_negative_int(
            value,
            message="token counts must be non-negative when set.",
        )


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
