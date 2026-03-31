"""Benchmark entities and aggregate metrics."""

from collections.abc import Sequence
from statistics import mean, pstdev
from typing import Any, Literal, NamedTuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from models.qwen import ChatMessage
from validation import (
    normalize_string_tuple,
    require_non_zero_float,
    strip_and_require_non_empty,
    strip_optional_non_empty,
)

JSON_VALUE_ADAPTER = TypeAdapter(Any)


class AggregateMetrics(NamedTuple):
    mean: float
    n: int
    stddev: float

    @staticmethod
    def from_scores(scores: Sequence[float]) -> "AggregateMetrics":
        assert scores, "No HB rubrics metrics collected."
        return AggregateMetrics(mean=mean(scores), n=len(scores), stddev=pstdev(scores))

    def __str__(self) -> str:
        return (
            f"AggregateMetrics(mean={self.mean:.6f}, "
            f"n={self.n}, stddev={self.stddev:.6f})"
        )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def dump_json_value(value: Any, *, indent: int | None = None) -> str:
    return JSON_VALUE_ADAPTER.dump_json(
        value,
        indent=indent,
        ensure_ascii=False,
    ).decode("utf-8")


class Rubric(StrictModel):
    criterion: str
    points: float
    tags: tuple[str, ...] = ()

    @field_validator("criterion")
    @classmethod
    def validate_rubric_criterion(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="rubric criterion must be non-empty.",
        )

    @field_validator("points")
    @classmethod
    def validate_rubric_points(cls, value: float) -> float:
        return require_non_zero_float(
            value,
            message="rubric points must be non-zero.",
        )

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_rubric_tags(cls, value: object) -> tuple[str, ...]:
        return normalize_string_tuple(
            value,
            message="rubric tags must be a sequence.",
        )


class Case(StrictModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    prompt_id: str
    prompt: tuple[ChatMessage, ...]
    case_tags: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias="example_tags",
    )
    rubrics: tuple[Rubric, ...]

    @field_validator("prompt_id")
    @classmethod
    def validate_case_prompt_id(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="prompt_id must be non-empty.",
        )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        assert value, "prompt must be non-empty."
        assert value[-1]["role"] == "user", "prompt must end with a user turn."
        for message in value:
            assert message["role"] != "system", (
                "prompt messages must not use the system role."
            )
            assert message["content"].strip(), "message content must be non-empty."
            assert "choice" not in message, (
                "prompt messages must not include completion choices."
            )
        return value

    @field_validator("case_tags", mode="before")
    @classmethod
    def normalize_case_tags_tuple(cls, value: object) -> tuple[str, ...]:
        return normalize_string_tuple(
            value,
            message="case tags must be a sequence.",
        )

    @field_validator("rubrics")
    @classmethod
    def validate_rubrics(cls, value: tuple[Rubric, ...]) -> tuple[Rubric, ...]:
        assert value, "rubrics must be non-empty."
        return value


class CaseRecord(Case):
    ideal_completions_data: Any | None = None


class LoadedCase(StrictModel):
    case: Case
    line_number: int
    raw_json: str
    ideal_completions_json: str | None
    question: str


class TargetModel(StrictModel):
    model_id: str
    backend: Literal["aws-bedrock", "azure-openai", "vllm"]
    request_parameters_json: str
    checkpoint_path: str | None = None
    serving_config_json: str | None = None
    vllm_base_model_name: str | None = None

    @field_validator(
        "model_id",
        "request_parameters_json",
    )
    @classmethod
    def validate_target_model_required_strings(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="target model fields must be non-empty.",
        )

    @field_validator("vllm_base_model_name")
    @classmethod
    def validate_target_model_base_name(cls, value: str | None) -> str | None:
        return strip_optional_non_empty(
            value,
            message="vllm_base_model_name must be non-empty when set.",
        )

    @field_validator("checkpoint_path", "serving_config_json")
    @classmethod
    def validate_target_model_optional_strings(
        cls,
        value: str | None,
    ) -> str | None:
        return strip_optional_non_empty(
            value,
            message="target model optional string fields must be non-empty when set.",
        )

    @model_validator(mode="after")
    def validate_backend_fields(self) -> "TargetModel":
        if self.backend == "vllm":
            assert self.vllm_base_model_name is not None, (
                "vLLM targets must include vllm_base_model_name."
            )
            assert self.serving_config_json is not None, (
                "vLLM targets must include serving_config_json."
            )
        else:
            assert self.vllm_base_model_name is None, (
                "vllm_base_model_name is only valid for vLLM targets."
            )
            assert self.checkpoint_path is None, (
                "checkpoint_path is only valid for vLLM targets."
            )
        if self.backend == "aws-bedrock":
            assert self.serving_config_json is None, (
                "aws-bedrock targets must not include serving_config_json."
            )
        if self.backend == "azure-openai":
            assert self.serving_config_json is not None, (
                "azure-openai targets must include serving_config_json."
            )
        return self

    @property
    def display_name(self) -> str:
        return self.model_id

    @property
    def provider(self) -> str:
        return {
            "aws-bedrock": "anthropic",
            "azure-openai": "azure-openai",
            "vllm": "local",
        }[self.backend]

    @classmethod
    def aws_bedrock(
        cls,
        *,
        model_id: str,
        request_parameters_json: str,
    ) -> "TargetModel":
        return cls(
            model_id=model_id,
            backend="aws-bedrock",
            request_parameters_json=request_parameters_json,
        )

    @classmethod
    def azure_openai(
        cls,
        *,
        model_id: str,
        request_parameters_json: str,
        serving_config_json: str,
    ) -> "TargetModel":
        return cls(
            model_id=model_id,
            backend="azure-openai",
            request_parameters_json=request_parameters_json,
            serving_config_json=serving_config_json,
        )

    @classmethod
    def vllm(
        cls,
        *,
        model_id: str,
        base_model_name: str,
        checkpoint_path: str | None,
        request_parameters_json: str,
        serving_config_json: str,
    ) -> "TargetModel":
        return cls(
            model_id=model_id,
            backend="vllm",
            request_parameters_json=request_parameters_json,
            checkpoint_path=checkpoint_path,
            serving_config_json=serving_config_json,
            vllm_base_model_name=base_model_name,
        )


class RunCaseResult(StrictModel):
    prompt_id: str
    score: float
    criteria_met: bool
    grader_explanation: str
    grader_response_json: str
    model_response: str
    input_token_count: int | None
    cached_input_token_count: int | None
    output_token_count: int | None

    @field_validator(
        "prompt_id",
        "grader_explanation",
        "grader_response_json",
        "model_response",
    )
    @classmethod
    def validate_run_case_result_strings(cls, value: str) -> str:
        return strip_and_require_non_empty(
            value,
            message="run case result string fields must be non-empty.",
        )
