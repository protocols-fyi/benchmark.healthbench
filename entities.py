"""Benchmark entities and aggregate metrics."""

from dataclasses import dataclass, field
from collections.abc import Sequence
import json
from statistics import mean, pstdev
from typing import Any, Literal, NamedTuple

from models.qwen import ChatMessage


class AggregateMetrics(NamedTuple):
    mean: float
    n: int
    stddev: float

    @staticmethod
    def from_scores(scores: Sequence[float]) -> "AggregateMetrics":
        assert scores, "No benchmark metrics collected."
        return AggregateMetrics(mean=mean(scores), n=len(scores), stddev=pstdev(scores))

    def __str__(self) -> str:
        return (
            f"AggregateMetrics(mean={self.mean:.6f}, "
            f"n={self.n}, stddev={self.stddev:.6f})"
        )


def dump_json_value(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        indent=indent,
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class Rubric:
    criterion: str
    points: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Case:
    prompt_id: str
    prompt: tuple[ChatMessage, ...]
    rubric: Rubric
    line_number: int
    raw_json: str
    case_tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def question(self) -> str:
        return "\n\n".join(
            f"{message['role']}: {message['content']}" for message in self.prompt
        )


@dataclass(frozen=True)
class TargetModel:
    model_id: str
    backend: Literal["aws-bedrock", "azure-openai", "vllm"]
    request_parameters_json: str
    checkpoint_path: str | None = None
    serving_config_json: str | None = None
    vllm_base_model_name: str | None = None

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


@dataclass(frozen=True)
class RunCaseResult:
    prompt_id: str
    score: float
    criteria_met: bool
    grader_explanation: str
    grader_response_json: str
    model_response: str
    input_token_count: int | None
    cached_input_token_count: int | None
    output_token_count: int | None
