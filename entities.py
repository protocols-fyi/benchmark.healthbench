"""Benchmark entities."""

from dataclasses import dataclass
from typing import Literal

from models.qwen import ChatMessage


@dataclass(frozen=True)
class Case:
    prompt_id: str
    prompt: tuple[ChatMessage, ...]
    raw_json: str

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
        )


@dataclass(frozen=True)
class Grade:
    prompt_id: str
    criteria_met: bool
    grader_explanation: str
    grader_response_json: str
    model_response: str
    input_token_count: int
    cached_input_token_count: int
    output_token_count: int
