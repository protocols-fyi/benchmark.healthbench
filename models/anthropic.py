"""Minimal Anthropic model helpers used by the benchmark runner."""

import asyncio
import logging
import os

from anthropic import AsyncAnthropicBedrock
from anthropic.types import Message

from config import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    DEFAULT_AWS_BEDROCK_REGION,
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_MODEL_ASK_QUESTION,
)

logger = logging.getLogger(__name__)

AWS_BEDROCK_BENCHMARK_REQUEST_PARAMETERS = {
    "temperature": 1.0,
    "max_tokens": 20_000,
    "thinking": {
        "type": "enabled",
        "budget_tokens": 16_000,
    },
}


def _resolve_bedrock_credentials() -> tuple[str | None, str | None, str | None]:
    access_key = os.environ.get("AWS_BEDROCK_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_BEDROCK_SECRET_ACCESS_KEY", "").strip()
    session_token = os.environ.get("AWS_BEDROCK_SESSION_TOKEN", "").strip() or None
    assert bool(access_key) == bool(secret_key), (
        "Set both AWS_BEDROCK_ACCESS_KEY_ID and AWS_BEDROCK_SECRET_ACCESS_KEY, "
        "or neither."
    )
    if access_key and secret_key:
        return access_key, secret_key, session_token
    return None, None, None


def _extract_text_blocks(message: Message) -> str:
    visible_blocks: list[str] = []
    for block in message.content:
        block_type = getattr(block, "type", "")
        if block_type != "text":
            continue
        text = getattr(block, "text", "")
        if isinstance(text, str) and text.strip():
            visible_blocks.append(text.strip())
    visible_text = "\n\n".join(visible_blocks).strip()
    assert visible_text, "Bedrock response did not contain any visible text blocks."
    return visible_text


def _resolve_region() -> str:
    resolved_region = (
        os.environ.get("AWS_REGION", "").strip()
        or os.environ.get("AWS_DEFAULT_REGION", "").strip()
        or DEFAULT_AWS_BEDROCK_REGION
    )
    return resolved_region


async def ask(
    *,
    model_name: str,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = "",
    max_tokens: int = AWS_BEDROCK_BENCHMARK_REQUEST_PARAMETERS["max_tokens"],
) -> tuple[str, int, int, int]:
    assert model_name in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
        f"Unsupported Bedrock model: {model_name}"
    )
    resolved_model_id = AWS_BEDROCK_SUPPORTED_MODEL_IDS[model_name]
    normalized_prompt = prompt.strip()
    if messages is None:
        assert normalized_prompt, (
            "prompt must be non-empty when messages are not provided."
        )
        request_messages = [{"role": "user", "content": normalized_prompt}]
    else:
        assert not normalized_prompt, (
            "prompt must be empty when messages are provided."
        )
        assert messages, "messages must be non-empty."
        request_messages = []
        for message in messages:
            role = str(message["role"]).strip()
            content = str(message["content"]).strip()
            assert role in {"user", "assistant"}, (
                "Bedrock messages must use only user/assistant roles."
            )
            assert content, "Bedrock message content must be non-empty."
            request_messages.append({"role": role, "content": content})

    access_key, secret_key, session_token = _resolve_bedrock_credentials()
    async with AsyncAnthropicBedrock(
        aws_access_key=access_key,
        aws_region=_resolve_region(),
        aws_secret_key=secret_key,
        aws_session_token=session_token,
        max_retries=0,
        timeout=DEFAULT_GENERATION_TIMEOUT_SECONDS,
    ) as client:
        request_kwargs: dict[str, object] = {
            "model": resolved_model_id,
            "max_tokens": max_tokens,
            "messages": request_messages,
            "temperature": AWS_BEDROCK_BENCHMARK_REQUEST_PARAMETERS["temperature"],
            "thinking": AWS_BEDROCK_BENCHMARK_REQUEST_PARAMETERS["thinking"],
        }
        response = await client.messages.create(**request_kwargs)

    logger.info(
        "Completed Bedrock request | model=%s | resolved_bedrock_model_id=%s",
        model_name,
        response.model,
    )
    cached_input_token_count = response.usage.cache_read_input_tokens or 0
    input_token_count = (
        response.usage.input_tokens
        + (response.usage.cache_creation_input_tokens or 0)
        + cached_input_token_count
    )
    return (
        _extract_text_blocks(response),
        input_token_count,
        cached_input_token_count,
        response.usage.output_tokens,
    )


async def _main() -> None:
    answer, _, _, _ = await ask(
        model_name=DEFAULT_BEDROCK_MODEL,
        prompt=DEFAULT_MODEL_ASK_QUESTION,
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(_main())
