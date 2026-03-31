"""Minimal Anthropic model helpers used by the benchmark runner."""

import asyncio
import logging
import os

from anthropic import AsyncAnthropicBedrock
from anthropic.types import Message

logger = logging.getLogger(__name__)
MAIN_QUESTION = "what's the main argument of Sutton's Bitter Lessons?"

AWS_BEDROCK_SUPPORTED_MODEL_IDS = {
    "anthropic-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic-sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
    "anthropic-opus-4.6": "us.anthropic.claude-opus-4-6-v1",
}
NOISY_LOGGERS = (
    "anthropic",
    "anthropic._base_client",
    "botocore",
    "botocore.auth",
    "botocore.credentials",
    "botocore.hooks",
    "botocore.session",
    "botocore.utils",
)


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


def _resolve_region(region: str | None) -> str:
    resolved_region = (
        region
        or os.environ.get("AWS_REGION", "").strip()
        or os.environ.get("AWS_DEFAULT_REGION", "").strip()
    )
    assert resolved_region, "Set AWS_REGION or AWS_DEFAULT_REGION."
    return resolved_region


async def ask_bedrock(
    *,
    model_name: str,
    region: str | None = None,
    profile: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout_seconds: float = 120.0,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = "",
    top_p: float | None = None,
) -> str:
    assert model_name in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
        f"Unsupported Bedrock model: {model_name}"
    )
    resolved_model_id = AWS_BEDROCK_SUPPORTED_MODEL_IDS[model_name]
    resolved_region = _resolve_region(region.strip() if region is not None else None)
    normalized_prompt = prompt.strip()
    normalized_system_prompt = system_prompt.strip()
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
        aws_profile=profile,
        aws_region=resolved_region,
        aws_secret_key=secret_key,
        aws_session_token=session_token,
        max_retries=0,
        timeout=timeout_seconds,
    ) as client:
        request_kwargs: dict[str, object] = {
            "model": resolved_model_id,
            "max_tokens": max_tokens,
            "messages": request_messages,
            "temperature": temperature,
        }
        if normalized_system_prompt:
            request_kwargs["system"] = normalized_system_prompt
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        response = await client.messages.create(**request_kwargs)
    logger.info(
        "Completed Bedrock request | model=%s | resolved_bedrock_model_id=%s",
        model_name,
        response.model,
    )
    return _extract_text_blocks(response)


ask = ask_bedrock


async def _main() -> None:
    answer = await ask(
        model_name="anthropic-haiku-4.5",
        max_tokens=512,
        temperature=0.0,
        timeout_seconds=120.0,
        prompt=MAIN_QUESTION,
        system_prompt="You are a helpful assistant.",
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(_main())
