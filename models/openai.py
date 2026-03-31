"""Azure OpenAI helpers shared by the benchmark runner and grader."""

import asyncio
from collections.abc import Iterable, Mapping
import os

from openai import AsyncAzureOpenAI
from config import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    DEFAULT_AZURE_OPENAI_DEPLOYMENT,
    DEFAULT_AZURE_OPENAI_ENDPOINT,
    DEFAULT_AZURE_OPENAI_MAX_RETRIES,
    DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS,
    DEFAULT_MODEL_ASK_MAX_TOKENS,
    DEFAULT_MODEL_ASK_QUESTION,
    DEFAULT_MODEL_ASK_SYSTEM_PROMPT,
    DEFAULT_MODEL_ASK_TEMPERATURE,
    DEFAULT_MODEL_ASK_TIMEOUT_SECONDS,
    DEFAULT_VLLM_PRESENCE_PENALTY,
)


def _require_env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    assert value is not None and value.strip(), (
        f"Missing required environment variable: {name}"
    )
    return value.strip()


def create_azure_openai_client(
    *,
    timeout_seconds: float = DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_AZURE_OPENAI_MAX_RETRIES,
) -> AsyncAzureOpenAI:
    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    assert api_key is not None and api_key.strip(), (
        "Missing required environment variable: AZURE_OPENAI_API_KEY"
    )
    return AsyncAzureOpenAI(
        api_key=api_key.strip(),
        azure_endpoint=_require_env(
            "AZURE_OPENAI_ENDPOINT",
            DEFAULT_AZURE_OPENAI_ENDPOINT,
        ),
        api_version=_require_env(
            "AZURE_OPENAI_API_VERSION",
            DEFAULT_AZURE_OPENAI_API_VERSION,
        ),
        timeout=timeout_seconds,
        max_retries=max_retries,
    )


def create_grader_client() -> AsyncAzureOpenAI:
    return create_azure_openai_client()


def grader_deployment_name() -> str:
    return _require_env(
        "AZURE_OPENAI_DEPLOYMENT",
        os.environ.get("AZURE_OPENAI_MODEL", DEFAULT_AZURE_OPENAI_DEPLOYMENT),
    )


def is_azure_openai_model(model_name: str | None) -> bool:
    if model_name is None:
        return False
    normalized = model_name.strip().lower()
    return normalized.startswith("gpt-") or normalized.startswith("o")


def azure_openai_serving_config() -> dict[str, str]:
    return {
        "azure_endpoint": os.environ.get(
            "AZURE_OPENAI_ENDPOINT",
            DEFAULT_AZURE_OPENAI_ENDPOINT,
        ),
        "api_version": os.environ.get(
            "AZURE_OPENAI_API_VERSION",
            DEFAULT_AZURE_OPENAI_API_VERSION,
        ),
    }


def extract_chat_completion_content(payload: Mapping[str, object]) -> str:
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "Response missing choices."
    first_choice = choices[0]
    assert isinstance(first_choice, Mapping), "First choice must be an object."
    message = first_choice.get("message")
    assert isinstance(message, Mapping), "Choice missing message object."
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        joined = "".join(parts).strip()
        assert joined, "OpenAI completion missing visible text content."
        return joined
    assert False, "OpenAI completion missing assistant content."


def extract_usage_counts(
    completion_payload: Mapping[str, object],
) -> tuple[int | None, int | None, int | None]:
    usage = completion_payload.get("usage")
    if not isinstance(usage, Mapping):
        return None, None, None
    input_token_count = usage.get("prompt_tokens")
    output_token_count = usage.get("completion_tokens")
    cached_input_token_count = None
    prompt_tokens_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_tokens_details, Mapping):
        cached_input_token_count = prompt_tokens_details.get("cached_tokens")
    return (
        int(input_token_count) if isinstance(input_token_count, int) else None,
        (
            int(cached_input_token_count)
            if isinstance(cached_input_token_count, int)
            else None
        ),
        int(output_token_count) if isinstance(output_token_count, int) else None,
    )


def _normalize_request_messages(
    *,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = "",
) -> list[dict[str, str]]:
    normalized_prompt = prompt.strip()
    normalized_system_prompt = system_prompt.strip()
    request_messages: list[dict[str, str]] = []

    if normalized_system_prompt:
        request_messages.append(
            {"role": "system", "content": normalized_system_prompt}
        )

    if messages is None:
        assert normalized_prompt, (
            "prompt must be non-empty when messages are not provided."
        )
        request_messages.append({"role": "user", "content": normalized_prompt})
        return request_messages

    assert not normalized_prompt, "prompt must be empty when messages are provided."
    assert messages, "messages must be non-empty."
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        assert role, "OpenAI message role must be non-empty."
        assert content, "OpenAI message content must be non-empty."
        if normalized_system_prompt:
            assert role != "system", (
                "messages must not include a system role when system_prompt is "
                "provided."
            )
        request_messages.append({"role": role, "content": content})
    return request_messages


async def request_azure_openai_chat_completion(
    *,
    messages: Iterable[Mapping[str, str]],
    model: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float | None = None,
    presence_penalty: float | None = None,
    response_format: Mapping[str, object] | None = None,
    client: AsyncAzureOpenAI | None = None,
) -> dict[str, object]:
    request_messages = list(messages)
    assert request_messages, "messages must be non-empty."
    assert model.strip(), "model must be non-empty."
    assert max_tokens > 0, "max_tokens must be > 0."
    assert 0.0 <= temperature <= 2.0, "temperature must be in [0, 2]."
    if top_p is not None:
        assert 0.0 <= top_p <= 1.0, "top_p must be in [0, 1]."
    if presence_penalty is not None:
        assert -2.0 <= presence_penalty <= 2.0, (
            "presence_penalty must be in [-2, 2]."
        )

    owns_client = client is None
    if client is None:
        client = create_azure_openai_client()

    try:
        request_kwargs: dict[str, object] = {
            "messages": request_messages,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            request_kwargs["top_p"] = top_p
        if presence_penalty is not None:
            request_kwargs["presence_penalty"] = presence_penalty
        if response_format is not None:
            request_kwargs["response_format"] = dict(response_format)
        completion = await client.chat.completions.create(**request_kwargs)
        payload = completion.model_dump(mode="json")
        assert isinstance(payload, dict), (
            "Expected chat completion payload to be a JSON object."
        )
        return payload
    finally:
        if owns_client:
            await client.close()


async def ask(
    *,
    model_name: str,
    region: str | None = None,
    profile: str | None = None,
    max_tokens: int = DEFAULT_MODEL_ASK_MAX_TOKENS,
    temperature: float = DEFAULT_MODEL_ASK_TEMPERATURE,
    timeout_seconds: float = DEFAULT_MODEL_ASK_TIMEOUT_SECONDS,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = "",
    top_p: float | None = None,
) -> str:
    del region, profile
    request_messages = _normalize_request_messages(
        prompt=prompt,
        messages=messages,
        system_prompt=system_prompt,
    )
    client = create_azure_openai_client(timeout_seconds=timeout_seconds)
    try:
        completion_payload = await request_azure_openai_chat_completion(
            messages=request_messages,
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=DEFAULT_VLLM_PRESENCE_PENALTY,
            client=client,
        )
    finally:
        await client.close()
    return extract_chat_completion_content(completion_payload)


async def _main() -> None:
    answer = await ask(
        model_name=grader_deployment_name(),
        max_tokens=DEFAULT_MODEL_ASK_MAX_TOKENS,
        temperature=DEFAULT_MODEL_ASK_TEMPERATURE,
        timeout_seconds=DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS,
        prompt=DEFAULT_MODEL_ASK_QUESTION,
        system_prompt=DEFAULT_MODEL_ASK_SYSTEM_PROMPT,
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(_main())
