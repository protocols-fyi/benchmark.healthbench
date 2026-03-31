"""Minimal vLLM helpers for local Qwen serving."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Literal, NotRequired, TypedDict

import httpx

from config import (
    DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_HOST,
    DEFAULT_VLLM_KV_CACHE_DTYPE,
    DEFAULT_VLLM_MAX_SEQ_LENGTH,
    DEFAULT_VLLM_PRESENCE_PENALTY,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_VLLM_TEMPERATURE,
    DEFAULT_VLLM_TOP_P,
    MODEL_REFERENCE_ALIASES,
)


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    choice: NotRequired[object]


class _ServeTarget(TypedDict):
    base_model_name: str
    request_model_name: str
    checkpoint_path: Path | None
    lora_rank: int | None


def uses_qwen3_reasoning(model_name: str) -> bool:
    return "qwen3" in model_name.strip().lower()


def resolve_model_reference(
    model_reference: str,
    *,
    config_key: str = "base_model",
) -> str:
    reference = model_reference.strip()
    if not reference:
        raise ValueError(f"{config_key} must be non-empty.")

    path = Path(reference).expanduser()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{config_key} must be a directory: {path}")
        return str(path.resolve())

    return MODEL_REFERENCE_ALIASES.get(reference.lower(), reference)


def _checkpoint_alias(checkpoint_path: Path) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in "._-" else "-"
        for char in checkpoint_path.name
    ).strip("-")
    if not safe_name:
        safe_name = "checkpoint"
    suffix = hashlib.sha256(str(checkpoint_path).encode("utf-8")).hexdigest()[:12]
    return f"checkpoint-{safe_name}-{suffix}"


def _resolve_serve_target(
    *,
    base_model: str,
    checkpoint: Path | None,
) -> _ServeTarget:
    if checkpoint is None:
        resolved_base_model = resolve_model_reference(base_model)
        return {
            "base_model_name": resolved_base_model,
            "request_model_name": resolved_base_model,
            "checkpoint_path": None,
            "lora_rank": None,
        }

    checkpoint_path = checkpoint.expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint must be a directory: {checkpoint_path}")

    adapter_config_path = checkpoint_path / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint must contain adapter_config.json: {adapter_config_path}"
        )
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    if not isinstance(adapter_config, dict):
        raise ValueError("adapter_config.json must contain a JSON object.")

    base_model_name = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name.strip():
        raise ValueError(
            "Checkpoint adapter_config.json must include base_model_name_or_path."
        )
    lora_rank = adapter_config.get("r")
    if not isinstance(lora_rank, int) or lora_rank <= 0:
        raise ValueError("Checkpoint adapter_config.json must include a positive r.")

    return {
        "base_model_name": resolve_model_reference(
            base_model_name,
            config_key="checkpoint.adapter_config.json base_model_name_or_path",
        ),
        "request_model_name": _checkpoint_alias(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "lora_rank": lora_rank,
    }


def _extract_assistant_text(payload: Mapping[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("vLLM response missing choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("vLLM response choice is not an object.")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("vLLM response missing message object.")

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    reasoning = message.get("reasoning", message.get("reasoning_content"))
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    raise ValueError("vLLM response did not contain assistant text.")


async def generate_vllm_chat_completion(
    *,
    model_name: str,
    base_model_name: str | None,
    messages: Sequence[ChatMessage],
    vllm_client: httpx.AsyncClient,
) -> tuple[str, int, int]:
    payload: dict[str, object] = {
        "model": model_name,
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ],
        "max_tokens": DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
        "top_p": DEFAULT_VLLM_TOP_P,
        "temperature": DEFAULT_VLLM_TEMPERATURE,
        "presence_penalty": DEFAULT_VLLM_PRESENCE_PENALTY,
        "include_reasoning": True,
    }
    if uses_qwen3_reasoning(base_model_name or model_name):
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    response = await vllm_client.post("/chat/completions", json=payload)
    response.raise_for_status()

    result = response.json()
    if not isinstance(result, Mapping):
        raise ValueError("Expected vLLM response payload to be a JSON object.")
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        raise ValueError("vLLM response missing usage.")

    return (
        _extract_assistant_text(result),
        int(usage["prompt_tokens"]),
        int(usage["completion_tokens"]),
    )


async def _fetch_served_model_names(base_url: str) -> set[str] | None:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url}/models")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None

    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None

    return {
        model_id
        for item in data
        if isinstance(item, Mapping)
        for model_id in [item.get("id")]
        if isinstance(model_id, str) and model_id.strip()
    }


async def _wait_for_vllm_ready(
    *,
    base_url: str,
    process: subprocess.Popen[str],
    timeout_seconds: float,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before becoming ready with code {process.returncode}."
            )
        if await _fetch_served_model_names(base_url) is not None:
            return
        await asyncio.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM at {base_url}/models.")


def _build_vllm_command(
    *,
    serve_target: _ServeTarget,
) -> list[str]:
    command = [
        "uv",
        "run",
        "--env-file",
        ".env",
        "vllm",
        "serve",
        serve_target["base_model_name"],
        "--host",
        DEFAULT_VLLM_HOST,
        "--port",
        str(DEFAULT_VLLM_PORT),
        "--served-model-name",
        serve_target["base_model_name"],
        "--max-model-len",
        str(DEFAULT_VLLM_MAX_SEQ_LENGTH),
        "--kv-cache-dtype",
        DEFAULT_VLLM_KV_CACHE_DTYPE,
        "--gpu-memory-utilization",
        str(DEFAULT_VLLM_GPU_MEMORY_UTILIZATION),
        "--enable-prefix-caching",
        "--language-model-only",
    ]
    if uses_qwen3_reasoning(serve_target["base_model_name"]):
        command.extend(["--reasoning-parser", "qwen3"])
    if serve_target["checkpoint_path"] is not None:
        command.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(serve_target["lora_rank"]),
                "--lora-modules",
                json.dumps(
                    {
                        "name": serve_target["request_model_name"],
                        "path": str(serve_target["checkpoint_path"]),
                        "base_model_name": serve_target["base_model_name"],
                    }
                ),
            ]
        )
    return command


async def ensure_vllm_server(
    *,
    repo_root: Path,
    base_model: str,
    checkpoint: Path | None,
) -> tuple[str, str, str]:
    serve_target = _resolve_serve_target(
        base_model=base_model,
        checkpoint=checkpoint,
    )
    base_url = f"http://{DEFAULT_VLLM_HOST}:{DEFAULT_VLLM_PORT}/v1"
    served_models = await _fetch_served_model_names(base_url)
    if served_models is not None:
        if serve_target["request_model_name"] not in served_models:
            raise RuntimeError(
                f"Server at {base_url} is serving {', '.join(sorted(served_models))}, "
                f"not {serve_target['request_model_name']}."
            )
        return (
            base_url,
            serve_target["request_model_name"],
            serve_target["base_model_name"],
        )

    command = _build_vllm_command(
        serve_target=serve_target,
    )
    log_path = repo_root / "run" / "benchmark.vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1, encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    await _wait_for_vllm_ready(
        base_url=base_url,
        process=process,
        timeout_seconds=DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS,
    )
    return (
        base_url,
        serve_target["request_model_name"],
        serve_target["base_model_name"],
    )
