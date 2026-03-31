"""Qwen/vLLM helpers and server management used by the benchmark."""

from collections.abc import Mapping, Sequence
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time
from typing import Any, Literal, NotRequired, TypedDict

import httpx
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator, model_validator

from config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_UNSLOTH_VLLM_STANDBY,
    DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
    DEFAULT_VLLM_ENABLE_THINKING,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_KV_CACHE_DTYPE,
    DEFAULT_VLLM_MAX_SEQ_LENGTH,
    DEFAULT_VLLM_PRESENCE_PENALTY,
    DEFAULT_VLLM_TEMPERATURE,
    DEFAULT_VLLM_TOP_P,
    LOCAL_MODEL_TOKENIZER_FILES,
    MODEL_REFERENCE_ALIASES,
    PROJECT_BIN_DIR,
    PROJECT_ROOT,
    VLLM_STANDBY_ENV_KEY,
)

logger = logging.getLogger(__name__)
_VLLM_SERVER_REGISTRY_FILENAME = "benchmark_vllm_servers.json"
JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
SERVER_REGISTRY_ADAPTER = TypeAdapter(dict[str, dict[str, object]])
MAIN_QUESTION = "what's the main argument of Sutton's Bitter Lessons?"
DEFAULT_VLLM_HOST = "127.0.0.1"


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    choice: NotRequired[Choice]


class Conversation(list[ChatMessage]):
    """Mutable conversation with optional assistant completion choices."""

    def __init__(
        self,
        prompts: Sequence[ChatMessage] | None = None,
        *,
        system_prompt: str = "",
    ) -> None:
        messages: list[ChatMessage] = []
        if system_prompt:
            assert system_prompt.strip(), "system_prompt must be non-empty."
            messages.append({"role": "system", "content": system_prompt})
        if prompts is not None:
            for prompt in prompts:
                if system_prompt:
                    assert prompt["role"] != "system", (
                        "prompts must not include a system message when "
                        "system_prompt is provided."
                    )
                message: ChatMessage = {
                    "role": prompt["role"],
                    "content": prompt["content"],
                }
                if (choice := prompt.get("choice")) is not None:
                    message["choice"] = choice
                messages.append(message)
        super().__init__(messages)

    def as_server_payload(self) -> list[dict[str, str]]:
        return [
            {"role": message["role"], "content": message["content"]}
            for message in self
        ]


class QwenStrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class VllmEngineConfig(QwenStrictModel):
    max_seq_length: int
    kv_cache_dtype: str
    gpu_memory_utilization: float

    @field_validator("max_seq_length")
    @classmethod
    def validate_max_seq_length(cls, value: int) -> int:
        assert value > 0, "vllm.max_seq_length must be > 0."
        return value

    @field_validator("kv_cache_dtype")
    @classmethod
    def validate_kv_cache_dtype(cls, value: str) -> str:
        value = value.strip()
        assert value, "vllm.kv_cache_dtype must be non-empty."
        return value

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_memory_utilization(cls, value: float) -> float:
        assert 0 < value <= 1, (
            "vllm.gpu_memory_utilization must be in (0, 1]."
        )
        return value


class VllmSamplingParams(QwenStrictModel):
    completion_token_limit: int
    top_p: float
    temperature: float
    presence_penalty: float

    @field_validator("completion_token_limit")
    @classmethod
    def validate_completion_token_limit(cls, value: int) -> int:
        assert value > 0, (
            "vllm.completion_token_limit must be > 0."
        )
        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float) -> float:
        assert 0 < value <= 1, "vllm.top_p must be in (0, 1]."
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        assert 0 <= value <= 2, "vllm.temperature must be in [0, 2]."
        return value

    @field_validator("presence_penalty")
    @classmethod
    def validate_presence_penalty(cls, value: float) -> float:
        assert -2 <= value <= 2, (
            "vllm.presence_penalty must be in [-2, 2]."
        )
        return value


class VllmRequestMetrics(QwenStrictModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration_seconds: float

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens")
    @classmethod
    def validate_non_negative_tokens(cls, value: int, info: Any) -> int:
        assert value >= 0, (
            f"vLLM completion usage.{info.field_name} must be a non-negative int."
        )
        return value

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration_seconds(cls, value: float) -> float:
        assert value > 0, "vLLM request duration must be > 0."
        return value

    @model_validator(mode="after")
    def validate_total_tokens(self) -> "VllmRequestMetrics":
        assert self.prompt_tokens >= 0, (
            "vLLM completion usage.prompt_tokens must be a non-negative int."
        )
        assert self.prompt_tokens + self.completion_tokens == self.total_tokens, (
            "vLLM completion usage.total_tokens must equal prompt_tokens + "
            "completion_tokens."
        )
        return self


class VllmServeTarget(QwenStrictModel):
    base_model_name: str
    request_model_name: str
    checkpoint_path: Path | None
    lora_rank: int | None

    @field_validator("base_model_name", "request_model_name")
    @classmethod
    def validate_non_empty_names(cls, value: str, info: Any) -> str:
        value = value.strip()
        assert value, f"{info.field_name} must be non-empty."
        return value

    @model_validator(mode="after")
    def validate_checkpoint_fields(self) -> "VllmServeTarget":
        if self.checkpoint_path is None:
            assert self.lora_rank is None, (
                "lora_rank must be omitted when checkpoint_path is not set."
            )
        else:
            assert self.lora_rank is not None and self.lora_rank > 0, (
                "lora_rank must be a positive integer when checkpoint_path is set."
            )
        return self


class VllmServerHandle(QwenStrictModel):
    process: subprocess.Popen[str] | None
    base_url: str
    model_name: str
    base_model_name: str
    log_path: Path
    reused: bool

    @field_validator("base_url", "model_name", "base_model_name")
    @classmethod
    def validate_non_empty_strings(cls, value: str, info: Any) -> str:
        value = value.strip()
        assert value, f"{info.field_name} must be non-empty."
        return value


def strip_think_traces(text: str) -> str:
    think_block_re = re.compile(r"<think>\s*.*?\s*</think>", re.IGNORECASE | re.DOTALL)
    think_tag_re = re.compile(r"</?think>", re.IGNORECASE)
    cleaned = think_block_re.sub("", text)

    while True:
        lowered = cleaned.lower()
        first_open = lowered.find("<think>")
        first_close = lowered.find("</think>")
        if first_close == -1 or (first_open != -1 and first_open < first_close):
            break
        cleaned = cleaned[first_close + len("</think>") :]

    lowered = cleaned.lower()
    first_open = lowered.find("<think>")
    if first_open != -1 and lowered.find("</think>", first_open) == -1:
        cleaned = cleaned[:first_open]

    without_tags = think_tag_re.sub("", cleaned)
    return without_tags.strip()


def require_chat_completion_message(payload: Mapping[str, object]) -> Mapping[str, object]:
    choices = payload.get("choices")
    assert isinstance(choices, list) and choices, "Response missing choices."
    first_choice = choices[0]
    assert isinstance(first_choice, dict), "First choice must be an object."
    message = first_choice.get("message")
    assert isinstance(message, dict), "Choice missing message object."
    return message


def uses_qwen3_reasoning(model_name: str) -> bool:
    return "qwen3" in model_name.strip().lower()


def resolve_model_reference(
    model_reference: object,
    *,
    config_key: str = "base_model",
) -> str:
    assert isinstance(model_reference, str), f"{config_key} must be a string."
    normalized_reference = model_reference.strip()
    assert normalized_reference, f"{config_key} must be non-empty."

    candidate_path = Path(normalized_reference).expanduser()
    if not candidate_path.exists():
        aliased_reference = MODEL_REFERENCE_ALIASES.get(
            normalized_reference.lower(),
        )
        if aliased_reference is not None:
            return aliased_reference
        return normalized_reference

    resolved_path = candidate_path.resolve()
    assert resolved_path.is_dir(), (
        f"{config_key} must point to a model directory when using a local path: "
        f"{resolved_path}"
    )
    config_json_path = resolved_path / "config.json"
    assert config_json_path.is_file(), (
        f"{config_key} local model directory is missing config.json: "
        f"{config_json_path}"
    )
    has_tokenizer_files = any(
        (resolved_path / tokenizer_file).exists()
        for tokenizer_file in LOCAL_MODEL_TOKENIZER_FILES
    )
    assert has_tokenizer_files, (
        f"{config_key} local model directory is missing tokenizer files: "
        f"{resolved_path}"
    )
    return str(resolved_path)


def _compose_raw_assistant_content(
    *,
    visible_content: str | None,
    reasoning_content: str | None,
) -> str | None:
    assert visible_content is None or isinstance(visible_content, str), (
        "vLLM completion message.content must be a string or null."
    )
    assert reasoning_content is None or isinstance(reasoning_content, str), (
        "vLLM completion message.reasoning must be a string or null."
    )

    if reasoning_content is None:
        return visible_content

    stripped_reasoning = reasoning_content.strip()
    cleaned_visible_content = re.sub(
        r"^\s*</think>\s*",
        "",
        visible_content or "",
        count=1,
        flags=re.IGNORECASE,
    )
    stripped_content = cleaned_visible_content.strip()
    if stripped_content:
        return f"<think>\n{stripped_reasoning}\n</think>\n\n{stripped_content}"
    return f"<think>\n{stripped_reasoning}\n</think>"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    assert isinstance(value, str), "Expected a string or null."
    return value


def _apply_runtime_env_defaults() -> None:
    if PROJECT_BIN_DIR.is_dir():
        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep) if current_path else []
        project_bin = str(PROJECT_BIN_DIR)
        if project_bin not in path_entries:
            os.environ["PATH"] = (
                f"{project_bin}{os.pathsep}{current_path}"
                if current_path
                else project_bin
            )
    os.environ.setdefault(VLLM_STANDBY_ENV_KEY, DEFAULT_UNSLOTH_VLLM_STANDBY)


def _normalize_request_messages(
    *,
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
    system_prompt: str = "",
) -> list[ChatMessage]:
    normalized_prompt = prompt.strip()
    normalized_system_prompt = system_prompt.strip()
    if messages is None:
        assert normalized_prompt, (
            "prompt must be non-empty when messages are not provided."
        )
        return [{"role": "user", "content": normalized_prompt}]

    assert not normalized_prompt, "prompt must be empty when messages are provided."
    assert messages, "messages must be non-empty."
    request_messages: list[ChatMessage] = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        assert role in {"system", "user", "assistant"}, (
            "Qwen messages must use only system/user/assistant roles."
        )
        assert content, "Qwen message content must be non-empty."
        if normalized_system_prompt:
            assert role != "system", (
                "messages must not include a system role when system_prompt is "
                "provided."
            )
        request_messages.append({"role": role, "content": content})
    return request_messages


async def generate_vllm_chat_completion(
    *,
    model_name: str,
    base_model_name: str | None = None,
    messages: Conversation,
    vllm_client: httpx.AsyncClient,
    sampling_params: VllmSamplingParams,
    enable_thinking: bool,
) -> tuple[str | None, str, VllmRequestMetrics]:
    assert messages, "messages must be non-empty."
    assert model_name.strip(), "model_name must be non-empty."
    if base_model_name is not None:
        assert base_model_name.strip(), "base_model_name must be non-empty when set."

    request_payload: dict[str, object] = {
        "model": model_name,
        "messages": messages.as_server_payload(),
        "max_tokens": sampling_params.completion_token_limit,
        "top_p": sampling_params.top_p,
        "temperature": sampling_params.temperature,
        "presence_penalty": sampling_params.presence_penalty,
        "include_reasoning": enable_thinking,
    }
    reasoning_model_name = base_model_name or model_name
    if uses_qwen3_reasoning(reasoning_model_name):
        request_payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    request_started_at = time.perf_counter()
    response = await vllm_client.post("/chat/completions", json=request_payload)
    request_duration_seconds = time.perf_counter() - request_started_at
    response.raise_for_status()
    payload = response.json()
    assert isinstance(payload, dict), (
        "Expected vLLM chat completion payload to be a JSON object."
    )
    usage = payload.get("usage")
    assert isinstance(usage, dict), "vLLM completion response must include usage."
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    message = require_chat_completion_message(payload)
    raw_content = _compose_raw_assistant_content(
        visible_content=_optional_str(message.get("content")),
        reasoning_content=_optional_str(
            message.get("reasoning", message.get("reasoning_content"))
        ),
    )
    logger.debug("generate assistant turn | raw_content: %s", raw_content)
    assistant_content = strip_think_traces(raw_content or "")
    return (
        raw_content,
        assistant_content,
        VllmRequestMetrics(
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(total_tokens),
            duration_seconds=request_duration_seconds,
        ),
    )


def _sanitize_filename_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    assert cleaned, "Expected a non-empty filename component."
    return cleaned


def _load_lora_adapter_config(
    checkpoint_path: Path,
    *,
    config_key: str,
) -> dict[str, object]:
    assert checkpoint_path.is_dir(), (
        f"{config_key} must point to a checkpoint directory: {checkpoint_path}"
    )
    adapter_config_path = checkpoint_path / "adapter_config.json"
    assert adapter_config_path.is_file(), (
        f"{config_key} must point to a LoRA adapter directory containing "
        f"adapter_config.json: {adapter_config_path}"
    )
    adapter_config = JSON_OBJECT_ADAPTER.validate_json(
        adapter_config_path.read_text(encoding="utf-8"),
    )
    assert adapter_config.get("peft_type") == "LORA", (
        f"{config_key} must point to a LoRA adapter directory: {checkpoint_path}"
    )
    return adapter_config


def resolve_vllm_serve_target(
    *,
    base_model: str,
    checkpoint: Path | None,
) -> VllmServeTarget:
    if checkpoint is None:
        resolved_base_model = resolve_model_reference(base_model)
        return VllmServeTarget(
            base_model_name=resolved_base_model,
            request_model_name=resolved_base_model,
            checkpoint_path=None,
            lora_rank=None,
        )

    checkpoint_path = checkpoint.expanduser().resolve()
    adapter_config = _load_lora_adapter_config(
        checkpoint_path,
        config_key="checkpoint",
    )
    checkpoint_base_model = adapter_config.get("base_model_name_or_path")
    assert isinstance(checkpoint_base_model, str) and checkpoint_base_model.strip(), (
        "checkpoint adapter_config.json must include a non-empty "
        "base_model_name_or_path."
    )
    checkpoint_base_model = resolve_model_reference(
        checkpoint_base_model.strip(),
        config_key=(
            "checkpoint adapter_config.json base_model_name_or_path"
        ),
    )
    lora_rank = adapter_config.get("r")
    assert isinstance(lora_rank, int) and lora_rank > 0, (
        "checkpoint adapter_config.json must include a positive integer r."
    )
    checkpoint_alias_suffix = hashlib.sha256(
        str(checkpoint_path).encode("utf-8")
    ).hexdigest()[:12]
    checkpoint_alias = (
        f"checkpoint-{_sanitize_filename_component(checkpoint_path.name)}-"
        f"{checkpoint_alias_suffix}"
    )
    return VllmServeTarget(
        base_model_name=checkpoint_base_model,
        request_model_name=checkpoint_alias,
        checkpoint_path=checkpoint_path,
        lora_rank=lora_rank,
    )


def vllm_server_signature(
    *,
    serve_target: VllmServeTarget,
    engine_config: VllmEngineConfig,
) -> dict[str, object]:
    signature: dict[str, object] = {
        "model_name": serve_target.request_model_name,
        "base_model_name": serve_target.base_model_name,
        "max_model_len": engine_config.max_seq_length,
        "kv_cache_dtype": engine_config.kv_cache_dtype,
        "gpu_memory_utilization": engine_config.gpu_memory_utilization,
        "enable_prefix_caching": True,
        "language_model_only": True,
    }
    if serve_target.checkpoint_path is not None:
        signature["checkpoint_path"] = str(serve_target.checkpoint_path)
    if uses_qwen3_reasoning(serve_target.base_model_name):
        signature["reasoning_parser"] = "qwen3"
    return signature


def _vllm_server_signature_sha256(signature: dict[str, object]) -> str:
    serialized_signature = JSON_OBJECT_ADAPTER.dump_json(signature).decode("utf-8")
    return hashlib.sha256(serialized_signature.encode("utf-8")).hexdigest()


def _vllm_server_registry_path(repo_root: Path) -> Path:
    return repo_root / "run" / _VLLM_SERVER_REGISTRY_FILENAME


def _load_vllm_server_registry(registry_path: Path) -> dict[str, dict[str, object]]:
    if not registry_path.is_file():
        return {}
    payload = SERVER_REGISTRY_ADAPTER.validate_json(
        registry_path.read_text(encoding="utf-8"),
    )
    return {
        str(model_name): entry
        for model_name, entry in payload.items()
        if isinstance(model_name, str) and isinstance(entry, dict)
    }


def _save_vllm_server_registry(
    registry_path: Path,
    registry: dict[str, dict[str, object]],
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        SERVER_REGISTRY_ADAPTER.dump_json(
            registry,
            indent=2,
            ensure_ascii=False,
        ).decode("utf-8")
        + "\n",
        encoding="utf-8",
    )


def _read_process_cmdline(pid: int) -> list[str]:
    assert pid > 0, "PID must be positive."
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    if not cmdline_path.is_file():
        return []
    raw_cmdline = cmdline_path.read_bytes()
    return [token for token in raw_cmdline.decode("utf-8").split("\0") if token]


def _process_is_alive(pid: int) -> bool:
    assert pid > 0, "PID must be positive."
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_process_group_id(pid: int) -> None:
    if not _process_is_alive(pid):
        return
    try:
        process_group_id = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while _process_is_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _process_is_alive(pid):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    _kill_process_group_id(process.pid)
    process.wait(timeout=5)


def _restart_registered_vllm_server(
    entry: dict[str, object],
    *,
    model_name: str,
) -> bool:
    launcher_pid = entry.get("launcher_pid")
    if not isinstance(launcher_pid, int) or launcher_pid <= 0:
        return False
    if not _process_is_alive(launcher_pid):
        return False

    joined_cmdline = " ".join(_read_process_cmdline(launcher_pid))
    if "vllm serve" not in joined_cmdline or model_name not in joined_cmdline:
        return False

    _kill_process_group_id(launcher_pid)
    return True


def _choose_port(requested_port: int) -> int:
    if requested_port > 0:
        return requested_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _listening_pid_for_port(host: str, port: int) -> int | None:
    result = subprocess.run(
        ["ss", "-ltnp"],
        check=True,
        capture_output=True,
        text=True,
    )
    address_suffix = f"{host}:{port}"
    pid_pattern = re.compile(r"pid=(\d+)")
    for line in result.stdout.splitlines():
        if address_suffix not in line:
            continue
        match = pid_pattern.search(line)
        if match is not None:
            return int(match.group(1))
    return None


def _stop_listener_on_port(host: str, port: int) -> None:
    listener_pid = _listening_pid_for_port(host, port)
    assert listener_pid is not None, (
        f"Expected a listening process on {host}:{port}, but none was found."
    )
    logger.info(
        "Stopping existing vLLM listener on requested port | host=%s | port=%d | "
        "pid=%d",
        host,
        port,
        listener_pid,
    )
    _kill_process_group_id(listener_pid)
    deadline = time.monotonic() + 10
    while not _port_is_available(host, port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _port_is_available(host, port), (
        f"Timed out waiting for {host}:{port} to become available after stopping "
        "the existing listener."
    )


async def _wait_for_vllm_ready(
    base_url: str,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: str | None = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while asyncio.get_running_loop().time() < deadline:
            try:
                response = await client.get(f"{base_url}/models")
                response.raise_for_status()
                payload = response.json()
                data = payload.get("data", [])
                assert isinstance(data, list), "Expected /models data to be a list."
                return data
            except (httpx.HTTPError, json.JSONDecodeError, AssertionError) as exc:
                last_error = str(exc)
                await asyncio.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for vLLM readiness at {base_url}/models. "
        f"Last error: {last_error or 'unknown'}"
    )


async def _fetch_served_model_names(
    base_url: str,
    timeout_seconds: float = 3.0,
) -> list[str] | None:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(f"{base_url}/models")
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, json.JSONDecodeError):
        return None

    data = payload.get("data", [])
    if not isinstance(data, list):
        return None
    model_names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            model_names.append(model_id)
    return model_names


def _build_vllm_command(
    *,
    repo_root: Path,
    serve_target: VllmServeTarget,
    engine_config: VllmEngineConfig,
    host: str,
    port: int,
) -> str:
    server_signature = vllm_server_signature(
        serve_target=serve_target,
        engine_config=engine_config,
    )
    parts = [
        "cd",
        shlex.quote(str(repo_root)),
        "&&",
        "uv",
        "run",
        "--env-file",
        ".env",
        "vllm",
        "serve",
        shlex.quote(serve_target.base_model_name),
        "--host",
        shlex.quote(host),
        "--port",
        str(port),
        "--served-model-name",
        shlex.quote(serve_target.base_model_name),
        "--max-model-len",
        str(server_signature["max_model_len"]),
        "--kv-cache-dtype",
        shlex.quote(str(server_signature["kv_cache_dtype"])),
        "--gpu-memory-utilization",
        str(server_signature["gpu_memory_utilization"]),
        "--enable-prefix-caching",
        "--language-model-only",
    ]
    if server_signature.get("reasoning_parser") == "qwen3":
        parts.extend(["--reasoning-parser", "qwen3"])
    if serve_target.checkpoint_path is not None:
        assert serve_target.lora_rank is not None, (
            "lora_rank must be set when checkpoint_path is set."
        )
        lora_spec = JSON_OBJECT_ADAPTER.dump_json(
            {
                "name": serve_target.request_model_name,
                "path": str(serve_target.checkpoint_path),
                "base_model_name": serve_target.base_model_name,
            },
            ensure_ascii=False,
        ).decode("utf-8")
        parts.extend(
            [
                "--enable-lora",
                "--max-lora-rank",
                str(serve_target.lora_rank),
                "--lora-modules",
                shlex.quote(lora_spec),
            ]
        )
    return " ".join(parts)


async def _start_vllm_server(
    *,
    repo_root: Path,
    serve_target: VllmServeTarget,
    engine_config: VllmEngineConfig,
    host: str,
    port: int,
    startup_timeout_seconds: float,
    log_path: Path,
) -> VllmServerHandle:
    command = _build_vllm_command(
        repo_root=repo_root,
        serve_target=serve_target,
        engine_config=engine_config,
        host=host,
        port=port,
    )
    with log_path.open("a", buffering=1) as log_handle:
        process = subprocess.Popen(
            ["bash", "-ic", command],
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    base_url = f"http://{host}:{port}/v1"
    logger.info(
        "Launching benchmark vLLM server | base_url=%s | model=%s | base_model=%s | "
        "checkpoint=%s | log=%s",
        base_url,
        serve_target.request_model_name,
        serve_target.base_model_name,
        str(serve_target.checkpoint_path) if serve_target.checkpoint_path else "",
        log_path,
    )
    try:
        await _wait_for_vllm_ready(base_url, startup_timeout_seconds)
    except BaseException:
        _kill_process_group(process)
        raise

    return VllmServerHandle(
        process=process,
        base_url=base_url,
        model_name=serve_target.request_model_name,
        base_model_name=serve_target.base_model_name,
        log_path=log_path,
        reused=False,
    )


async def ensure_vllm_server(
    *,
    repo_root: Path,
    base_model: str,
    checkpoint: Path | None,
    engine_config: VllmEngineConfig,
    host: str,
    requested_port: int,
    startup_timeout_seconds: float,
    log_path: Path,
) -> VllmServerHandle:
    serve_target = resolve_vllm_serve_target(
        base_model=base_model,
        checkpoint=checkpoint,
    )
    server_signature = vllm_server_signature(
        serve_target=serve_target,
        engine_config=engine_config,
    )
    server_config_sha256 = _vllm_server_signature_sha256(server_signature)
    model_name = str(server_signature["model_name"])
    registry_path = _vllm_server_registry_path(repo_root)
    registry = _load_vllm_server_registry(registry_path)

    registered_entry = registry.get(model_name)
    if registered_entry is not None:
        registered_base_url = registered_entry.get("base_url")
        registered_log_path = registered_entry.get("log_path")
        registered_config_sha256 = registered_entry.get("server_config_sha256")
        if isinstance(registered_base_url, str):
            served_models = await _fetch_served_model_names(registered_base_url)
            if (
                served_models is not None
                and model_name in served_models
                and registered_config_sha256 == server_config_sha256
            ):
                resolved_log_path = (
                    Path(registered_log_path)
                    if isinstance(registered_log_path, str)
                    else log_path
                )
                return VllmServerHandle(
                    process=None,
                    base_url=registered_base_url,
                    model_name=model_name,
                    base_model_name=str(server_signature["base_model_name"]),
                    log_path=resolved_log_path,
                    reused=True,
                )
        restarted = _restart_registered_vllm_server(
            registered_entry,
            model_name=model_name,
        )
        if restarted:
            logger.info(
                "Restarting benchmark vLLM server | model=%s | old_config_sha256=%s "
                "| new_config_sha256=%s",
                model_name,
                registered_config_sha256,
                server_config_sha256,
            )
        registry.pop(model_name, None)
        _save_vllm_server_registry(registry_path, registry)

    selected_port = requested_port
    if requested_port > 0:
        if not _port_is_available(host, requested_port):
            _stop_listener_on_port(host, requested_port)
    else:
        selected_port = _choose_port(0)

    handle = await _start_vllm_server(
        repo_root=repo_root,
        serve_target=serve_target,
        engine_config=engine_config,
        host=host,
        port=selected_port,
        startup_timeout_seconds=startup_timeout_seconds,
        log_path=log_path,
    )
    registry[model_name] = {
        "base_url": handle.base_url,
        "host": host,
        "launcher_pid": handle.process.pid if handle.process is not None else None,
        "log_path": str(handle.log_path),
        "model_name": model_name,
        "port": selected_port,
        "server_signature": server_signature,
        "server_config_sha256": server_config_sha256,
    }
    _save_vllm_server_registry(registry_path, registry)
    return handle


async def ask(
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
    del region, profile
    _apply_runtime_env_defaults()
    normalized_system_prompt = system_prompt.strip()
    conversation = Conversation(
        _normalize_request_messages(
            prompt=prompt,
            messages=messages,
            system_prompt=normalized_system_prompt,
        ),
        system_prompt=normalized_system_prompt,
    )
    handle = await ensure_vllm_server(
        repo_root=PROJECT_ROOT,
        base_model=model_name,
        checkpoint=None,
        engine_config=VllmEngineConfig(
            max_seq_length=DEFAULT_VLLM_MAX_SEQ_LENGTH,
            kv_cache_dtype=DEFAULT_VLLM_KV_CACHE_DTYPE,
            gpu_memory_utilization=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        ),
        host=DEFAULT_VLLM_HOST,
        requested_port=0,
        startup_timeout_seconds=max(
            timeout_seconds,
            float(DEFAULT_GENERATION_TIMEOUT_SECONDS),
        ),
        log_path=PROJECT_ROOT / "run" / "qwen.ask.vllm.log",
    )
    async with httpx.AsyncClient(
        base_url=handle.base_url,
        timeout=timeout_seconds,
    ) as client:
        _, assistant_content, _ = await asyncio.wait_for(
            generate_vllm_chat_completion(
                model_name=handle.model_name,
                base_model_name=handle.base_model_name,
                messages=conversation,
                vllm_client=client,
                sampling_params=VllmSamplingParams(
                    completion_token_limit=max_tokens,
                    top_p=DEFAULT_VLLM_TOP_P if top_p is None else top_p,
                    temperature=temperature,
                    presence_penalty=DEFAULT_VLLM_PRESENCE_PENALTY,
                ),
                enable_thinking=DEFAULT_VLLM_ENABLE_THINKING,
            ),
            timeout=timeout_seconds,
        )
    return assistant_content


async def _main() -> None:
    answer = await ask(
        model_name=DEFAULT_BASE_MODEL,
        max_tokens=DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
        temperature=DEFAULT_VLLM_TEMPERATURE,
        timeout_seconds=float(DEFAULT_GENERATION_TIMEOUT_SECONDS),
        prompt=MAIN_QUESTION,
        system_prompt="You are a helpful assistant.",
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(_main())
