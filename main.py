"""Run inference-only HealthBench benchmarks against local vLLM, Azure OpenAI,
or AWS Bedrock.

Usage:
    uv run --env-file .env main.py [--model ...] [--checkpoint ...] [--case-count N] [--rollout-count N]
    uv run --env-file .env main.py --model gpt-4.1 --case-count 1 --rollout-count 2 --results-db ./results.sqlite3

The benchmark runs the `physician_agreed_category:not-enough-context` prompts
from `context_seeking_consensus_hard_104.jsonl`, grades outputs
against the shared context-awareness rubric for that slice with Azure OpenAI,
and reports aggregate metrics.

The command applies the hard-coded benchmark settings from this repo, starts or
reuses a local vLLM server here, generates one or more responses per input,
logs to `run/*.log`, and prints focused aggregate metrics as CSV.
"""

import asyncio
import logging
import os
from pathlib import Path

import click
from config import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    DEFAULT_BASE_MODEL,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_UNSLOTH_VLLM_STANDBY,
    DEFAULT_VLLM_BASE_URL,
    DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
    DEFAULT_VLLM_ENABLE_THINKING,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_HOST,
    DEFAULT_VLLM_KV_CACHE_DTYPE,
    DEFAULT_VLLM_MAX_SEQ_LENGTH,
    DEFAULT_VLLM_PRESENCE_PENALTY,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS,
    DEFAULT_VLLM_TEMPERATURE,
    DEFAULT_VLLM_TOP_P,
    NOISY_LOGGERS,
    PROJECT_BIN_DIR,
    PROJECT_ROOT,
    VLLM_STANDBY_ENV_KEY,
)
from entities import (
    BenchmarkRuntime,
    TargetModel,
    dump_json_value,
)
from models.openai import (
    azure_openai_serving_config,
    is_azure_openai_model,
)
from rollout import load_cases, run_rollouts
from models.qwen import (
    VllmEngineConfig,
    VllmSamplingParams,
    ensure_vllm_server,
)
from utils import init_logging

logger = logging.getLogger(__name__)
DATASET_PATH = PROJECT_ROOT / "context_seeking_consensus_hard_104.jsonl"
DEFAULT_RESULTS_DB_PATH = Path("./results.sqlite3")
CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
SUPPORTED_NAMED_MODEL_IDS = (
    "gpt-4.1",
    "qwen3-8b",
    *sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS),
)
MODEL_OPTION_HELP = (
    "Target model to benchmark. Supported named targets: "
    f"{', '.join(SUPPORTED_NAMED_MODEL_IDS)}. "
    "For local vLLM runs, you can also pass any Hugging Face model id or "
    "a local model directory."
)


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


def _build_benchmark_runtime(
    *,
    rollout_count: int | None,
) -> BenchmarkRuntime:
    resolved_rollout_size = (
        DEFAULT_ROLLOUT_COUNT if rollout_count is None else rollout_count
    )
    return BenchmarkRuntime(
        rollout_size=resolved_rollout_size,
        generation_timeout_seconds=DEFAULT_GENERATION_TIMEOUT_SECONDS,
        sampling_params=VllmSamplingParams(
            completion_token_limit=DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
            top_p=DEFAULT_VLLM_TOP_P,
            temperature=DEFAULT_VLLM_TEMPERATURE,
            presence_penalty=DEFAULT_VLLM_PRESENCE_PENALTY,
        ),
        enable_thinking=DEFAULT_VLLM_ENABLE_THINKING,
        vllm_engine_config=VllmEngineConfig(
            max_seq_length=DEFAULT_VLLM_MAX_SEQ_LENGTH,
            kv_cache_dtype=DEFAULT_VLLM_KV_CACHE_DTYPE,
            gpu_memory_utilization=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        ),
    )


async def run_benchmark(
    *,
    case_count: int,
    rollout_count: int | None,
    model_name: str | None,
    checkpoint: Path | None,
    repo_root: Path,
    run_log_path: Path,
    results_db_path: Path,
) -> tuple[str, dict[str, list[float]]]:
    is_anthropic = model_name is not None and model_name.startswith("anthropic-")
    is_azure_openai = is_azure_openai_model(model_name)
    runtime = _build_benchmark_runtime(rollout_count=rollout_count)
    local_checkpoint = None if (is_anthropic or is_azure_openai) else checkpoint
    if local_checkpoint is not None and model_name is not None:
        logger.debug(
            "Ignoring --model because --checkpoint was provided | model=%s | "
            "checkpoint=%s",
            model_name,
            str(local_checkpoint.expanduser().resolve()),
        )
    local_model_name = (
        DEFAULT_BASE_MODEL
        if (
            model_name is None
            or is_anthropic
            or is_azure_openai
            or local_checkpoint is not None
        )
        else model_name
    )
    cases, rubric = load_cases(
        DATASET_PATH,
        case_count=case_count,
    )

    experiment_id = "default"
    sampling_params = runtime.sampling_params
    request_parameters_json = dump_json_value(
        {
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "presence_penalty": sampling_params.presence_penalty,
            "max_tokens": sampling_params.completion_token_limit,
        }
    )
    run_experiment_id = experiment_id
    if is_anthropic:
        assert model_name is not None
        assert model_name in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
            f"Unsupported Anthropic model: {model_name}"
        )
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logger.info(
            "Running benchmark on AWS Bedrock | model=%s | concurrency=1",
            AWS_BEDROCK_SUPPORTED_MODEL_IDS[model_name],
        )
        run_experiment_id = f"{experiment_id}-{model_name}"
        target_model = TargetModel.aws_bedrock(
            model_id=model_name,
            request_parameters_json=request_parameters_json,
        )
    elif is_azure_openai:
        assert model_name is not None
        logger.info(
            "Running benchmark on Azure OpenAI | model=%s | concurrency=1",
            model_name,
        )
        run_experiment_id = f"{experiment_id}-{model_name}"
        target_model = TargetModel.azure_openai(
            model_id=model_name,
            request_parameters_json=request_parameters_json,
            serving_config_json=dump_json_value(azure_openai_serving_config()),
        )
    else:
        vllm_handle = await ensure_vllm_server(
            repo_root=repo_root,
            base_model=local_model_name,
            checkpoint=local_checkpoint,
            engine_config=runtime.vllm_engine_config,
            host=DEFAULT_VLLM_HOST,
            requested_port=DEFAULT_VLLM_PORT,
            startup_timeout_seconds=DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS,
            log_path=run_log_path.with_suffix(".vllm.log"),
        )
        assert vllm_handle.base_url == DEFAULT_VLLM_BASE_URL, (
            "Expected vLLM server at "
            f"{DEFAULT_VLLM_BASE_URL}, got {vllm_handle.base_url}."
        )
        target_model = TargetModel.vllm(
            model_id=vllm_handle.model_name,
            base_model_name=vllm_handle.base_model_name,
            checkpoint_path=(
                str(local_checkpoint.expanduser().resolve())
                if local_checkpoint is not None
                else None
            ),
            request_parameters_json=request_parameters_json,
            serving_config_json=dump_json_value({"base_url": vllm_handle.base_url}),
        )

    return await run_rollouts(
        experiment_id=run_experiment_id,
        target_model=target_model,
        cases=cases,
        rubric=rubric,
        runtime=runtime,
        results_db_path=results_db_path,
    )


@click.command(context_settings=CLICK_CONTEXT_SETTINGS)
@click.option(
    "--model",
    type=str,
    default=None,
    help=MODEL_OPTION_HELP,
)
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help=(
        "Override vllm.checkpoint for this run with a LoRA adapter directory "
        "to benchmark. When present, --model is ignored for local runs."
    ),
)
@click.option(
    "--case-count",
    "case_count",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help=(
        "Number of dataset cases to benchmark. Use 0 for all cases. "
        "When > 0, selects a deterministic random subset with a fixed seed."
    ),
)
@click.option(
    "--rollout-count",
    type=click.IntRange(min=1),
    default=DEFAULT_ROLLOUT_COUNT,
    show_default=True,
    help=(
        "Number of rollout samples to generate per case."
    ),
)
@click.option(
    "--results-db",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_RESULTS_DB_PATH,
    show_default=True,
    help="SQLite file used to persist benchmark cases, models, and sessions.",
)
def cli(
    model: str | None,
    checkpoint: Path | None,
    case_count: int,
    rollout_count: int | None,
    results_db: Path,
) -> None:
    """Run HealthBench rubric benchmark rollouts and persist them to SQLite."""
    _apply_runtime_env_defaults()
    run_log_path = init_logging(level=logging.DEBUG)
    experiment_id, results = asyncio.run(
        run_benchmark(
            case_count=case_count,
            rollout_count=rollout_count,
            model_name=model.lower() if model is not None else None,
            checkpoint=checkpoint,
            repo_root=PROJECT_ROOT,
            run_log_path=run_log_path,
            results_db_path=results_db,
        )
    )


if __name__ == "__main__":
    cli()
