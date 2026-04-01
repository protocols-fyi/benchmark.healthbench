"""Run inference-only HealthBench benchmarks against local vLLM, Azure OpenAI,
or AWS Bedrock.

Usage:
    uv run --env-file .env main.py [--model ...] [--checkpoint ...] [--case-count N] [--rollout-count N]
    uv run --env-file .env main.py --model gpt-4.1 --case-count 1 --rollout-count 2 --results-db ./results.sqlite3
    uv run --env-file .env main.py --model qwen3-8b-context-seeking-0268 --case-count 1 --rollout-count 2 --results-db ./results.sqlite3
    uv run --env-file .env main.py --model all --case-count 1 --rollout-count 2 --results-db ./results.sqlite3

The benchmark runs the `physician_agreed_category:not-enough-context` prompts
from `context_seeking_consensus_hard_104.jsonl`, grades outputs against the
shared context-awareness rubric for that slice with Azure OpenAI, and reports
aggregate metrics.

The command applies the hard-coded benchmark settings from this repo, starts or
reuses a local vLLM server here, generates one or more responses per input,
logs to `run/*.log`, and prints focused aggregate metrics as CSV.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import click

from config import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    DEFAULT_BASE_MODEL,
    DEFAULT_LOCAL_CHECKPOINT_MODEL_ID,
    DEFAULT_LOCAL_CHECKPOINT_PATH,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_UNSLOTH_VLLM_STANDBY,
    DEFAULT_VLLM_BASE_URL,
    NOISY_LOGGERS,
    PROJECT_BIN_DIR,
    PROJECT_ROOT,
    VLLM_STANDBY_ENV_KEY,
)
from entities import TargetModel
from executor import execute, load_cases
from models.anthropic import AWS_BEDROCK_BENCHMARK_REQUEST_PARAMETERS
from models.openai import (
    AZURE_OPENAI_BENCHMARK_REQUEST_PARAMETERS,
    azure_openai_serving_config,
    is_azure_openai_model,
)
from models.qwen import VLLM_BENCHMARK_REQUEST_PARAMETERS, ensure_vllm_server
from utils import init_logging

logger = logging.getLogger(__name__)
DATASET_PATH = PROJECT_ROOT / "context_seeking_consensus_hard_104.jsonl"
DEFAULT_RESULTS_DB_PATH = Path("./results.sqlite3")
CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
ALL_MODEL_OPTION = "all"
SUPPORTED_NAMED_MODEL_IDS = (
    "gpt-4.1",
    "qwen3-8b",
    DEFAULT_LOCAL_CHECKPOINT_MODEL_ID,
    *sorted(AWS_BEDROCK_SUPPORTED_MODEL_IDS),
)
MODEL_OPTION_HELP = (
    "Target model to benchmark. Supported named targets: "
    f"{', '.join(SUPPORTED_NAMED_MODEL_IDS)}. "
    f"Use '{ALL_MODEL_OPTION}' to benchmark each named target sequentially. "
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


def _resolve_requested_models(
    *,
    model: str | None,
    checkpoint: Path | None,
) -> tuple[str | None, ...]:
    if model is None:
        return (None,)

    stripped_model = model.strip()
    normalized_model = stripped_model.lower()
    if normalized_model == ALL_MODEL_OPTION:
        if checkpoint is not None:
            raise click.UsageError("--model all cannot be combined with --checkpoint.")
        return SUPPORTED_NAMED_MODEL_IDS
    if normalized_model in SUPPORTED_NAMED_MODEL_IDS or is_azure_openai_model(
        normalized_model
    ):
        return (normalized_model,)
    return (stripped_model,)


async def run_case(
    *,
    case_count: int,
    rollout_count: int | None,
    model_name: str | None,
    checkpoint: Path | None,
    repo_root: Path,
    results_db_path: Path,
) -> tuple[str, dict[str, list[float]]]:
    named_local_checkpoint = checkpoint is None and (
        model_name == DEFAULT_LOCAL_CHECKPOINT_MODEL_ID
    )
    resolved_checkpoint = (
        DEFAULT_LOCAL_CHECKPOINT_PATH if named_local_checkpoint else checkpoint
    )
    served_model_name = (
        DEFAULT_LOCAL_CHECKPOINT_MODEL_ID if named_local_checkpoint else None
    )
    is_anthropic = model_name is not None and model_name.startswith("anthropic-")
    is_azure_openai = is_azure_openai_model(model_name)
    resolved_rollout_count = (
        DEFAULT_ROLLOUT_COUNT if rollout_count is None else rollout_count
    )
    local_checkpoint = (
        None if (is_anthropic or is_azure_openai) else resolved_checkpoint
    )
    if local_checkpoint is not None and model_name is not None:
        if named_local_checkpoint:
            logger.info(
                "Running benchmark on repo-local checkpoint | model=%s | "
                "checkpoint=%s",
                DEFAULT_LOCAL_CHECKPOINT_MODEL_ID,
                str(local_checkpoint.expanduser().resolve()),
            )
        else:
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
    cases = load_cases(
        DATASET_PATH,
        case_count=case_count,
    )

    experiment_id = "default"
    run_experiment_id = experiment_id
    if is_anthropic:
        assert model_name is not None
        assert model_name in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
            f"Unsupported Anthropic model: {model_name}"
        )
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        logger.info(
            "Running benchmark on AWS Bedrock | model=%s | concurrency=4",
            AWS_BEDROCK_SUPPORTED_MODEL_IDS[model_name],
        )
        run_experiment_id = f"{experiment_id}-{model_name}"
        target_model = TargetModel.aws_bedrock(
            model_id=model_name,
            request_parameters_json=json.dumps(
                AWS_BEDROCK_BENCHMARK_REQUEST_PARAMETERS,
                ensure_ascii=False,
            ),
        )
    elif is_azure_openai:
        assert model_name is not None
        logger.info(
            "Running benchmark on Azure OpenAI | model=%s | concurrency=4",
            model_name,
        )
        run_experiment_id = f"{experiment_id}-{model_name}"
        target_model = TargetModel.azure_openai(
            model_id=model_name,
            request_parameters_json=json.dumps(
                AZURE_OPENAI_BENCHMARK_REQUEST_PARAMETERS,
                ensure_ascii=False,
            ),
            serving_config_json=json.dumps(
                azure_openai_serving_config(),
                ensure_ascii=False,
            ),
        )
    else:
        base_url, model_id = await ensure_vllm_server(
            repo_root=repo_root,
            base_model=local_model_name,
            checkpoint=local_checkpoint,
            served_model_name=served_model_name,
        )
        assert base_url == DEFAULT_VLLM_BASE_URL, (
            "Expected vLLM server at "
            f"{DEFAULT_VLLM_BASE_URL}, got {base_url}."
        )
        target_model = TargetModel.vllm(
            model_id=model_id,
            checkpoint_path=(
                str(local_checkpoint.expanduser().resolve())
                if local_checkpoint is not None
                else None
            ),
            request_parameters_json=json.dumps(
                VLLM_BENCHMARK_REQUEST_PARAMETERS,
                ensure_ascii=False,
            ),
            serving_config_json=json.dumps({"base_url": base_url}, ensure_ascii=False),
        )

    return await execute(
        experiment_id=run_experiment_id,
        target_model=target_model,
        cases=cases,
        rollout_size=resolved_rollout_count,
        results_db_path=results_db_path,
    )


async def run_benchmark(
    *,
    case_count: int,
    rollout_count: int | None,
    model_names: tuple[str | None, ...],
    checkpoint: Path | None,
    repo_root: Path,
    results_db_path: Path,
) -> list[tuple[str, dict[str, list[float]]]]:
    benchmark_jobs = list(enumerate(model_names, start=1))
    collected_results: list[tuple[str, dict[str, list[float]]]] = []

    for job_index, model_name in benchmark_jobs:
        model_label = DEFAULT_BASE_MODEL if model_name is None else model_name
        try:
            logger.info(
                "Starting benchmark run | run_index=%d/%d | model=%s",
                job_index,
                len(model_names),
                model_label,
            )
            collected_results.append(
                await run_case(
                    case_count=case_count,
                    rollout_count=rollout_count,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    repo_root=repo_root,
                    results_db_path=results_db_path,
                )
            )
        except Exception as error:
            logger.exception(
                "Benchmark run failed | run_index=%d/%d | model=%s",
                job_index,
                len(model_names),
                model_label,
                exc_info=error,
            )
    return collected_results


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
    help="Number of rollout samples to generate per case.",
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
    init_logging(level=logging.DEBUG)
    requested_models = _resolve_requested_models(model=model, checkpoint=checkpoint)
    asyncio.run(
        run_benchmark(
            case_count=case_count,
            rollout_count=rollout_count,
            model_names=requested_models,
            checkpoint=checkpoint,
            repo_root=PROJECT_ROOT,
            results_db_path=results_db,
        )
    )


if __name__ == "__main__":
    cli()
