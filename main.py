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
from collections import defaultdict
import logging
import os
from pathlib import Path
import random

import click
import httpx
from models.anthropic import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    NOISY_LOGGERS,
    ask_bedrock,
)
from config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_UNSLOTH_VLLM_STANDBY,
    DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
    DEFAULT_VLLM_ENABLE_THINKING,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    DEFAULT_VLLM_KV_CACHE_DTYPE,
    DEFAULT_VLLM_MAX_SEQ_LENGTH,
    DEFAULT_VLLM_PRESENCE_PENALTY,
    DEFAULT_VLLM_TEMPERATURE,
    DEFAULT_VLLM_TOP_P,
    PROJECT_BIN_DIR,
    PROJECT_ROOT,
    VLLM_STANDBY_ENV_KEY,
)
from db import BenchmarkStore, StoredCase, StoredModel, StoredSession
from entities import (
    BenchmarkRuntime,
    Case,
    CaseRecord,
    LoadedCase,
    Rubric,
    RunCaseResult,
    TargetModel,
    dump_json_value,
)
from models.openai import (
    AsyncAzureOpenAI,
    azure_openai_serving_config,
    create_azure_openai_client,
    create_grader_client,
    extract_chat_completion_content,
    extract_usage_counts,
    is_azure_openai_model,
    request_azure_openai_chat_completion,
)
from grader import grade as grade_assistant_content
from models.qwen import (
    Conversation,
    VllmEngineConfig,
    VllmSamplingParams,
    ensure_vllm_server,
    generate_vllm_chat_completion,
)
from utils import init_logging

logger = logging.getLogger(__name__)
PROMPT_SAMPLE_SEED = 0
REQUIRED_CASE_TAG = "physician_agreed_category:not-enough-context"
FOCUS_METRIC_NAME = "axis:context_awareness"
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
BENCHMARK_REPO_ROOT = PROJECT_ROOT
DATASET_PATH = BENCHMARK_REPO_ROOT / "context_seeking_consensus_hard_104.jsonl"
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


def _log_benchmark_runtime(
    *,
    runtime: BenchmarkRuntime,
    local_model_name: str,
    checkpoint: Path | None,
) -> None:
    logger.debug(
        "Benchmark runtime | rollout_size=%d | timeout_seconds=%d | "
        "completion_token_limit=%d | top_p=%.3f | temperature=%.3f | "
        "presence_penalty=%.3f | enable_thinking=%s | max_seq_length=%d | "
        "kv_cache_dtype=%s | gpu_memory_utilization=%.3f | local_model=%s | "
        "checkpoint=%s",
        runtime.rollout_size,
        runtime.generation_timeout_seconds,
        runtime.sampling_params.completion_token_limit,
        runtime.sampling_params.top_p,
        runtime.sampling_params.temperature,
        runtime.sampling_params.presence_penalty,
        runtime.enable_thinking,
        runtime.vllm_engine_config.max_seq_length,
        runtime.vllm_engine_config.kv_cache_dtype,
        runtime.vllm_engine_config.gpu_memory_utilization,
        local_model_name,
        str(checkpoint.expanduser().resolve()) if checkpoint is not None else "",
    )

def _render_question(case: Case) -> str:
    return "\n\n".join(
        f"{message['role']}: {message['content']}" for message in case.prompt
    )


def load_cases(
    dataset_path: Path,
    *,
    case_count: int = 0,
) -> tuple[list[LoadedCase], Rubric]:
    assert dataset_path.is_file(), f"HB rubrics dataset not found: {dataset_path}"
    assert case_count >= 0, "--case-count must be >= 0."

    cases: list[LoadedCase] = []
    rubric: Rubric | None = None
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not (line := raw_line.strip()):
                continue
            case_record = CaseRecord.model_validate_json(line)
            case = Case.model_validate(case_record.model_dump())
            if REQUIRED_CASE_TAG not in case.case_tags:
                continue
            ideal_completions = case_record.ideal_completions_data
            cases.append(
                LoadedCase(
                    case=case,
                    line_number=line_number,
                    raw_json=line,
                    ideal_completions_json=(
                        dump_json_value(ideal_completions)
                        if ideal_completions is not None
                        else None
                    ),
                    question=_render_question(case),
                )
            )

            matching_rubrics = [
                case_rubric
                for case_rubric in case.rubrics
                if FOCUS_METRIC_NAME in case_rubric.tags
            ]
            assert len(matching_rubrics) == 1, (
                f"Expected exactly one context-awareness rubric for {case.prompt_id}."
            )
            case_rubric = matching_rubrics[0]
            if rubric is None:
                rubric = case_rubric
                continue
            assert case_rubric == rubric, (
                "Benchmark tag selection must resolve to one shared "
                "context-awareness rubric."
            )

    assert cases, (
        f"No benchmark cases matched {REQUIRED_CASE_TAG!r} in {dataset_path}."
    )
    assert rubric is not None, (
        "Expected at least one context-awareness rubric in the benchmark slice."
    )

    if case_count == 0:
        return cases, rubric

    assert case_count <= len(cases), (
        f"--case-count={case_count} exceeds available dataset cases "
        f"({len(cases)})."
    )
    return random.Random(PROMPT_SAMPLE_SEED).sample(cases, k=case_count), rubric


async def run_case(
    *,
    target_model: TargetModel,
    loaded_case: LoadedCase,
    rubric: Rubric,
    runtime: BenchmarkRuntime,
    target_client: AsyncAzureOpenAI | None,
    grader: AsyncAzureOpenAI,
    concurrency_limiter: asyncio.Semaphore,
) -> RunCaseResult:
    case = loaded_case.case
    queried_prompt = Conversation(
        case.prompt,
        system_prompt="You are a helpful assistant.",
    )
    async with concurrency_limiter:
        assistant_content = ""
        input_token_count: int | None = None
        cached_input_token_count: int | None = None
        output_token_count: int | None = None

        generation_timeout_seconds = runtime.generation_timeout_seconds
        sampling_params = runtime.sampling_params
        if target_model.backend == "aws-bedrock":
            assert target_model.model_id in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
                f"Unsupported Anthropic model: {target_model.model_id}"
            )
            request_messages = queried_prompt.as_server_payload()
            system_prompt = ""
            top_p = (
                None if sampling_params.temperature > 0 else sampling_params.top_p
            )
            if request_messages and request_messages[0]["role"] == "system":
                system_prompt = request_messages[0]["content"]
                request_messages = request_messages[1:]
            assistant_content = await ask_bedrock(
                model_name=target_model.model_id,
                region=None,
                profile=None,
                max_tokens=sampling_params.completion_token_limit,
                temperature=sampling_params.temperature,
                timeout_seconds=float(generation_timeout_seconds),
                messages=request_messages,
                system_prompt=system_prompt,
                top_p=top_p,
            )
        elif target_model.backend == "azure-openai":
            assert target_client is not None, "Azure target client must be present."
            completion_payload = await request_azure_openai_chat_completion(
                messages=queried_prompt.as_server_payload(),
                model=target_model.model_id,
                max_tokens=sampling_params.completion_token_limit,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                presence_penalty=sampling_params.presence_penalty,
                client=target_client,
            )
            assistant_content = extract_chat_completion_content(completion_payload)
            (
                input_token_count,
                cached_input_token_count,
                output_token_count,
            ) = extract_usage_counts(completion_payload)
        else:
            async with httpx.AsyncClient(
                base_url=VLLM_BASE_URL,
                timeout=float(generation_timeout_seconds),
            ) as vllm:
                _, assistant_content, request_metrics = (
                    await asyncio.wait_for(
                        generate_vllm_chat_completion(
                            model_name=target_model.model_id,
                            base_model_name=target_model.vllm_base_model_name,
                            messages=queried_prompt,
                            vllm_client=vllm,
                            sampling_params=sampling_params,
                            enable_thinking=runtime.enable_thinking,
                        ),
                        timeout=float(generation_timeout_seconds),
                    )
                )
            input_token_count = request_metrics.prompt_tokens
            cached_input_token_count = 0
            output_token_count = request_metrics.completion_tokens
        grade = await grade_assistant_content(
            conversation=queried_prompt.as_server_payload(),
            assistant_response=assistant_content,
            rubric=rubric,
            client=grader,
        )
        return RunCaseResult(
            prompt_id=case.prompt_id,
            score=grade.as_metric(),
            criteria_met=grade.criteria_met,
            grader_explanation=grade.explanation,
            grader_response_json=dump_json_value(grade.raw_response),
            model_response=assistant_content,
            input_token_count=input_token_count,
            cached_input_token_count=cached_input_token_count,
            output_token_count=output_token_count,
        )


async def run_rollouts(
    *,
    experiment_id: str,
    target_model: TargetModel,
    cases: list[LoadedCase],
    rubric: Rubric,
    runtime: BenchmarkRuntime,
    results_db_path: Path,
) -> tuple[str, dict[str, list[float]]]:
    rollout_size = runtime.rollout_size
    results: dict[str, list[float]] = defaultdict(list)
    failure_count = 0
    backend = target_model.backend
    concurrency_limit = 1 if backend in {"aws-bedrock", "azure-openai"} else 12
    concurrency_limiter = asyncio.Semaphore(concurrency_limit)
    store = BenchmarkStore(results_db_path)
    for loaded_case in cases:
        store.upsert_case(
            StoredCase(
                prompt_id=loaded_case.case.prompt_id,
                line_number=loaded_case.line_number,
                question=loaded_case.question,
                raw_json=loaded_case.raw_json,
                ideal_completions_json=loaded_case.ideal_completions_json,
            )
        )
    store.upsert_model(
        StoredModel(
            model_id=target_model.model_id,
            display_name=target_model.display_name,
            provider=target_model.provider,
            backend=target_model.backend,
            checkpoint_path=target_model.checkpoint_path,
            request_parameters_json=target_model.request_parameters_json,
            serving_config_json=target_model.serving_config_json,
        )
    )
    store.commit()
    grader = create_grader_client()
    target_client = (
        create_azure_openai_client()
        if target_model.backend == "azure-openai"
        else None
    )
    try:
        tasks = [
            run_case(
                target_model=target_model,
                loaded_case=loaded_case,
                rubric=rubric,
                runtime=runtime,
                target_client=target_client,
                grader=grader,
                concurrency_limiter=concurrency_limiter,
            )
            for loaded_case in cases
            for rollout_index in range(rollout_size)
        ]
        task_keys = [
            (loaded_case.case.prompt_id, rollout_index)
            for loaded_case in cases
            for rollout_index in range(rollout_size)
        ]
        for task_key, result in zip(
            task_keys,
            await asyncio.gather(*tasks, return_exceptions=True),
            strict=True,
        ):
            prompt_id, rollout_index = task_key
            if isinstance(result, BaseException):
                logger.error(
                    "Benchmark rollout failed | backend=%s | prompt_id=%s | "
                    "rollout_index=%d | error_type=%s | error=%s",
                    backend,
                    prompt_id,
                    rollout_index,
                    type(result).__name__,
                    result,
                )
                failure_count += 1
                continue
            store.insert_session(
                StoredSession(
                    model_id=target_model.model_id,
                    prompt_id=result.prompt_id,
                    rollout_index=rollout_index,
                    model_response=result.model_response,
                    grader_response_json=result.grader_response_json,
                    grader_explanation=result.grader_explanation,
                    criteria_met=int(result.criteria_met),
                    score=result.score,
                    input_token_count=result.input_token_count,
                    cached_input_token_count=result.cached_input_token_count,
                    output_token_count=result.output_token_count,
                )
            )
            results[result.prompt_id].append(result.score)
        store.commit()
    finally:
        if target_client is not None:
            await target_client.close()
        await grader.close()
        store.close()
    if failure_count > 0:
        logger.warning("%d benchmark rollout(s) failed.", failure_count)
    return experiment_id, results


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
    _log_benchmark_runtime(
        runtime=runtime,
        local_model_name=local_model_name,
        checkpoint=local_checkpoint,
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
            host=VLLM_HOST,
            requested_port=VLLM_PORT,
            startup_timeout_seconds=600.0,
            log_path=run_log_path.with_suffix(".vllm.log"),
        )
        assert vllm_handle.base_url == VLLM_BASE_URL, (
            f"Expected vLLM server at {VLLM_BASE_URL}, got {vllm_handle.base_url}."
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
            repo_root=BENCHMARK_REPO_ROOT,
            run_log_path=run_log_path,
            results_db_path=results_db,
        )
    )


if __name__ == "__main__":
    cli()
