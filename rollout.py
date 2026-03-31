"""Case sampling and rollout execution for HealthBench benchmarks."""

import asyncio
from collections import defaultdict
import logging
from pathlib import Path
import random

import httpx
from models.anthropic import ask_bedrock

from config import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    FOCUS_METRIC_NAME,
    PROMPT_SAMPLE_SEED,
    REQUIRED_CASE_TAG,
    VLLM_BASE_URL,
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
from grader import grade as grade_assistant_content
from models.openai import (
    AsyncAzureOpenAI,
    create_azure_openai_client,
    create_grader_client,
    extract_chat_completion_content,
    extract_usage_counts,
    request_azure_openai_chat_completion,
)
from models.qwen import Conversation, generate_vllm_chat_completion

logger = logging.getLogger(__name__)


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
