"""Case sampling and rollout execution for HealthBench benchmarks."""

import asyncio
from collections import defaultdict
import json
import logging
from pathlib import Path
import random

import httpx
from models.anthropic import ask_bedrock

from config import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    DEFAULT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
    DEFAULT_VLLM_PRESENCE_PENALTY,
    DEFAULT_VLLM_TEMPERATURE,
    DEFAULT_VLLM_TOP_P,
    FOCUS_METRIC_NAME,
    PROMPT_SAMPLE_SEED,
    REQUIRED_CASE_TAG,
    VLLM_BASE_URL,
)
from db import open_benchmark_db
from entities import (
    Case,
    Rubric,
    RunCaseResult,
    TargetModel,
    dump_json_value,
)
from grader import (
    create_grader_client,
    grade as grade_assistant_content,
)
from models.openai import (
    AsyncAzureOpenAI,
    create_azure_openai_client,
    extract_chat_completion_content,
    extract_usage_counts,
    request_azure_openai_chat_completion,
)
from models.qwen import (
    ChatMessage,
    generate_vllm_chat_completion,
)

logger = logging.getLogger(__name__)


def load_cases(
    dataset_path: Path,
    *,
    case_count: int = 0,
) -> list[Case]:
    assert dataset_path.is_file(), f"HealthBench dataset not found: {dataset_path}"
    assert case_count >= 0, "--case-count must be >= 0."

    cases: list[Case] = []
    shared_rubric: Rubric | None = None
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not (line := raw_line.strip()):
                continue
            payload = json.loads(line)
            case_tags = tuple(payload.get("example_tags") or ())
            if REQUIRED_CASE_TAG not in case_tags:
                continue

            raw_rubrics = payload["rubrics"]
            assert isinstance(raw_rubrics, list) and raw_rubrics, (
                f"Expected rubrics list on line {line_number}."
            )
            matching_rubrics: list[Rubric] = []
            for raw_rubric in raw_rubrics:
                assert isinstance(raw_rubric, dict), (
                    f"Expected rubric object on line {line_number}."
                )
                criterion = raw_rubric.get("criterion")
                points = raw_rubric.get("points")
                raw_tags = raw_rubric.get("tags")
                assert isinstance(criterion, str) and criterion.strip(), (
                    f"Expected non-empty rubric criterion on line {line_number}."
                )
                assert isinstance(points, int | float) and points != 0, (
                    f"Expected non-zero rubric points on line {line_number}."
                )
                assert isinstance(raw_tags, list), (
                    f"Expected rubric tags list on line {line_number}."
                )
                tags = tuple(str(tag) for tag in raw_tags)
                if FOCUS_METRIC_NAME not in tags:
                    continue
                matching_rubrics.append(
                    Rubric(
                        criterion=criterion,
                        points=float(points),
                        tags=tags,
                    )
                )
            prompt_id = payload["prompt_id"]
            assert len(matching_rubrics) == 1, (
                f"Expected exactly one context-awareness rubric for {prompt_id}."
            )
            case_rubric = matching_rubrics[0]
            if shared_rubric is None:
                shared_rubric = case_rubric
            else:
                assert case_rubric == shared_rubric, (
                    "Benchmark tag selection must resolve to one shared "
                    "context-awareness rubric."
                )

            raw_prompt = payload["prompt"]
            assert isinstance(prompt_id, str) and prompt_id, (
                f"Missing prompt_id on line {line_number}."
            )
            assert isinstance(raw_prompt, list) and raw_prompt, (
                f"Prompt must be non-empty on line {line_number}."
            )
            prompt = tuple(raw_prompt)
            assert prompt[-1]["role"] == "user", (
                f"Prompt must end with a user turn on line {line_number}."
            )
            for message in prompt:
                assert message["role"] != "system", (
                    f"Prompt messages must not use the system role on line {line_number}."
                )
                assert message["content"].strip(), (
                    f"Prompt messages must be non-empty on line {line_number}."
                )
                assert "choice" not in message, (
                    f"Prompt messages must not include choices on line {line_number}."
                )
            cases.append(
                Case(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    case_tags=case_tags,
                    rubric=case_rubric,
                    line_number=line_number,
                    raw_json=line,
                )
            )

    assert cases, (
        f"No benchmark cases matched {REQUIRED_CASE_TAG!r} in {dataset_path}."
    )
    assert shared_rubric is not None, (
        "Expected at least one context-awareness rubric in the benchmark slice."
    )

    if case_count == 0:
        return cases

    assert case_count <= len(cases), (
        f"--case-count={case_count} exceeds available dataset cases "
        f"({len(cases)})."
    )
    return random.Random(PROMPT_SAMPLE_SEED).sample(cases, k=case_count)


async def run_case(
    *,
    target_model: TargetModel,
    case: Case,
    target_client: AsyncAzureOpenAI | None,
    grader: AsyncAzureOpenAI,
    concurrency_limiter: asyncio.Semaphore,
) -> RunCaseResult:
    queried_prompt: list[ChatMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        *case.prompt,
    ]
    server_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in queried_prompt
    ]
    async with concurrency_limiter:
        assistant_content = ""
        input_token_count: int | None = None
        cached_input_token_count: int | None = None
        output_token_count: int | None = None

        generation_timeout_seconds = DEFAULT_GENERATION_TIMEOUT_SECONDS
        if target_model.backend == "aws-bedrock":
            assert target_model.model_id in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
                f"Unsupported Anthropic model: {target_model.model_id}"
            )
            request_messages = server_messages
            system_prompt = ""
            top_p = None if DEFAULT_VLLM_TEMPERATURE > 0 else DEFAULT_VLLM_TOP_P
            if request_messages and request_messages[0]["role"] == "system":
                system_prompt = request_messages[0]["content"]
                request_messages = request_messages[1:]
            assistant_content = await ask_bedrock(
                model_name=target_model.model_id,
                region=None,
                profile=None,
                max_tokens=DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
                temperature=DEFAULT_VLLM_TEMPERATURE,
                timeout_seconds=float(generation_timeout_seconds),
                messages=request_messages,
                system_prompt=system_prompt,
                top_p=top_p,
            )
        elif target_model.backend == "azure-openai":
            assert target_client is not None, "Azure target client must be present."
            completion_payload = await request_azure_openai_chat_completion(
                messages=server_messages,
                model=target_model.model_id,
                max_tokens=DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT,
                temperature=DEFAULT_VLLM_TEMPERATURE,
                top_p=DEFAULT_VLLM_TOP_P,
                presence_penalty=DEFAULT_VLLM_PRESENCE_PENALTY,
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
                assistant_content, input_token_count, output_token_count = (
                    await asyncio.wait_for(
                        generate_vllm_chat_completion(
                            model_name=target_model.model_id,
                            base_model_name=target_model.vllm_base_model_name,
                            messages=queried_prompt,
                            vllm_client=vllm,
                        ),
                        timeout=float(generation_timeout_seconds),
                    )
                )
            cached_input_token_count = 0
        grade = await grade_assistant_content(
            conversation=server_messages,
            assistant_response=assistant_content,
            rubric=case.rubric,
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
    cases: list[Case],
    rollout_size: int,
    results_db_path: Path,
) -> tuple[str, dict[str, list[float]]]:
    results: dict[str, list[float]] = defaultdict(list)
    failure_count = 0
    backend = target_model.backend
    concurrency_limit = 1 if backend in {"aws-bedrock", "azure-openai"} else 12
    concurrency_limiter = asyncio.Semaphore(concurrency_limit)
    db = open_benchmark_db(results_db_path)
    for case in cases:
        db["cases"].insert(
            {
                "prompt_id": case.prompt_id,
                "line_number": case.line_number,
                "question": case.question,
                "raw_json": case.raw_json,
            },
            replace=True,
        )
    db["models"].insert(
        {
            "model_id": target_model.model_id,
            "display_name": target_model.display_name,
            "provider": target_model.provider,
            "backend": target_model.backend,
            "checkpoint_path": target_model.checkpoint_path,
            "request_parameters_json": target_model.request_parameters_json,
            "serving_config_json": target_model.serving_config_json,
        },
        replace=True,
    )
    db.conn.commit()
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
                case=case,
                target_client=target_client,
                grader=grader,
                concurrency_limiter=concurrency_limiter,
            )
            for case in cases
            for rollout_index in range(rollout_size)
        ]
        task_keys = [
            (case.prompt_id, rollout_index)
            for case in cases
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
            db["sessions"].insert(
                {
                    "model_id": target_model.model_id,
                    "prompt_id": result.prompt_id,
                    "rollout_index": rollout_index,
                    "model_response": result.model_response,
                    "grader_response_json": result.grader_response_json,
                    "grader_explanation": result.grader_explanation,
                    "criteria_met": int(result.criteria_met),
                    "score": result.score,
                    "input_token_count": result.input_token_count,
                    "cached_input_token_count": result.cached_input_token_count,
                    "output_token_count": result.output_token_count,
                }
            )
            results[result.prompt_id].append(result.score)
        db.conn.commit()
    finally:
        if target_client is not None:
            await target_client.close()
        await grader.close()
        db.conn.close()
    if failure_count > 0:
        logger.warning("%d benchmark rollout(s) failed.", failure_count)
    return experiment_id, results
