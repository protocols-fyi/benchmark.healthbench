"""Case sampling and rollout execution for HealthBench benchmarks."""

import asyncio
from collections import defaultdict
import json
import logging
from pathlib import Path
import random

from config import (
    AWS_BEDROCK_SUPPORTED_MODEL_IDS,
    PROMPT_SAMPLE_SEED,
    REQUIRED_CASE_TAG,
)
from db import open_benchmark_db
from entities import Case, Grade, TargetModel
from grader import grade as grade_assistant_content
from models.anthropic import ask as ask_anthropic
from models.openai import ask as ask_azure_openai
from models.qwen import ChatMessage, ask as ask_vllm

logger = logging.getLogger(__name__)


def load_cases(
    dataset_path: Path,
    *,
    case_count: int = 0,
) -> list[Case]:
    assert dataset_path.is_file(), f"HealthBench dataset not found: {dataset_path}"
    assert case_count >= 0, "--case-count must be >= 0."

    cases: list[Case] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not (line := raw_line.strip()):
                continue

            payload = json.loads(line)
            example_tags = payload.get("example_tags") or ()
            if REQUIRED_CASE_TAG not in example_tags:
                continue

            prompt_id = payload["prompt_id"]
            raw_prompt = payload["prompt"]
            assert isinstance(prompt_id, str) and prompt_id, "Missing prompt_id."
            assert isinstance(raw_prompt, list) and raw_prompt, (
                f"Prompt must be non-empty for {prompt_id}."
            )
            prompt = tuple(raw_prompt)
            assert prompt[-1]["role"] == "user", (
                f"Prompt must end with a user turn for {prompt_id}."
            )
            for message in prompt:
                assert message["role"] != "system", (
                    f"Prompt messages must not use the system role for {prompt_id}."
                )
                assert message["content"].strip(), (
                    f"Prompt messages must be non-empty for {prompt_id}."
                )
                assert "choice" not in message, (
                    f"Prompt messages must not include choices for {prompt_id}."
                )
            cases.append(
                Case(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    raw_json=line,
                )
            )

    assert cases, (
        f"No benchmark cases matched {REQUIRED_CASE_TAG!r} in {dataset_path}."
    )
    if case_count == 0:
        return cases

    assert case_count <= len(cases), (
        f"--case-count={case_count} exceeds available dataset cases "
        f"({len(cases)})."
    )
    return random.Random(PROMPT_SAMPLE_SEED).sample(cases, k=case_count)


async def _execute_case(
    *,
    target_model: TargetModel,
    case: Case,
    concurrency_limiter: asyncio.Semaphore,
) -> Grade:
    queried_prompt: list[ChatMessage] = [
        {"role": "system", "content": "You are a helpful assistant."},
        *case.prompt,
    ]
    server_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in queried_prompt
    ]

    async with concurrency_limiter:
        if target_model.backend == "aws-bedrock":
            assert target_model.model_id in AWS_BEDROCK_SUPPORTED_MODEL_IDS, (
                f"Unsupported Anthropic model: {target_model.model_id}"
            )
            request_messages = server_messages
            system_prompt = ""
            if request_messages and request_messages[0]["role"] == "system":
                system_prompt = request_messages[0]["content"]
                request_messages = request_messages[1:]
            (
                assistant_content,
                input_token_count,
                cached_input_token_count,
                output_token_count,
            ) = await ask_anthropic(
                model_name=target_model.model_id,
                messages=request_messages,
                system_prompt=system_prompt,
            )
        elif target_model.backend == "azure-openai":
            (
                assistant_content,
                input_token_count,
                cached_input_token_count,
                output_token_count,
            ) = await ask_azure_openai(
                model_name=target_model.model_id,
                messages=server_messages,
            )
        else:
            (
                assistant_content,
                input_token_count,
                cached_input_token_count,
                output_token_count,
            ) = await ask_vllm(
                model_name=target_model.model_id,
                messages=queried_prompt,
            )

        criteria_met, grader_explanation, grader_response_json = (
            await grade_assistant_content(
                conversation=server_messages,
                assistant_response=assistant_content,
            )
        )
        return Grade(
            prompt_id=case.prompt_id,
            criteria_met=criteria_met,
            grader_explanation=grader_explanation,
            grader_response_json=grader_response_json,
            model_response=assistant_content,
            input_token_count=input_token_count,
            cached_input_token_count=cached_input_token_count,
            output_token_count=output_token_count,
        )


async def execute(
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
    concurrency_limit = 4
    concurrency_limiter = asyncio.Semaphore(concurrency_limit)
    db = open_benchmark_db(results_db_path)
    for case in cases:
        db["cases"].insert(
            {
                "prompt_id": case.prompt_id,
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
    try:
        pending_tasks: dict[asyncio.Task[Grade], tuple[Case, int]] = {}
        for case in cases:
            for rollout_index in range(rollout_size):
                task = asyncio.create_task(
                    _execute_case(
                        target_model=target_model,
                        case=case,
                        concurrency_limiter=concurrency_limiter,
                    )
                )
                pending_tasks[task] = (case, rollout_index)
        while pending_tasks:
            done_tasks, _ = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            wrote_sessions = False
            for task in done_tasks:
                case, rollout_index = pending_tasks.pop(task)
                try:
                    result = task.result()
                except BaseException as error:
                    logger.error(
                        "Benchmark rollout failed | backend=%s | prompt_id=%s | "
                        "rollout_index=%d | question=%s | error_type=%s | error=%s",
                        backend,
                        case.prompt_id,
                        rollout_index,
                        case.question,
                        type(error).__name__,
                        error,
                    )
                    failure_count += 1
                    continue

                score = 1.0 if result.criteria_met else 0.0
                db["sessions"].insert(
                    {
                        "model_id": target_model.model_id,
                        "prompt_id": result.prompt_id,
                        "rollout_index": rollout_index,
                        "model_response": result.model_response,
                        "grader_response_json": result.grader_response_json,
                        "grader_explanation": result.grader_explanation,
                        "criteria_met": int(result.criteria_met),
                        "score": score,
                        "input_token_count": result.input_token_count,
                        "cached_input_token_count": result.cached_input_token_count,
                        "output_token_count": result.output_token_count,
                    }
                )
                wrote_sessions = True
                results[result.prompt_id].append(score)
                logger.info(
                    "Completed benchmark rollout | backend=%s | model_id=%s | "
                    "prompt_id=%s | rollout_index=%d | question=%s | model_response=%s | "
                    "criteria_met=%s | score=%.1f | grader_explanation=%s | "
                    "input_token_count=%d | cached_input_token_count=%d | "
                    "output_token_count=%d",
                    backend,
                    target_model.model_id,
                    result.prompt_id,
                    rollout_index,
                    case.question,
                    result.model_response,
                    result.criteria_met,
                    score,
                    result.grader_explanation,
                    result.input_token_count,
                    result.cached_input_token_count,
                    result.output_token_count,
                )
            if wrote_sessions:
                db.conn.commit()
    finally:
        db.conn.close()
    if failure_count > 0:
        logger.warning("%d benchmark rollout(s) failed.", failure_count)
    return experiment_id, results
