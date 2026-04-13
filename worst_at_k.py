"""Export per-model HealthBench worst-at-k curves from benchmark SQLite results.

Usage:
    uv run python worst_at_k.py --results-db ./results.sqlite3
    uv run python worst_at_k.py --results-db ./results.sqlite3 --model gpt-4.1
    uv run python worst_at_k.py --results-db ./results.sqlite3 --model gpt-4.1 --model qwen3-8b

The output is wide CSV so it can be pasted directly into Excel. Each row is a
model, and each `worst_at_<k>` column contains the overall HealthBench
worst-at-k score for that model.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import sqlite3
import sys

DEFAULT_RESULTS_DB_PATH = Path("./results.sqlite3")


def load_session_scores(
    *,
    results_db_path: Path,
    model_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    resolved_db_path = results_db_path.expanduser().resolve()
    assert resolved_db_path.is_file(), (
        f"Results database not found: {resolved_db_path}"
    )

    where_clause = ""
    query_parameters: tuple[str, ...] = ()
    if model_ids:
        placeholders = ", ".join("?" for _ in model_ids)
        where_clause = f"WHERE s.model_id IN ({placeholders})"
        query_parameters = model_ids

    connection = sqlite3.connect(resolved_db_path)
    connection.row_factory = sqlite3.Row
    try:
        session_rows = connection.execute(
            f"""
            SELECT
                s.model_id,
                m.display_name,
                m.provider,
                m.backend,
                s.prompt_id,
                s.score
            FROM sessions AS s
            INNER JOIN models AS m ON m.model_id = s.model_id
            {where_clause}
            ORDER BY s.model_id, s.prompt_id, s.session_id
            """,
            query_parameters,
        ).fetchall()
    finally:
        connection.close()

    requested_models_text = ", ".join(model_ids)
    assert session_rows, (
        "No benchmark sessions found"
        + (f" for model(s): {requested_models_text}" if model_ids else "")
        + f" in {resolved_db_path}."
    )

    scores_by_model: dict[str, dict[str, object]] = {}
    for row in session_rows:
        model_id = str(row["model_id"])
        prompt_id = str(row["prompt_id"])
        score = float(row["score"])
        assert 0.0 <= score <= 1.0, (
            f"Score must be in [0, 1] for model={model_id} prompt_id={prompt_id}, "
            f"got {score}."
        )
        if model_id not in scores_by_model:
            scores_by_model[model_id] = {
                "display_name": str(row["display_name"]),
                "provider": str(row["provider"]),
                "backend": str(row["backend"]),
                "prompt_scores": defaultdict(list),
            }
        prompt_scores = scores_by_model[model_id]["prompt_scores"]
        assert isinstance(prompt_scores, defaultdict)
        prompt_scores[prompt_id].append(score)

    return scores_by_model


def compute_case_worst_at_k(scores: list[float], k: int) -> float:
    sample_count = len(scores)
    assert 1 <= k <= sample_count, (
        f"k must be between 1 and the number of samples ({sample_count}), got {k}."
    )

    sorted_scores = sorted(scores)
    denominator = math.comb(sample_count, k)
    weighted_minimum_sum = 0.0
    for index, score in enumerate(sorted_scores[: sample_count - k + 1]):
        weighted_minimum_sum += score * math.comb(sample_count - index - 1, k - 1)
    return weighted_minimum_sum / denominator


def build_export_rows(
    scores_by_model: dict[str, dict[str, object]],
) -> tuple[list[dict[str, str]], int]:
    export_rows: list[dict[str, str]] = []
    global_max_supported_k = 0

    for model_id in sorted(scores_by_model):
        model_data = scores_by_model[model_id]
        prompt_scores = model_data["prompt_scores"]
        assert isinstance(prompt_scores, defaultdict)
        assert prompt_scores, f"Model {model_id} has no prompt scores."
        sample_counts = [len(scores) for scores in prompt_scores.values()]
        min_case_sample_count = min(sample_counts)
        max_case_sample_count = max(sample_counts)
        global_max_supported_k = max(global_max_supported_k, min_case_sample_count)

        row = {
            "model_id": model_id,
            "display_name": str(model_data["display_name"]),
            "provider": str(model_data["provider"]),
            "backend": str(model_data["backend"]),
            "prompt_count": str(len(prompt_scores)),
            "min_case_sample_count": str(min_case_sample_count),
            "max_case_sample_count": str(max_case_sample_count),
        }
        for k in range(1, min_case_sample_count + 1):
            case_scores = [
                compute_case_worst_at_k(scores, k) for scores in prompt_scores.values()
            ]
            overall_score = sum(case_scores) / len(case_scores)
            row[f"worst_at_{k}"] = f"{min(1.0, max(0.0, overall_score)):.6f}"
        export_rows.append(row)

    return export_rows, global_max_supported_k


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-model HealthBench worst-at-k curves from a benchmark "
            "SQLite database as wide CSV."
        )
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=DEFAULT_RESULTS_DB_PATH,
        help="SQLite file to read benchmark sessions from.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Optional model_id filter. Repeat --model to export multiple models. "
            "When omitted, exports every model in the database."
        ),
    )
    args = parser.parse_args()

    scores_by_model = load_session_scores(
        results_db_path=args.results_db,
        model_ids=tuple(args.model),
    )
    export_rows, global_max_supported_k = build_export_rows(scores_by_model)
    fieldnames = [
        "model_id",
        "display_name",
        "provider",
        "backend",
        "prompt_count",
        "min_case_sample_count",
        "max_case_sample_count",
        *[f"worst_at_{k}" for k in range(1, global_max_supported_k + 1)],
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(export_rows)


if __name__ == "__main__":
    main()
