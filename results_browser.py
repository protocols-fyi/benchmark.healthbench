"""Local FastHTML app for browsing HealthBench benchmark results.

Usage:
    uv run --env-file .env python results_browser.py --results-db ./results.sqlite3
    uv run --env-file .env python results_browser.py --results-db ./run/main.all.sqlite3 --port 8080

The app is read-only. It opens the benchmark SQLite database, lists models,
cases, and recent rollouts, and lets you inspect the full message sequence for
an individual rollout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fasthtml.common import (
    A,
    Code,
    Div,
    H2,
    P,
    Pre,
    Style,
    Table,
    Tbody,
    Td,
    Th,
    Thead,
    Titled,
    Tr,
    fast_app,
    serve,
)
from sqlite_utils import Database

from db import open_benchmark_db

RUNTIME_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_RESULTS_DB_PATH = Path("./results.sqlite3")
RESULTS_DB_ENV_KEY = "HEALTHBENCH_RESULTS_DB"
RESULTS_DB_PATH: Path | None = None
APP_CSS = """
html,
body {
    background: #ffffff;
    color: #111111;
}

body,
input,
select,
button,
table,
th,
td,
pre,
code {
    font: 13px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}

main.container {
    max-width: none;
    padding: 12px;
}

h1 {
    margin: 0 0 8px;
    font-size: 24px;
    line-height: 1.1;
    font-weight: 700;
}

h2 {
    margin: 16px 0 6px;
    font-size: 16px;
    line-height: 1.2;
    font-weight: 700;
}

p {
    margin: 0 0 8px;
}

a {
    color: #0000cc;
    text-decoration: underline;
}

form {
    display: flex;
    flex-wrap: wrap;
    align-items: end;
    gap: 8px;
    margin: 0 0 12px;
}

label {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 12px;
    font-weight: 700;
}

input,
select,
button {
    background: #ffffff;
    color: #111111;
    border: 1px solid #666666;
    border-radius: 0;
    box-shadow: none;
    padding: 3px 6px;
}

button {
    cursor: pointer;
    width: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 0 0 12px;
}

th,
td {
    border: 1px solid #999999;
    background: #ffffff;
    padding: 4px 6px;
    text-align: left;
    vertical-align: top;
}

th {
    background: #dddddd;
    font-size: 12px;
    font-weight: 700;
}

pre {
    margin: 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font: inherit;
}

code {
    padding: 0 2px;
}

.mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
}

.small {
    font-size: 12px;
    color: #111111;
}

.pass {
    color: #005500;
    font-weight: 700;
}

.fail {
    color: #7a0000;
    font-weight: 700;
}
""".strip()

app, rt = fast_app(
    title="HealthBench Results Browser",
    hdrs=(Style(APP_CSS),),
    pico=False,
)


def database() -> Database:
    assert RESULTS_DB_PATH is not None, "Results database is not initialized."
    return open_benchmark_db(RESULTS_DB_PATH)


def configure_results_database(db_path: Path) -> None:
    resolved_db_path = db_path.expanduser().resolve()
    assert resolved_db_path.is_file(), (
        f"Results database not found: {resolved_db_path}"
    )

    global RESULTS_DB_PATH
    RESULTS_DB_PATH = resolved_db_path


def truncate_text(text: str, limit: int) -> str:
    assert limit > 3, "limit must leave room for ellipsis."
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def result_text(*, criteria_met: bool | int, score: float) -> Div:
    passed = bool(criteria_met)
    return Div(
        f"{'pass' if passed else 'fail'} {score:.3f}",
        cls="pass" if passed else "fail",
    )


def detail_table(rows: list[tuple[str, object]]) -> Table:
    return Table(
        Tbody(
            *[
                Tr(
                    Th(label),
                    Td(value),
                )
                for label, value in rows
            ]
        )
    )


def transcript_table(rows: list[tuple[int, str, str, str]]) -> Table:
    return Table(
        Thead(
            Tr(
                Th("#"),
                Th("Role"),
                Th("Context"),
                Th("Content"),
            )
        ),
        Tbody(
            *[
                Tr(
                    Td(str(index)),
                    Td(role),
                    Td(note or ""),
                    Td(Pre(content)),
                )
                for index, role, note, content in rows
            ]
        ),
    )


@rt("/")
def home_page(model_id: str = "", prompt_id: str = ""):
    db = database()
    try:
        summary = next(
            db.query(
                """
                SELECT
                    (SELECT COUNT(*) FROM models) AS model_count,
                    (SELECT COUNT(*) FROM cases) AS case_count,
                    (SELECT COUNT(*) FROM sessions) AS session_count
                """
            )
        )
        model_rows = list(
            db.query(
                """
                SELECT
                    m.model_id,
                    m.provider,
                    m.backend,
                    COUNT(s.session_id) AS session_count,
                    AVG(s.score) AS mean_score,
                    MAX(s.created_at) AS latest_rollout_at
                FROM models AS m
                LEFT JOIN sessions AS s ON s.model_id = m.model_id
                GROUP BY m.model_id, m.provider, m.backend
                ORDER BY latest_rollout_at DESC, m.model_id
                """
            )
        )
        case_rows = list(
            db.query(
                """
                SELECT
                    c.prompt_id,
                    c.question,
                    COUNT(s.session_id) AS session_count,
                    COUNT(DISTINCT s.model_id) AS model_count,
                    MAX(s.created_at) AS latest_rollout_at
                FROM cases AS c
                LEFT JOIN sessions AS s ON s.prompt_id = c.prompt_id
                GROUP BY c.prompt_id, c.question
                ORDER BY latest_rollout_at DESC, c.prompt_id
                """
            )
        )
        rollout_rows = list(
            db.query(
                """
                SELECT
                    s.session_id,
                    s.model_id,
                    s.prompt_id,
                    s.rollout_index,
                    s.criteria_met,
                    s.score,
                    s.created_at,
                    c.question
                FROM sessions AS s
                JOIN cases AS c ON c.prompt_id = s.prompt_id
                WHERE (:model_id = '' OR s.model_id = :model_id)
                  AND (:prompt_id = '' OR s.prompt_id = :prompt_id)
                ORDER BY s.created_at DESC, s.session_id DESC
                LIMIT 200
                """,
                {"model_id": model_id, "prompt_id": prompt_id},
            )
        )
        return Titled(
            "HealthBench Results Browser",
            P(
                "Read-only viewer for ",
                Code(str(RESULTS_DB_PATH), cls="mono"),
                ".",
            ),
            detail_table(
                [
                    ("Models", str(summary["model_count"])),
                    ("Cases", str(summary["case_count"])),
                    ("Sessions", str(summary["session_count"])),
                ]
            ),
            (
                P(
                    "view: ",
                    f"model_id={model_id or 'all'}; ",
                    f"prompt_id={prompt_id or 'all'}; ",
                    A("clear", href="/"),
                    cls="small",
                )
                if model_id or prompt_id
                else ""
            ),
            H2("Models"),
            Table(
                Thead(
                    Tr(
                        Th("Model"),
                        Th("Provider"),
                        Th("Backend"),
                        Th("Sessions"),
                        Th("Mean Score"),
                        Th("Latest Rollout"),
                    )
                ),
                Tbody(
                    *[
                        Tr(
                            Td(
                                A(
                                    row["model_id"],
                                    href=f"/?model_id={row['model_id']}",
                                    cls="mono",
                                )
                            ),
                            Td(row["provider"]),
                            Td(row["backend"]),
                            Td(str(row["session_count"])),
                            Td(
                                f"{row['mean_score']:.3f}"
                                if row["mean_score"] is not None
                                else "—"
                            ),
                            Td(row["latest_rollout_at"] or "—"),
                        )
                        for row in model_rows
                    ]
                    if model_rows
                    else [Tr(Td("No model rows found.", colspan="6"))]
                ),
            ),
            H2("Cases"),
            Table(
                Thead(
                    Tr(
                        Th("Prompt"),
                        Th("Question"),
                        Th("Sessions"),
                        Th("Models"),
                        Th("Latest Rollout"),
                    )
                ),
                Tbody(
                    *[
                        Tr(
                            Td(
                                Div(
                                    A(
                                        row["prompt_id"],
                                        href=f"/cases/{row['prompt_id']}",
                                        cls="mono",
                                    ),
                                    Div(
                                        A(
                                            "filter",
                                            href=f"/?prompt_id={row['prompt_id']}",
                                        ),
                                        cls="small",
                                    ),
                                )
                            ),
                            Td(truncate_text(row["question"], 220)),
                            Td(str(row["session_count"])),
                            Td(str(row["model_count"])),
                            Td(row["latest_rollout_at"] or "—"),
                        )
                        for row in case_rows
                    ]
                    if case_rows
                    else [Tr(Td("No case rows found.", colspan="5"))]
                ),
            ),
            H2("Recent Rollouts"),
            Table(
                Thead(
                    Tr(
                        Th("Session"),
                        Th("Model"),
                        Th("Prompt"),
                        Th("Rollout"),
                        Th("Result"),
                        Th("Question"),
                        Th("Created"),
                    )
                ),
                Tbody(
                    *[
                        Tr(
                            Td(
                                A(
                                    str(row["session_id"]),
                                    href=f"/sessions/{row['session_id']}",
                                    cls="mono",
                                )
                            ),
                            Td(
                                A(
                                    row["model_id"],
                                    href=f"/?model_id={row['model_id']}",
                                    cls="mono",
                                )
                            ),
                            Td(
                                A(
                                    row["prompt_id"],
                                    href=f"/cases/{row['prompt_id']}",
                                    cls="mono",
                                )
                            ),
                            Td(str(row["rollout_index"])),
                            Td(
                                result_text(
                                    criteria_met=row["criteria_met"],
                                    score=row["score"],
                                )
                            ),
                            Td(truncate_text(row["question"], 180)),
                            Td(row["created_at"]),
                        )
                        for row in rollout_rows
                    ]
                    if rollout_rows
                    else [Tr(Td("No sessions matched the active filters.", colspan="7"))]
                ),
            ),
        )
    finally:
        db.conn.close()


@rt("/cases/{prompt_id}")
def case_detail_page(prompt_id: str):
    db = database()
    try:
        case_rows = list(
            db.query(
                """
                SELECT
                    c.prompt_id,
                    c.question,
                    c.raw_json,
                    COUNT(s.session_id) AS session_count
                FROM cases AS c
                LEFT JOIN sessions AS s ON s.prompt_id = c.prompt_id
                WHERE c.prompt_id = :prompt_id
                GROUP BY c.prompt_id, c.question, c.raw_json
                """,
                {"prompt_id": prompt_id},
            )
        )
        assert case_rows, f"Unknown prompt_id: {prompt_id}"
        case_row = case_rows[0]
        raw_payload = json.loads(case_row["raw_json"])
        prompt_messages = raw_payload.get("prompt")
        assert isinstance(prompt_messages, list) and prompt_messages, (
            f"Case {prompt_id} must contain a non-empty prompt list."
        )
        session_rows = list(
            db.query(
                """
                SELECT
                    session_id,
                    model_id,
                    rollout_index,
                    criteria_met,
                    score,
                    created_at
                FROM sessions
                WHERE prompt_id = :prompt_id
                ORDER BY created_at DESC, session_id DESC
                """,
                {"prompt_id": prompt_id},
            )
        )

        return Titled(
            f"Case {prompt_id}",
            P(
                A("overview", href="/"),
                " | ",
                A("filter this case", href=f"/?prompt_id={prompt_id}"),
            ),
            H2("Case"),
            detail_table(
                [
                    ("Prompt ID", Code(case_row["prompt_id"], cls="mono")),
                    ("Stored Sessions", str(case_row["session_count"])),
                    ("Question", Pre(case_row["question"])),
                ]
            ),
            H2("Prompt Messages"),
            transcript_table(
                [
                    (index, message["role"], "", message["content"])
                    for index, message in enumerate(prompt_messages, start=1)
                ]
            ),
            H2("Rollouts"),
            Table(
                Thead(
                    Tr(
                        Th("Session"),
                        Th("Model"),
                        Th("Rollout"),
                        Th("Result"),
                        Th("Created"),
                    )
                ),
                Tbody(
                    *[
                        Tr(
                            Td(
                                A(
                                    str(row["session_id"]),
                                    href=f"/sessions/{row['session_id']}",
                                    cls="mono",
                                )
                            ),
                            Td(row["model_id"], cls="mono"),
                            Td(str(row["rollout_index"])),
                            Td(
                                result_text(
                                    criteria_met=row["criteria_met"],
                                    score=row["score"],
                                )
                            ),
                            Td(row["created_at"]),
                        )
                        for row in session_rows
                    ]
                    if session_rows
                    else [Tr(Td("No rollout rows stored for this case yet.", colspan="5"))]
                ),
            ),
        )
    finally:
        db.conn.close()


@rt("/sessions/{session_id}")
def session_detail_page(session_id: int):
    db = database()
    try:
        session_rows = list(
            db.query(
                """
                SELECT
                    s.session_id,
                    s.model_id,
                    s.prompt_id,
                    s.rollout_index,
                    s.model_response,
                    s.grader_response_json,
                    s.grader_explanation,
                    s.criteria_met,
                    s.score,
                    s.input_token_count,
                    s.cached_input_token_count,
                    s.output_token_count,
                    s.created_at,
                    c.question,
                    c.raw_json,
                    m.provider,
                    m.backend
                FROM sessions AS s
                JOIN cases AS c ON c.prompt_id = s.prompt_id
                JOIN models AS m ON m.model_id = s.model_id
                WHERE s.session_id = :session_id
                """,
                {"session_id": session_id},
            )
        )
        assert session_rows, f"Unknown session_id: {session_id}"
        session_row = session_rows[0]
        raw_payload = json.loads(session_row["raw_json"])
        prompt_messages = raw_payload.get("prompt")
        assert isinstance(prompt_messages, list) and prompt_messages, (
            f"Case {session_row['prompt_id']} must contain a non-empty prompt list."
        )
        grader_payload = json.loads(session_row["grader_response_json"])

        return Titled(
            f"Session {session_id}",
            P(
                A("overview", href="/"),
                " | ",
                A(
                    f"case {session_row['prompt_id']}",
                    href=f"/cases/{session_row['prompt_id']}",
                ),
                " | ",
                A(
                    f"filter model {session_row['model_id']}",
                    href=f"/?model_id={session_row['model_id']}",
                ),
            ),
            H2("Session"),
            detail_table(
                [
                    ("Session ID", str(session_row["session_id"])),
                    (
                        "Model",
                        A(
                            session_row["model_id"],
                            href=f"/?model_id={session_row['model_id']}",
                            cls="mono",
                        ),
                    ),
                    (
                        "Prompt",
                        A(
                            session_row["prompt_id"],
                            href=f"/cases/{session_row['prompt_id']}",
                            cls="mono",
                        ),
                    ),
                    ("Provider", session_row["provider"]),
                    ("Backend", session_row["backend"]),
                    ("Rollout Index", str(session_row["rollout_index"])),
                    (
                        "Result",
                        result_text(
                            criteria_met=session_row["criteria_met"],
                            score=session_row["score"],
                        ),
                    ),
                    ("Created", session_row["created_at"]),
                    ("Input Tokens", str(session_row["input_token_count"])),
                    (
                        "Cached Input Tokens",
                        str(session_row["cached_input_token_count"]),
                    ),
                    ("Output Tokens", str(session_row["output_token_count"])),
                    ("Question", Pre(session_row["question"])),
                ]
            ),
            H2("Rollout Messages"),
            transcript_table(
                [
                    (
                        1,
                        "system",
                        "runtime system prompt",
                        RUNTIME_SYSTEM_PROMPT,
                    ),
                    *[
                        (index + 2, message["role"], "", message["content"])
                        for index, message in enumerate(prompt_messages)
                    ],
                    (
                        len(prompt_messages) + 2,
                        "assistant",
                        "stored model output",
                        session_row["model_response"],
                    ),
                ]
            ),
            H2("Grader"),
            detail_table(
                [
                    (
                        "Explanation",
                        Pre(
                            session_row["grader_explanation"]
                            or "(no explanation stored)"
                        ),
                    ),
                    (
                        "Raw Grader JSON",
                        Pre(json.dumps(grader_payload, ensure_ascii=False, indent=2)),
                    ),
                ]
            ),
        )
    finally:
        db.conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local FastHTML browser for benchmark SQLite results."
    )
    parser.add_argument(
        "--results-db",
        type=Path,
        default=DEFAULT_RESULTS_DB_PATH,
        help="SQLite file to browse.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for the local web app.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Bind port for the local web app.",
    )
    args = parser.parse_args()

    resolved_db_path = args.results_db.expanduser().resolve()
    os.environ[RESULTS_DB_ENV_KEY] = str(resolved_db_path)
    configure_results_database(resolved_db_path)
    serve(host=args.host, port=args.port, reload=False)


if configured_results_db_path := os.environ.get(RESULTS_DB_ENV_KEY):
    configure_results_database(Path(configured_results_db_path))


if __name__ == "__main__":
    main()
