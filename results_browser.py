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

from fasthtml.common import A, Card, Code, Div, H2, H3, P, Pre, Small, Strong, Style, Titled, fast_app, serve
from sqlite_utils import Database

from db import open_benchmark_db

RUNTIME_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_RESULTS_DB_PATH = Path("./results.sqlite3")
RESULTS_DB_ENV_KEY = "HEALTHBENCH_RESULTS_DB"
RESULTS_DB_PATH: Path | None = None
APP_CSS = """
:root {
    --hb-muted: #52606d;
    --hb-border: #d9e2ec;
    --hb-surface: #f8fafc;
    --hb-good: #127a4a;
    --hb-bad: #a61b1b;
}

body {
    background: linear-gradient(180deg, #f7fbff 0%, #ffffff 55%);
}

main.container {
    max-width: 1280px;
}

.summary-grid,
.message-list,
.detail-grid,
.filters {
    display: grid;
    gap: 1rem;
}

.summary-grid {
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
}

.detail-grid {
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}

.filters {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    margin-bottom: 1.25rem;
}

.meta {
    color: var(--hb-muted);
}

.muted {
    color: var(--hb-muted);
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.chip {
    border: 1px solid var(--hb-border);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    background: white;
    color: var(--hb-muted);
    font-size: 0.9rem;
    text-decoration: none;
}

.message-card {
    border: 1px solid var(--hb-border);
    border-left-width: 6px;
    border-radius: 16px;
    padding: 1rem;
    background: white;
}

.message-card.system {
    border-left-color: #486581;
}

.message-card.user {
    border-left-color: #20639b;
}

.message-card.assistant {
    border-left-color: #127a4a;
}

.message-card.grader {
    border-left-color: #8d6e00;
}

.message-card pre,
pre {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    background: var(--hb-surface);
    border: 1px solid var(--hb-border);
    border-radius: 12px;
    padding: 0.9rem;
}

.score-good {
    color: var(--hb-good);
}

.score-bad {
    color: var(--hb-bad);
}

table td,
table th {
    vertical-align: top;
}
""".strip()

app, rt = fast_app(
    title="HealthBench Results Browser",
    hdrs=(Style(APP_CSS),),
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


def build_message_card(*, role: str, content: str, note: str | None = None) -> Card:
    assert role in {"system", "user", "assistant", "grader"}, f"Unsupported role: {role}"
    assert content.strip(), f"{role} content must be non-empty."
    header = Div(
        Strong(role),
        Div(Small(note, cls="muted")) if note else "",
        cls="chip-row",
    )
    return Card(
        header,
        Pre(content),
        cls=f"message-card {role}",
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
        filters = []
        if model_id:
            filters.append(
                Card(
                    H3("Model Filter"),
                    P(Code(model_id)),
                    P(A("Clear model filter", href=f"/?prompt_id={prompt_id}")),
                )
            )
        if prompt_id:
            filters.append(
                Card(
                    H3("Prompt Filter"),
                    P(Code(prompt_id)),
                    P(A("Open case detail", href=f"/cases/{prompt_id}")),
                    P(A("Clear prompt filter", href=f"/?model_id={model_id}")),
                )
            )

        return Titled(
            "HealthBench Results Browser",
            P(
                "Read-only viewer for ",
                Code(str(RESULTS_DB_PATH)),
                ". Click a case or rollout to inspect the full prompt and outputs.",
                cls="meta",
            ),
            Div(
                Card(H3("Models"), P(str(summary["model_count"]))),
                Card(H3("Cases"), P(str(summary["case_count"]))),
                Card(H3("Sessions"), P(str(summary["session_count"]))),
                cls="summary-grid",
            ),
            Div(*filters, cls="filters") if filters else "",
            H2("Models"),
            Div(
                *[
                    Card(
                        H3(A(row["model_id"], href=f"/?model_id={row['model_id']}")),
                        P(f"provider={row['provider']} | backend={row['backend']}", cls="meta"),
                        P(
                            f"sessions={row['session_count']} | "
                            f"mean_score={row['mean_score']:.3f}" if row["mean_score"] is not None else f"sessions={row['session_count']}",
                        ),
                    )
                    for row in model_rows
                ]
                if model_rows
                else [P("No model rows found.")]
            ),
            H2("Cases"),
            Div(
                *[
                    Card(
                        H3(A(row["prompt_id"], href=f"/cases/{row['prompt_id']}")),
                        P(row["question"][:280] + ("..." if len(row["question"]) > 280 else "")),
                        P(
                            f"sessions={row['session_count']} | "
                            f"models={row['model_count']}",
                            cls="meta",
                        ),
                        P(A("Filter rollouts to this case", href=f"/?prompt_id={row['prompt_id']}")),
                    )
                    for row in case_rows
                ]
                if case_rows
                else [P("No case rows found.")]
            ),
            H2("Recent Rollouts"),
            Div(
                *[
                    Card(
                        H3(A(f"session {row['session_id']}", href=f"/sessions/{row['session_id']}")),
                        P(
                            A(row["model_id"], href=f"/?model_id={row['model_id']}"),
                            " | ",
                            A(row["prompt_id"], href=f"/cases/{row['prompt_id']}"),
                        ),
                        P(
                            f"rollout_index={row['rollout_index']} | "
                            f"score={row['score']:.1f} | "
                            f"criteria_met={bool(row['criteria_met'])}",
                            cls="score-good" if row["criteria_met"] else "score-bad",
                        ),
                        P(row["question"][:240] + ("..." if len(row["question"]) > 240 else ""), cls="meta"),
                    )
                    for row in rollout_rows
                ]
                if rollout_rows
                else [P("No sessions matched the active filters.")]
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
            P(A("Back to overview", href="/")),
            P(
                f"Stored sessions: {case_row['session_count']}",
                cls="meta",
            ),
            Div(
                Card(
                    H3("Rendered Question"),
                    Pre(case_row["question"]),
                ),
                Card(
                    H3("Prompt Messages"),
                    Div(
                        *[
                            build_message_card(
                                role=message["role"],
                                content=message["content"],
                            )
                            for message in prompt_messages
                        ],
                        cls="message-list",
                    ),
                ),
                cls="detail-grid",
            ),
            H2("Rollouts"),
            Div(
                *[
                    Card(
                        H3(A(f"session {row['session_id']}", href=f"/sessions/{row['session_id']}")),
                        P(row["model_id"]),
                        P(
                            f"rollout_index={row['rollout_index']} | "
                            f"score={row['score']:.1f} | "
                            f"criteria_met={bool(row['criteria_met'])}",
                            cls="score-good" if row["criteria_met"] else "score-bad",
                        ),
                        P(row["created_at"], cls="meta"),
                    )
                    for row in session_rows
                ]
                if session_rows
                else [P("No rollout rows stored for this case yet.")]
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
                A("Back to overview", href="/"),
                " | ",
                A(
                    f"Open case {session_row['prompt_id']}",
                    href=f"/cases/{session_row['prompt_id']}",
                ),
            ),
            Div(
                Card(
                    H3("Rollout Metadata"),
                    P(
                        Strong("model_id: "),
                        A(
                            session_row["model_id"],
                            href=f"/?model_id={session_row['model_id']}",
                        ),
                    ),
                    P(
                        Strong("prompt_id: "),
                        A(
                            session_row["prompt_id"],
                            href=f"/cases/{session_row['prompt_id']}",
                        ),
                    ),
                    P(
                        f"provider={session_row['provider']} | "
                        f"backend={session_row['backend']} | "
                        f"rollout_index={session_row['rollout_index']}"
                    ),
                    P(
                        f"score={session_row['score']:.1f} | "
                        f"criteria_met={bool(session_row['criteria_met'])}",
                        cls="score-good" if session_row["criteria_met"] else "score-bad",
                    ),
                    P(
                        f"input_tokens={session_row['input_token_count']} | "
                        f"cached_input_tokens={session_row['cached_input_token_count']} | "
                        f"output_tokens={session_row['output_token_count']}"
                    ),
                    P(session_row["created_at"], cls="meta"),
                ),
                Card(
                    H3("Question"),
                    Pre(session_row["question"]),
                ),
                cls="detail-grid",
            ),
            H2("Rollout Messages"),
            Div(
                build_message_card(
                    role="system",
                    content=RUNTIME_SYSTEM_PROMPT,
                    note="Added by executor at runtime before each model call.",
                ),
                *[
                    build_message_card(
                        role=message["role"],
                        content=message["content"],
                    )
                    for message in prompt_messages
                ],
                build_message_card(
                    role="assistant",
                    content=session_row["model_response"],
                    note="Persisted assistant output from this rollout.",
                ),
                cls="message-list",
            ),
            H2("Grader Result"),
            Div(
                build_message_card(
                    role="grader",
                    content=session_row["grader_explanation"] or "(no explanation stored)",
                    note="Parsed explanation stored in the sessions table.",
                ),
                Card(
                    H3("Raw Grader JSON"),
                    Pre(json.dumps(grader_payload, ensure_ascii=False, indent=2)),
                ),
                cls="detail-grid",
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
