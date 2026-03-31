# Project Instructions

- Keep the project intentionally minimal.
- Prefer explicit failures and short, actionable assertions over silent fallbacks.
- Use `uv run --env-file .env -m main --help` when checking CLI usage.
- Keep terminal logging enabled and inspect `run/*.log` for progress.
- Read the top-level docstrings in `main.py` and `grader.py` before changing behavior.
- This repo is an inference-only benchmark runner. Keep changes aligned with the current shape of the repo: benchmark CLI, model helpers, grader, SQLite persistence, and docs.
- Avoid thin wrapper functions. Do not add one-line pass-through helpers that are only called once.
- Avoid nested function definitions unless a closure is genuinely required. Prefer file-level helpers so logic is easier to test, inspect, and reuse.
- Avoid inline imports inside functions. Prefer module-level imports, and if import order matters, enforce that order at module scope.
- Prefer direct async entrypoints: `if __name__ == "__main__": asyncio.run(main())`.
- Only introduce wrappers when they are reused or provide real behavior (validation, adaptation, or error handling).
- In script-style modules and CLIs, prefer a small number of substantial functions over many tiny helpers.
- Inline single-use helpers when they only build a simple payload, adapt arguments once, or wrap a short stdlib expression. Naming alone is not enough reason to keep a helper.
- When a refactor removes the original need for an abstraction, delete the helper or data class instead of keeping dead scaffolding around it.
- Keep a helper, even if short, when it captures domain logic, a non-obvious invariant, or a contract that is worth testing or reusing independently.
- Use a single exception boundary per failure mode. Avoid stacked `try/except` blocks that translate or re-catch the same condition in multiple layers.
- Do not use `try: import ... except ...` as a capability check. Use explicit assertions, and `importlib.util.find_spec` when needed, then import directly.
- Backend requirements must be declared in config or dependencies and enforced with assertions. Do not use exception-driven fallback selection.
- Preserve function intent and scope. Example: `apply_*_defaults` functions should only apply defaults; do not add backend probing, install guidance, or runtime smoke tests there.
- Keep bootstrap and runtime entry code boring and linear. Move non-essential validation and diagnostics to dedicated scripts or commands.
- Minimize diff surface: implement only what the user asked for; avoid opportunistic hardening or side quests unless explicitly requested.
- Avoid embedding environment-specific install URLs or compatibility matrices in runtime code. Put those in docs or dependency metadata.
- When a change increases complexity, prefer a small note in docs or README over additional runtime branches.
- Logs must tell a full story. Log enough context to reconstruct the program flow. For benchmark runs, prefer one structured log record per rollout containing the prompt, model answer, grader conclusion, focused metrics, token usage, and error details when present.
- Before adding new Python code, inspect the existing symbols with `uvx suoyin *.py` and prefer reusing or extending what is already in the repo instead of creating parallel helpers or modules. If you still add a new top-level symbol, explain briefly why the existing code was not the right place.

## Lint And `noqa` Policy

- Prefer Ruff configuration in `pyproject.toml` over inline `# noqa` comments.
- For known file-level import-order exceptions, use `[tool.ruff.lint.per-file-ignores]` instead of annotating each import line.
- Treat inline `# noqa` as a last resort only when the suppression is truly line-specific and cannot be expressed in config.
- Any inline suppression must include the specific rule code and a short reason.
- For side-effect imports, prefer explicit usage instead of `# noqa: F401` where practical.
- After lint-related edits, run `uv run --env-file .env -m ruff check`.

## Benchmark Outputs

- Keep benchmark output formats stable and machine-readable.
- Runtime logs belong under `run/`.
- Persistent benchmark data belongs in the SQLite database passed via `--results-db`.
- If you change persisted benchmark fields, update `db.sql` and any corresponding docs in the same change.
