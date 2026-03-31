# Project Instructions

- Keep the project intentionally minimal.
- Prefer explicit failures and short, actionable assertions over silent fallbacks.
- Use `uv run --env-file .env -m main --help` when checking CLI usage.
- Keep terminal logging enabled and inspect `run/*.log` for progress.
- Launch long training runs with `make train`, not raw `uv run --env-file .env -m main`. `make train` pauses apt/snap/unattended maintenance during the run. This is necessary on this box because host OOM and reboot events overlapped with `apt-daily-upgrade` and `snapd`, so bypassing the wrapper removes a real stability guard.
- Read the top-level docstrings in `main.py` and `server.py` before changing behavior.
- Avoid thin wrapper functions. Do not add one-line pass-through helpers that are only called once.
- Avoid nested function definitions unless a closure is genuinely required. Prefer file-level helpers so logic is easier to test, inspect, and reuse.
- Avoid inline imports inside functions. Prefer module-level imports, and if import order matters, enforce that order at module scope.
- Prefer direct async entrypoints: `if __name__ == "__main__": asyncio.run(main())`.
- Only introduce wrappers when they are reused or provide real behavior (validation, adaptation, or error handling).
- In script-style modules and CLIs, prefer a small number of substantial functions over many tiny helpers.
- Inline single-use helpers when they only build a simple payload, adapt arguments once, or wrap a short stdlib expression. Naming alone is not enough reason to keep a helper.
- When a refactor removes the original need for an abstraction, delete the helper/data class instead of keeping dead scaffolding around it.
- Keep a helper, even if short, when it captures domain logic, a non-obvious invariant, or a contract that is worth testing or reusing independently.
- Use a single exception boundary per failure mode. Avoid stacked `try/except` blocks that translate/re-catch the same condition in multiple layers.
- Do not use `try: import ... except ...` as a capability check. Use explicit assertions (and `importlib.util.find_spec` when needed), then import directly.
- Backend requirements must be declared in config/dependencies and enforced with assertions; do not use exception-driven fallback selection.
- Preserve function intent and scope. Example: `apply_*_defaults` functions should only apply defaults; do not add backend probing, install guidance, or runtime smoke tests there.
- Keep bootstrap/runtime entry code boring and linear. Move non-essential validation and diagnostics to dedicated scripts/commands.
- Minimize diff surface: implement only what the user asked for; avoid opportunistic hardening or side quests unless explicitly requested.
- Avoid embedding environment-specific install URLs or compatibility matrices in runtime code. Put those in docs or dependency metadata.
- When a change increases complexity, prefer a small comment in docs/README over additional runtime branches.
- Logs must tell a full story. Log enough context to reconstruct the entire program flow. For benchmark-style scripts, prefer one structured log record per rollout containing the prompt, model answer, grader conclusion, focused metrics, token usage, and error details when present. Do not require duplicate trace artifacts unless the user explicitly wants them.
- Before adding new Python code, inspect the existing symbols with `uvx suoyin *.py` and prefer reusing or extending what is already in the repo instead of creating parallel helpers or modules; if you still add a new top-level symbol, explain briefly why the existing code was not the right place.

## Lint And `noqa` Policy

- Prefer Ruff configuration in `pyproject.toml` over inline `# noqa` comments.
- For known file-level import-order exceptions (for example, runtime env bootstrap before imports), use `[tool.ruff.lint.per-file-ignores]` (for example `E402`) instead of annotating each import line.
- Treat inline `# noqa` as a last resort only when the suppression is truly line-specific and cannot be expressed in config.
- Any inline suppression must include the specific rule code and a short reason.
- For side-effect imports, prefer explicit usage (for example `_ = unsloth`) instead of `# noqa: F401` where practical.
- After lint-related edits, run: `uv run --env-file .env -m ruff check`.

## Experiment Config Structure

- Store runnable experiment configs under `experiments/<experiment-name>/config.yaml`.
- `make train` should resolve the default config from the most recently modified `experiments/*/config.yaml`.
- When creating a new experiment, make the folder name descriptive and keep `training.model_name` aligned with that experiment.
- When an experiment writes `experiments/<name>/output.tsv`, keep it tab-separated with a stable header row and no extra prose.
- Existing per-step metric TSVs under `run/*.tsv` use this same schema, and new experiment outputs should match it.
- Current per-step metric TSVs use these columns. For each signal, e.g. discovery/successCriteria/failureCriteria
  - `step`: training step or checkpoint number.
  - `*_mean`: mean discovery score across rollouts for that scenario at that step.
  - `*_max`: best rollout score at that step.
  - `*_min`: worst rollout score at that step.
  - `*_std`: standard deviation of rollout scores at that step.
  - `*_entropy`: mean training entropy recorded for that step.
- For quick graphs, prefer `gnuplot` over ad hoc scripts. 
