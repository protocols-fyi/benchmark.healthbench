# HealthBench Benchmark Runner

This repo holds the inference-only benchmark runner, the Azure OpenAI grader,
the constant-backed benchmark config, and the SQLite schema used to persist
benchmark results.

## Usage

From this directory:

```bash
uv run --env-file .env main.py --help
uv run --env-file .env main.py
uv run --env-file .env main.py --model gpt-4.1 --case-count 1 --rollout-count 2 --results-db ./results.sqlite3
```

For the grader:

```bash
uv run --env-file .env python grader.py
```
