"""Small utility helpers for the benchmark repo."""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

RUN_LOG_PATH_ENV_KEY = "HEALTHBENCH_RUN_LOG_PATH"


def _resolve_run_log_path(
    *,
    run_dir: Path,
    configured_log_path: str = "",
    current_time: datetime | None = None,
) -> Path:
    normalized_configured_path = configured_log_path.strip()
    if normalized_configured_path:
        return Path(normalized_configured_path).expanduser().resolve()
    if current_time is None:
        current_time = datetime.now()
    return run_dir / f"{current_time.strftime('%y%m%d-%H%M%S')}.log"


def init_logging(level: int = logging.DEBUG, run_dir: Path | None = None) -> Path:
    log_format = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | "
        "%(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    if run_dir is None:
        run_dir = Path(__file__).resolve().parent / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = _resolve_run_log_path(
        run_dir=run_dir,
        configured_log_path=os.environ.get(RUN_LOG_PATH_ENV_KEY, ""),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(log_format, date_format)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logging.captureWarnings(True)
    for noisy_logger in (
        "httpcore",
        "httpcore.http11",
        "urllib3.connectionpool",
        "filelock",
        "httpx",
        "openai._base_client",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    return log_path
