"""Hard-coded benchmark constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_BIN_DIR = PROJECT_ROOT / "bin"

DEFAULT_MODEL_ASK_QUESTION = "what's the main argument of Sutton's Bitter Lessons?"
DEFAULT_MODEL_ASK_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL_ASK_MAX_TOKENS = 512
DEFAULT_MODEL_ASK_TEMPERATURE = 0.0
DEFAULT_MODEL_ASK_TIMEOUT_SECONDS = 120.0

PROMPT_SAMPLE_SEED = 0
REQUIRED_CASE_TAG = "physician_agreed_category:not-enough-context"
FOCUS_METRIC_NAME = "axis:context_awareness"

VLLM_STANDBY_ENV_KEY = "UNSLOTH_VLLM_STANDBY"
DEFAULT_UNSLOTH_VLLM_STANDBY = "1"
DEFAULT_VLLM_HOST = "127.0.0.1"
DEFAULT_VLLM_PORT = 8000
DEFAULT_VLLM_BASE_URL = f"http://{DEFAULT_VLLM_HOST}:{DEFAULT_VLLM_PORT}/v1"
VLLM_BASE_URL = DEFAULT_VLLM_BASE_URL
DEFAULT_VLLM_STARTUP_TIMEOUT_SECONDS = 600.0
DEFAULT_QWEN_ASK_LOG_PATH = PROJECT_ROOT / "run" / "qwen.ask.vllm.log"

MODEL_REFERENCE_ALIASES = {
    "qwen3-8b": "Qwen/Qwen3-8B",
}
LOCAL_MODEL_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
)

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ROLLOUT_COUNT = 16
DEFAULT_GENERATION_TIMEOUT_SECONDS = 480

DEFAULT_VLLM_MAX_SEQ_LENGTH = 6144
DEFAULT_VLLM_COMPLETION_TOKEN_LIMIT = 1536
DEFAULT_VLLM_TOP_P = 0.95 
DEFAULT_VLLM_TEMPERATURE = 0.6
DEFAULT_VLLM_PRESENCE_PENALTY = 0.2
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.5
DEFAULT_VLLM_KV_CACHE_DTYPE = "fp8"

AWS_BEDROCK_SUPPORTED_MODEL_IDS = {
    "anthropic-haiku-4.5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic-sonnet-4.6": "global.anthropic.claude-sonnet-4-6",
    "anthropic-opus-4.6": "global.anthropic.claude-opus-4-6-v1",
}
NOISY_LOGGERS = (
    "anthropic",
    "anthropic._base_client",
    "botocore",
    "botocore.auth",
    "botocore.credentials",
    "botocore.hooks",
    "botocore.session",
    "botocore.utils",
)
DEFAULT_BEDROCK_MODEL = "anthropic-haiku-4.5"

DEFAULT_AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
DEFAULT_AZURE_OPENAI_ENDPOINT = "https://gpt41-endpoint.openai.azure.com/"
DEFAULT_AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
DEFAULT_AZURE_OPENAI_MAX_RETRIES = 3
DEFAULT_AZURE_OPENAI_TIMEOUT_SECONDS = 20 * 60
