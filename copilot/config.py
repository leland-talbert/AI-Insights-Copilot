"""Central configuration. Everything is overridable through environment variables (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- Storage ---------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://copilot:copilot@localhost:5432/copilot")
DEFAULT_WORKSPACE = os.getenv("DEFAULT_WORKSPACE", "demo")
MAX_UPLOAD_MB = _env_float("MAX_UPLOAD_MB", 15)

# --- Chunking --------------------------------------------------------------
CHUNK_MAX_CHARS = _env_int("CHUNK_MAX_CHARS", 900)
CHUNK_OVERLAP_CHARS = _env_int("CHUNK_OVERLAP_CHARS", 150)

# --- Embeddings & retrieval ------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536  # must match db/schema.sql
TOP_K = _env_int("TOP_K", 6)
# Passages below this cosine similarity are never shown to the model.
# If nothing clears it, the app answers "insufficient evidence" without an LLM call.
MIN_SIMILARITY = _env_float("MIN_SIMILARITY", 0.25)


# --- Generation ------------------------------------------------------------
@dataclass(frozen=True)
class ModelOption:
    key: str
    provider: str  # "openai" | "ollama"
    model_id: str
    label: str


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

MODELS: dict[str, ModelOption] = {
    "gpt-4.1-mini": ModelOption("gpt-4.1-mini", "openai", "gpt-4.1-mini", "GPT-4.1-mini (lower cost)"),
    "gpt-4.1": ModelOption("gpt-4.1", "openai", "gpt-4.1", "GPT-4.1 (higher capability)"),
    "ollama": ModelOption("ollama", "ollama", OLLAMA_MODEL, f"Local model via Ollama ({OLLAMA_MODEL})"),
}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")

# USD per 1M tokens (input, output). Pricing assumptions for estimates only —
# confirm against the provider's current price list before quoting numbers.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4.1": (_env_float("PRICE_GPT41_IN", 2.00), _env_float("PRICE_GPT41_OUT", 8.00)),
    "gpt-4.1-mini": (_env_float("PRICE_GPT41MINI_IN", 0.40), _env_float("PRICE_GPT41MINI_OUT", 1.60)),
    "text-embedding-3-small": (_env_float("PRICE_EMBED_SMALL", 0.02), 0.0),
}

# --- Observability ---------------------------------------------------------
PHOENIX_ENABLED = _env_bool("PHOENIX_ENABLED", True)
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "ai-insights-copilot")

# --- Evaluation ------------------------------------------------------------
EVAL_WORKSPACE = os.getenv("EVAL_WORKSPACE", "eval")
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "evals" / "questions.json"
EVAL_RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_docs"
