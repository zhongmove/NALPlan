"""Environment-based configuration for the public NALPlan release."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = PACKAGE_ROOT.parent
DATA_ROOT = RELEASE_ROOT / "data"
BENCHMARK_ROOT = RELEASE_ROOT / "benchmark"
CACHE_ROOT = RELEASE_ROOT / "cache"

LLM_BASE_URL = os.getenv("NALPLAN_LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("NALPLAN_LLM_API_KEY", "")
LLM_MODEL = os.getenv("NALPLAN_LLM_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("NALPLAN_EMBEDDING_MODEL", "text-embedding-3-large")
SECONDO_HOME = Path(os.getenv("SECONDO_HOME", str(Path.home() / "secondo"))).expanduser()
SECONDO_BIN = Path(os.getenv("SECONDO_BIN", str(SECONDO_HOME / "bin" / "SecondoBDB")))
ALGEBRAS_PATH = Path(os.getenv("SECONDO_ALGEBRAS", str(SECONDO_HOME / "Algebras")))
DATABASE_ROOT = Path(os.getenv("SECONDO_DATABASE_ROOT", str(Path.home() / "secondo-databases"))).expanduser()

NUM_PATHS = int(os.getenv("NALPLAN_NUM_PATHS", "7"))
CANDIDATES_PER_PATH = int(os.getenv("NALPLAN_CANDIDATES_PER_PATH", "3"))
OPERATOR_RETRIEVAL_DEPTH = int(os.getenv("NALPLAN_OPERATOR_DEPTH", "30"))
TEMPERATURE = float(os.getenv("NALPLAN_TEMPERATURE", "0.85"))
QUERY_TIMEOUT = int(os.getenv("NALPLAN_QUERY_TIMEOUT", "10"))


def dataset_path(dataset: str) -> Path:
    """Return the public data directory for a dataset identifier."""
    return DATA_ROOT / dataset
