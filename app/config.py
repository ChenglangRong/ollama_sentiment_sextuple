from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODELS = ["qwen2.5:7b", "llama3.1:8b", "qwen2.5:3b"]
DEMO_MODELS = ["qwen2.5:7b", "llama3.1:8b", "qwen2.5:3b", "heuristic-demo"]

DEFAULT_OPTIONS = {
    "temperature": 0.0,
    "top_p": 0.9,
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192")),
    "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "768")),
}

OUTPUT_SCHEMA_FIELDS = [
    "doc_id",
    "speaker",
    "target",
    "aspect",
    "opinion",
    "sentiment",
    "rationale",
    "confidence",
    "sentiment_flip",
]
