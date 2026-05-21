from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import DEFAULT_OPTIONS, OLLAMA_BASE_URL
from .json_utils import extract_json_object
from .prompts import SYSTEM_PROMPT


@dataclass
class ModelResponse:
    text: str
    latency_seconds: float
    raw: dict[str, Any]
    valid_json: bool
    json_error: str = ""


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 180) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException:
            return []
        return [item.get("name", "") for item in payload.get("models", []) if item.get("name")]

    def generate_json(
        self,
        model: str,
        prompt: str,
        system: str = SYSTEM_PROMPT,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], ModelResponse]:
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {**DEFAULT_OPTIONS, **(options or {})},
        }
        started = time.perf_counter()
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        latency = time.perf_counter() - started
        response.raise_for_status()
        raw = response.json()
        text = raw.get("response", "")
        parsed, valid, error = extract_json_object(text)
        return parsed, ModelResponse(text=text, latency_seconds=latency, raw=raw, valid_json=valid, json_error=error)

