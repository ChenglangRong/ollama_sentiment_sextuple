from __future__ import annotations

import os
import time
from typing import Any

from .heuristics import heuristic_extract
from .ollama_client import OllamaClient
from .prompts import prompt_one_shot, prompt_step1, prompt_step2, prompt_step3, prompt_step4
from .schemas import (
    normalize_quadruples,
    normalize_sextuples,
    normalize_target_aspects,
    validate_pipeline_result,
)


STEP_OPTIONS = {
    "step1": {"num_predict": int(os.getenv("OLLAMA_STEP1_NUM_PREDICT", "256"))},
    "step2": {"num_predict": int(os.getenv("OLLAMA_STEP2_NUM_PREDICT", "384"))},
    "step3": {"num_predict": int(os.getenv("OLLAMA_STEP3_NUM_PREDICT", "768"))},
    "step4": {"num_predict": int(os.getenv("OLLAMA_STEP4_NUM_PREDICT", "384"))},
}


class SextuplePipeline:
    def __init__(self, client: OllamaClient | None = None) -> None:
        self.client = client or OllamaClient()

    def run(self, dialogue: str, model: str, doc_id: str = "demo", mode: str = "four-step") -> dict[str, Any]:
        dialogue = (dialogue or "").strip()
        if not dialogue:
            return validate_pipeline_result(
                {
                    "doc_id": doc_id,
                    "model": model,
                    "target_aspects": [],
                    "quadruples": [],
                    "sextuples": [],
                    "sentiment_flips": [],
                    "valid_json": True,
                    "json_errors": [],
                    "latency_seconds": 0.0,
                    "mode": mode,
                }
            )

        if model == "heuristic-demo":
            result = heuristic_extract(dialogue, doc_id)
            result["model"] = model
            return validate_pipeline_result(result)

        if mode == "one-shot":
            return self._run_one_shot(dialogue, model, doc_id)
        return self._run_four_step(dialogue, model, doc_id)

    def _run_one_shot(self, dialogue: str, model: str, doc_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        parsed, response = self.client.generate_json(model, prompt_one_shot(dialogue, doc_id))
        result = {
            "doc_id": doc_id,
            "model": model,
            "target_aspects": [],
            "quadruples": [],
            "sextuples": normalize_sextuples(parsed.get("sextuples", []), doc_id),
            "sentiment_flips": parsed.get("sentiment_flips", []),
            "valid_json": response.valid_json,
            "json_errors": [response.json_error] if response.json_error else [],
            "latency_seconds": round(time.perf_counter() - started, 4),
            "mode": "one-shot",
            "raw_steps": {"one_shot": response.text},
        }
        return validate_pipeline_result(result)

    def _run_four_step(self, dialogue: str, model: str, doc_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        errors: list[str] = []
        raw_steps: dict[str, str] = {}

        parsed1, response1 = self.client.generate_json(model, prompt_step1(dialogue, doc_id), options=STEP_OPTIONS["step1"])
        target_aspects = normalize_target_aspects(parsed1)
        raw_steps["step1"] = response1.text
        if response1.json_error:
            errors.append(f"step1: {response1.json_error}")

        parsed2, response2 = self.client.generate_json(model, prompt_step2(dialogue, doc_id, target_aspects), options=STEP_OPTIONS["step2"])
        quadruples = normalize_quadruples(parsed2)
        raw_steps["step2"] = response2.text
        if response2.json_error:
            errors.append(f"step2: {response2.json_error}")

        parsed3, response3 = self.client.generate_json(model, prompt_step3(dialogue, doc_id, quadruples), options=STEP_OPTIONS["step3"])
        sextuples = normalize_sextuples(parsed3.get("sextuples", []), doc_id)
        raw_steps["step3"] = response3.text
        if response3.json_error:
            errors.append(f"step3: {response3.json_error}")

        parsed4, response4 = self.client.generate_json(model, prompt_step4(dialogue, doc_id, sextuples), options=STEP_OPTIONS["step4"])
        flips = parsed4.get("sentiment_flips", [])
        raw_steps["step4"] = response4.text
        if response4.json_error:
            errors.append(f"step4: {response4.json_error}")

        result = {
            "doc_id": doc_id,
            "model": model,
            "target_aspects": target_aspects,
            "quadruples": quadruples,
            "sextuples": sextuples,
            "sentiment_flips": flips if isinstance(flips, list) else [],
            "valid_json": all([response1.valid_json, response2.valid_json, response3.valid_json, response4.valid_json]),
            "json_errors": errors,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "mode": "four-step",
            "raw_steps": raw_steps,
        }
        return validate_pipeline_result(result)
