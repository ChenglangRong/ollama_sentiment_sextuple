from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> tuple[dict[str, Any], bool, str]:
    """Extract the first JSON object from a model response."""
    if not text:
        return {}, False, "empty response"

    cleaned = strip_fences(text.strip())
    candidates = [cleaned]
    object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, True, ""
        except json.JSONDecodeError:
            pass

    last_error = ""
    for candidate in candidates:
        repaired = repair_common_json_issues(candidate)
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed, True, "repaired common JSON issues"
        except json.JSONDecodeError as exc:
            last_error = str(exc)

    for candidate in candidates:
        recovered = recover_partial_object(candidate)
        if recovered:
            return recovered, True, f"recovered partial JSON after parse error: {last_error}"

    if last_error:
        return {}, False, last_error

    return {}, False, "parsed JSON root is not an object"


def strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def recover_partial_object(text: str) -> dict[str, Any]:
    """Recover completed top-level arrays from a truncated JSON object."""
    result: dict[str, Any] = {}
    doc_id_match = re.search(r'"doc_id"\s*:\s*"([^"]+)"', text)
    if doc_id_match:
        result["doc_id"] = doc_id_match.group(1)

    for key in ("target_aspects", "quadruples", "sextuples", "sentiment_flips"):
        array_text = extract_balanced_array(text, key)
        if array_text is None:
            continue
        try:
            result[key] = json.loads(array_text)
        except json.JSONDecodeError:
            continue

    if any(key in result for key in ("target_aspects", "quadruples", "sextuples")):
        result.setdefault("sentiment_flips", [])
        return result
    if "sentiment_flips" in result:
        return result
    return {}


def extract_balanced_array(text: str, key: str) -> str | None:
    key_match = re.search(rf'"{re.escape(key)}"\s*:', text)
    if not key_match:
        return None
    start = text.find("[", key_match.end())
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def dumps_pretty(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
