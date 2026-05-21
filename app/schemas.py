from __future__ import annotations

from typing import Any

from .config import OUTPUT_SCHEMA_FIELDS


VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed", "unknown"}


def normalize_sentiment(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "pos": "positive",
        "正面": "positive",
        "积极": "positive",
        "positive": "positive",
        "neg": "negative",
        "负面": "negative",
        "消极": "negative",
        "negative": "negative",
        "neu": "neutral",
        "中性": "neutral",
        "neutral": "neutral",
        "mixed": "mixed",
        "混合": "mixed",
    }
    return mapping.get(text, "unknown")


def normalize_confidence(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, score))


def normalize_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_sextuple(row: dict[str, Any], doc_id: str = "") -> dict[str, Any]:
    normalized = {field: "" for field in OUTPUT_SCHEMA_FIELDS}
    normalized.update(
        {
            "doc_id": normalize_text(row.get("doc_id") or doc_id),
            "speaker": normalize_text(row.get("speaker") or row.get("holder")),
            "target": normalize_text(row.get("target") or row.get("aspect_term")),
            "aspect": normalize_text(row.get("aspect") or row.get("attribute")),
            "opinion": normalize_text(row.get("opinion") or row.get("opinion_term")),
            "sentiment": normalize_sentiment(row.get("sentiment")),
            "rationale": normalize_text(row.get("rationale") or row.get("evidence")),
            "confidence": normalize_confidence(row.get("confidence", 0.5)),
            "sentiment_flip": normalize_text(row.get("sentiment_flip") or "None"),
        }
    )
    return normalized


def normalize_sextuples(rows: Any, doc_id: str = "") -> list[dict[str, Any]]:
    if isinstance(rows, dict):
        rows = rows.get("sextuples") or rows.get("sextuple") or rows.get("items") or []
    if not isinstance(rows, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            normalized.append(normalize_sextuple(item, doc_id))
        elif isinstance(item, (list, tuple)):
            padded = list(item) + [""] * 8
            normalized.append(
                normalize_sextuple(
                    {
                        "speaker": padded[0],
                        "target": padded[1],
                        "aspect": padded[2],
                        "opinion": padded[3],
                        "sentiment": padded[4],
                        "rationale": padded[5],
                        "confidence": padded[6] if len(padded) > 6 else 0.5,
                        "sentiment_flip": padded[7] if len(padded) > 7 else "None",
                    },
                    doc_id,
                )
            )
    return normalized


def normalize_target_aspects(payload: Any) -> list[dict[str, str]]:
    rows = payload.get("target_aspects", payload.get("two-tuples", [])) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    result = []
    for item in rows:
        if isinstance(item, dict):
            target = normalize_text(item.get("target"))
            aspect = normalize_text(item.get("aspect"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            target, aspect = normalize_text(item[0]), normalize_text(item[1])
        else:
            continue
        if target or aspect:
            result.append({"target": target, "aspect": aspect})
    return result


def normalize_quadruples(payload: Any) -> list[dict[str, str]]:
    rows = payload.get("quadruples", payload.get("items", [])) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return []
    result = []
    for item in rows:
        if isinstance(item, dict):
            row = {
                "speaker": normalize_text(item.get("speaker") or item.get("holder")),
                "target": normalize_text(item.get("target")),
                "aspect": normalize_text(item.get("aspect")),
                "opinion": normalize_text(item.get("opinion")),
            }
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            row = {
                "speaker": normalize_text(item[0]),
                "target": normalize_text(item[1]),
                "aspect": normalize_text(item[2]),
                "opinion": normalize_text(item[3]),
            }
        else:
            continue
        if row["target"] or row["aspect"] or row["opinion"]:
            result.append(row)
    return result


def validate_pipeline_result(result: dict[str, Any]) -> dict[str, Any]:
    sextuples = normalize_sextuples(result.get("sextuples", []), str(result.get("doc_id", "")))
    flips = result.get("sentiment_flips", [])
    if not isinstance(flips, list):
        flips = []
    result["sextuples"] = sextuples
    result["sentiment_flips"] = flips
    result["valid_json"] = bool(result.get("valid_json", True))
    return result

