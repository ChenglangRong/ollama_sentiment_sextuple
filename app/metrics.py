from __future__ import annotations

import re
from typing import Any

from .schemas import normalize_sentiment


def norm(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^\w\u4e00-\u9fa5]+|[^\w\u4e00-\u9fa5]+$", "", text)
    return text


def pred_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    target = norm(row.get("target"))
    opinion = norm(row.get("opinion"))
    sentiment = normalize_sentiment(row.get("sentiment"))
    return target, opinion, sentiment


def gold_triplet(row: dict[str, Any]) -> tuple[str, str, str]:
    target = norm(row.get("target"))
    opinion = norm(row.get("opinion"))
    sentiment = normalize_sentiment(row.get("sentiment"))
    return target, opinion, sentiment


def score_triplets(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, float]:
    pred_set = {pred_triplet(item) for item in predicted if item.get("target") or item.get("opinion")}
    gold_set = {gold_triplet(item) for item in gold if item.get("target") or item.get("opinion")}
    exact_true_positive = len(pred_set & gold_set)
    matched_gold: set[tuple[str, str, str]] = set()
    soft_true_positive = 0
    for pred in pred_set:
        for candidate in gold_set:
            if candidate in matched_gold:
                continue
            if soft_triplet_match(pred, candidate):
                matched_gold.add(candidate)
                soft_true_positive += 1
                break

    precision = soft_true_positive / len(pred_set) if pred_set else 0.0
    recall = soft_true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact_precision = exact_true_positive / len(pred_set) if pred_set else 0.0
    exact_recall = exact_true_positive / len(gold_set) if gold_set else 0.0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if exact_precision + exact_recall else 0.0

    sentiment_matches = 0
    sentiment_total = 0
    gold_by_pair = {(item[0], item[1]): item[2] for item in gold_set}
    for target, opinion, sentiment in pred_set:
        key = (target, opinion)
        if key in gold_by_pair:
            sentiment_total += 1
            sentiment_matches += int(gold_by_pair[key] == sentiment)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sentiment_accuracy": sentiment_matches / sentiment_total if sentiment_total else 0.0,
        "exact_precision": exact_precision,
        "exact_recall": exact_recall,
        "exact_f1": exact_f1,
        "pred_count": float(len(pred_set)),
        "gold_count": float(len(gold_set)),
        "tp": float(soft_true_positive),
        "exact_tp": float(exact_true_positive),
    }


def soft_triplet_match(pred: tuple[str, str, str], gold: tuple[str, str, str]) -> bool:
    pred_target, pred_opinion, pred_sentiment = pred
    gold_target, gold_opinion, gold_sentiment = gold
    if pred_sentiment != gold_sentiment:
        return False
    target_ok = span_overlap(pred_target, gold_target) >= 0.5 or pred_target in gold_target or gold_target in pred_target
    opinion_ok = span_overlap(pred_opinion, gold_opinion) >= 0.4 or pred_opinion in gold_opinion or gold_opinion in pred_opinion
    return bool(target_ok and opinion_ok)


def span_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens))


def score_flip(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, float]:
    def key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            norm(item.get("speaker")),
            norm(item.get("target")),
            norm(item.get("aspect")),
            normalize_sentiment(item.get("from_sentiment")),
            normalize_sentiment(item.get("to_sentiment")),
        )

    pred_set = {key(item) for item in predicted}
    gold_set = {key(item) for item in gold}
    true_positive = len(pred_set & gold_set)
    precision = true_positive / len(pred_set) if pred_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"flip_precision": precision, "flip_recall": recall, "flip_f1": f1}


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}
