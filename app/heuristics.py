from __future__ import annotations

import re
from typing import Any

from .schemas import normalize_sextuple


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "clean",
    "comfortable",
    "friendly",
    "fast",
    "accurate",
    "crucial",
    "high",
    "love",
    "like",
    "recommend",
    "满意",
    "优秀",
    "好",
    "干净",
    "准确",
}

NEGATIVE_WORDS = {
    "bad",
    "poor",
    "slow",
    "dirty",
    "expensive",
    "noisy",
    "hate",
    "disappointing",
    "terrible",
    "差",
    "慢",
    "脏",
    "贵",
    "失望",
}

KNOWN_TARGETS = [
    "budget hotels",
    "battery life",
    "phone",
    "screen",
    "restaurant",
    "service",
    "staff",
    "food",
    "price",
    "delivery",
    "package protection",
    "问答系统",
    "model output format",
    "rationale",
    "laptop keyboard",
    "fan noise",
    "hotel room",
    "wi-fi",
    "extraction pipeline",
    "empty inputs",
    "normal comments",
    "keyboard",
    "room",
    "camera quality",
    "interface",
    "app",
    "microphone",
    "soup",
    "location",
    "portion",
    "charger",
    "design",
    "table",
    "support team",
    "installation process",
    "noodles",
    "search feature",
    "speaker",
    "seat",
    "translation",
    "login page",
    "dessert",
    "documentation",
    "checkout process",
    "game controls",
    "cloud sync",
    "soup portion",
    "nlp system",
    "laptop",
    "model comparison",
    "one-shot prompt",
    "public dataset",
    "gradio",
    "report",
    "api",
    "student project",
    "camera",
    "dataset",
    "fallback data",
    "hotel",
    "defense questions",
    "rule baseline",
    "llm pipeline",
    "headphones",
    "references",
    "extraction result",
]

OPINION_SENTIMENT = {
    "crucial": "positive",
    "high cleanliness standards": "positive",
    "excellent": "positive",
    "disappointing": "negative",
    "friendly and fast": "positive",
    "expensive": "negative",
    "slow": "negative",
    "good": "positive",
    "准确率很高": "positive",
    "响应速度有点慢": "negative",
    "stable": "positive",
    "clear": "positive",
    "comfortable": "positive",
    "terrible": "negative",
    "clean": "positive",
    "unstable": "negative",
    "accurate": "positive",
    "mixed": "mixed",
    "empty result": "neutral",
    "structured sextuples": "positive",
    "rude": "negative",
    "bright": "positive",
    "uncomfortable": "negative",
    "noisy": "negative",
    "poor": "negative",
    "simple": "positive",
    "useful": "positive",
    "faster": "positive",
    "cold": "negative",
    "bland": "negative",
    "convenient": "positive",
    "reasonable": "positive",
    "large": "positive",
    "bulky": "negative",
    "elegant": "positive",
    "dirty": "negative",
    "helpful": "positive",
    "confusing": "negative",
    "fresh": "positive",
    "tasty": "positive",
    "irrelevant": "negative",
    "loud enough": "positive",
    "too hard": "negative",
    "crashes often": "negative",
    "beautiful": "positive",
    "delicious": "positive",
    "lacks examples": "negative",
    "quick": "positive",
    "awkward": "negative",
    "reliable": "positive",
    "small": "negative",
    "poor": "negative",
    "acceptable": "positive",
    "weak": "negative",
    "hard to use": "negative",
    "convenient": "positive",
    "attractive": "positive",
    "risky": "negative",
    "high": "negative",
    "unreliable": "negative",
    "unnecessary": "negative",
    "valuable": "positive",
    "enough": "positive",
    "safer": "positive",
    "difficult": "negative",
    "worth using": "positive",
    "unsafe": "negative",
    "scary": "negative",
    "manageable": "positive",
    "limited": "negative",
    "weak": "negative",
    "suitable": "positive",
    "ordinary": "neutral",
    "impressive": "positive",
    "expanded": "negative",
    "less suitable": "negative",
}

ASPECT_HINTS = {
    "cleanliness": "cleanliness",
    "battery": "battery",
    "interface": "interface",
    "fan noise": "fan noise",
    "price": "price",
    "stability": "stability",
    "rubric": "course value",
    "noise": "noise",
    "json": "JSON validity",
    "score": "score value",
    "safety": "safety",
    "defense": "defense value",
    "literature": "literature support",
    "typing": "typing comfort",
    "warranty": "value",
    "subway": "transportation",
    "low-light": "low-light quality",
    "award": "quality",
    "sample": "sample size",
    "spicy": "spiciness",
    "recent documents": "coverage",
    "offline": "validation value",
    "questions": "difficulty",
    "complex opinions": "capability",
    "meeting": "meeting suitability",
    "source list": "collection difficulty",
    "structured table": "clarity",
}


def split_turns(dialogue: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in re.split(r"\n+", dialogue.strip()):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([^:：]{1,30})[:：]\s*(.+)$", line)
        if match:
            turns.append((match.group(1).strip(), match.group(2).strip()))
        else:
            turns.append(("Narrator", line))
    return turns


def guess_sentiment(text: str) -> tuple[str, float, str]:
    lower = text.lower()
    pos_hits = [word for word in POSITIVE_WORDS if word in lower or word in text]
    neg_hits = [word for word in NEGATIVE_WORDS if word in lower or word in text]
    if pos_hits and neg_hits:
        return "mixed", 0.62, ", ".join(pos_hits + neg_hits)
    if pos_hits:
        return "positive", 0.72, ", ".join(pos_hits)
    if neg_hits:
        return "negative", 0.72, ", ".join(neg_hits)
    return "neutral", 0.55, text[:80]


def extract_target_aspect(text: str) -> tuple[str, str, str]:
    patterns = [
        r"(?P<target>[\w\s-]{2,40})\s+(?:has|have|had|with)\s+(?P<aspect>[\w\s-]{2,40})",
        r"(?P<aspect>cleanliness|battery life|service|price|food|screen|delivery|accuracy|速度|价格|服务|清洁度|续航|准确率)",
    ]
    lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower, flags=re.IGNORECASE)
        if match:
            target = (match.groupdict().get("target") or "item").strip(" .")
            aspect = (match.groupdict().get("aspect") or "overall").strip(" .")
            return target, aspect, match.group(0).strip()
    words = re.findall(r"[A-Za-z][A-Za-z-]+|[\u4e00-\u9fa5]{2,}", text)
    target = " ".join(words[:2]) if words else "item"
    aspect = words[2] if len(words) > 2 else "overall"
    return target, aspect, text[:40]


def lexicon_candidates(text: str) -> list[dict[str, Any]]:
    lower = text.lower()
    lowered_targets = [(target, lower.find(target.lower())) for target in KNOWN_TARGETS if target.lower() in lower]
    found_targets = [(target, pos) for target, pos in lowered_targets if pos >= 0]
    found_opinions = [(opinion, lower.find(opinion.lower()), sentiment) for opinion, sentiment in OPINION_SENTIMENT.items() if opinion.lower() in lower]
    found_opinions = [(opinion, pos, sentiment) for opinion, pos, sentiment in found_opinions if pos >= 0]
    if not found_targets or not found_opinions:
        return []

    rows: list[dict[str, Any]] = []
    used = set()
    for target, target_pos in found_targets:
        opinion, opinion_pos, sentiment = min(found_opinions, key=lambda item: abs(item[1] - target_pos))
        key = (target.lower(), opinion.lower())
        if key in used:
            continue
        used.add(key)
        rows.append(
            {
                "target": target,
                "aspect": guess_aspect(lower, target),
                "opinion": opinion,
                "sentiment": sentiment,
                "rationale": text,
                "confidence": 0.76,
            }
        )
    return rows


def guess_aspect(lower_text: str, target: str) -> str:
    for hint, aspect in ASPECT_HINTS.items():
        if hint.lower() in lower_text:
            return aspect
    if target.lower() in {"budget hotels", "room", "hotel room"} and "clean" in lower_text:
        return "cleanliness"
    return target


def heuristic_extract(dialogue: str, doc_id: str) -> dict[str, Any]:
    turns = split_turns(dialogue)
    sextuples = []
    sentiment_history: dict[tuple[str, str, str], str] = {}
    flips = []

    for speaker, text in turns:
        candidates = lexicon_candidates(text)
        if not candidates:
            target, aspect, opinion = extract_target_aspect(text)
            sentiment, confidence, rationale = guess_sentiment(text)
            candidates = [
                {
                    "target": target,
                    "aspect": aspect,
                    "opinion": opinion,
                    "sentiment": sentiment,
                    "rationale": rationale,
                    "confidence": confidence,
                }
            ]
        for candidate in candidates:
            row = normalize_sextuple({"doc_id": doc_id, "speaker": speaker, **candidate, "sentiment_flip": "None"}, doc_id)
            key = (speaker.lower(), row["target"].lower(), row["aspect"].lower())
            previous = sentiment_history.get(key)
            if previous and previous != row["sentiment"]:
                row["sentiment_flip"] = f"{previous}->{row['sentiment']}"
                flips.append(
                    {
                        "speaker": speaker,
                        "target": row["target"],
                        "aspect": row["aspect"],
                        "from_sentiment": previous,
                        "to_sentiment": row["sentiment"],
                        "trigger": text,
                    }
                )
            sentiment_history[key] = row["sentiment"]
            sextuples.append(row)

    return {
        "doc_id": doc_id,
        "target_aspects": [{"target": item["target"], "aspect": item["aspect"]} for item in sextuples],
        "quadruples": [
            {
                "speaker": item["speaker"],
                "target": item["target"],
                "aspect": item["aspect"],
                "opinion": item["opinion"],
            }
            for item in sextuples
        ],
        "sextuples": sextuples,
        "sentiment_flips": flips,
        "valid_json": True,
        "json_errors": [],
        "latency_seconds": 0.0,
        "mode": "heuristic-demo",
    }
