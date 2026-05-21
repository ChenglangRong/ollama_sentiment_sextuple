from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

from .config import DATA_DIR


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_examples(path: str | None = None) -> list[dict[str, Any]]:
    source = Path(path) if path else DATA_DIR / "examples.jsonl"
    return read_jsonl(source)


def parse_aste_output(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = []
    triplets = []
    for item in value or []:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            triplets.append(
                {
                    "target": str(item[0]).strip(),
                    "opinion": str(item[1]).strip(),
                    "sentiment": str(item[2]).strip().lower(),
                }
            )
    return triplets


def load_public_aste(limit: int = 120, seed: int = 42) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        dataset = load_dataset("NEUDM/aste-data-v2", split="train")
        rows = []
        for idx, item in enumerate(dataset):
            text_value = item.get("input", "")
            if isinstance(text_value, list):
                text = " ".join(map(str, text_value))
            else:
                text = str(text_value)
            gold = parse_aste_output(item.get("output", "[]"))
            if gold:
                rows.append({"doc_id": f"aste_{idx}", "dialogue": f"Reviewer: {text}", "gold_triplets": gold})
        random.Random(seed).shuffle(rows)
        if rows:
            return rows[:limit]
    except Exception:
        pass

    fallback = read_jsonl(DATA_DIR / "public_aste_fallback.jsonl")
    return fallback[:limit]

