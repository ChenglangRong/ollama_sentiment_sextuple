from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from .config import OUTPUT_DIR
from .data_loader import load_examples, load_public_aste, read_jsonl, write_jsonl
from .metrics import aggregate, score_flip, score_triplets
from .pipeline import SextuplePipeline


def run_inference(
    model: str,
    input_path: str,
    output_path: str,
    mode: str = "four-step",
) -> list[dict[str, Any]]:
    pipeline = SextuplePipeline()
    rows = load_examples(input_path)
    predictions = []
    for idx, row in enumerate(rows):
        result = pipeline.run(row.get("dialogue", row.get("text", "")), model=model, doc_id=row.get("doc_id", f"doc_{idx}"), mode=mode)
        predictions.append(result)
    write_jsonl(Path(output_path), predictions)
    return predictions


def evaluate_models(
    models: list[str],
    limit: int = 120,
    mode: str = "four-step",
    dataset: str = "public-aste",
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if dataset == "examples":
        samples = load_examples()
    else:
        samples = load_public_aste(limit=limit)

    pipeline = SextuplePipeline()
    summary_rows = []
    detail_rows = []
    raw_predictions: dict[str, list[dict[str, Any]]] = {}

    for model in models:
        print(f"[evaluate] model={model} mode={mode} dataset={dataset} samples={len(samples[:limit])}", flush=True)
        sample_scores = []
        valid_json_count = 0
        latencies = []
        predictions = []
        started_model = time.perf_counter()

        for idx, sample in enumerate(samples[:limit]):
            doc_id = sample.get("doc_id", f"sample_{idx}")
            dialogue = sample.get("dialogue", sample.get("text", ""))
            print(f"[evaluate] {model} {mode} {idx + 1}/{len(samples[:limit])} doc_id={doc_id}", flush=True)
            result = pipeline.run(dialogue, model=model, doc_id=doc_id, mode=mode)
            predictions.append(result)
            valid_json_count += int(bool(result.get("valid_json")))
            latencies.append(float(result.get("latency_seconds", 0.0)))
            scores = score_triplets(result.get("sextuples", []), sample.get("gold_triplets", []))
            sample_scores.append(scores)
            detail_rows.append(
                {
                    "model": model,
                    "doc_id": doc_id,
                    "precision": scores["precision"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    "sentiment_accuracy": scores["sentiment_accuracy"],
                    "exact_precision": scores["exact_precision"],
                    "exact_recall": scores["exact_recall"],
                    "exact_f1": scores["exact_f1"],
                    "valid_json": bool(result.get("valid_json")),
                    "latency_seconds": result.get("latency_seconds", 0.0),
                    "gold": json.dumps(sample.get("gold_triplets", []), ensure_ascii=False),
                    "predicted": json.dumps(result.get("sextuples", []), ensure_ascii=False),
                }
            )

        metrics = aggregate(sample_scores)
        elapsed = time.perf_counter() - started_model
        summary_rows.append(
            {
                "model": model,
                "mode": mode,
                "dataset": dataset,
                "samples": len(samples[:limit]),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1", 0.0),
                "sentiment_accuracy": metrics.get("sentiment_accuracy", 0.0),
                "valid_json_rate": valid_json_count / len(samples[:limit]) if samples[:limit] else 0.0,
                "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
                "total_seconds": elapsed,
            }
        )
        raw_predictions[model] = predictions
        write_jsonl(OUTPUT_DIR / f"predictions_{safe_name(model)}_{mode}.jsonl", predictions)
        write_csv(OUTPUT_DIR / f"evaluation_details_{safe_name(dataset)}_{safe_name(mode)}_{safe_name(model)}.csv", detail_rows)

    suffix = f"{safe_name(dataset)}_{safe_name(mode)}"
    write_csv(OUTPUT_DIR / "evaluation_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "evaluation_details.csv", detail_rows)
    write_csv(OUTPUT_DIR / f"evaluation_summary_{suffix}.csv", summary_rows)
    write_csv(OUTPUT_DIR / f"evaluation_details_{suffix}.csv", detail_rows)
    return {"summary": summary_rows, "details": detail_rows, "predictions": raw_predictions}


def evaluate_flip_examples(model: str = "heuristic-demo", mode: str = "four-step") -> dict[str, Any]:
    samples = read_jsonl(Path(__file__).resolve().parents[1] / "data" / "dialogue_flip_gold.jsonl")
    pipeline = SextuplePipeline()
    rows = []
    scores = []
    print(f"[evaluate-flips] model={model} mode={mode} samples={len(samples)}", flush=True)
    for idx, sample in enumerate(samples):
        print(f"[evaluate-flips] {model} {mode} {idx + 1}/{len(samples)} doc_id={sample['doc_id']}", flush=True)
        result = pipeline.run(sample["dialogue"], model=model, doc_id=sample["doc_id"], mode=mode)
        score = score_flip(result.get("sentiment_flips", []), sample.get("gold_flips", []))
        scores.append(score)
        rows.append(
            {
                "doc_id": sample["doc_id"],
                "model": model,
                **score,
                "predicted": json.dumps(result.get("sentiment_flips", []), ensure_ascii=False),
                "gold": json.dumps(sample.get("gold_flips", []), ensure_ascii=False),
            }
        )
    write_csv(OUTPUT_DIR / "flip_evaluation.csv", rows)
    summary = aggregate(scores)
    with (OUTPUT_DIR / "flip_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return {"summary": summary, "details": rows}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")
