from __future__ import annotations

import argparse
import json

from .config import DEFAULT_MODELS, OLLAMA_BASE_URL
from .evaluation import evaluate_flip_examples, evaluate_models, run_inference
from .ollama_client import OllamaClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama sentiment sextuple extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-ollama", help="Check Ollama API and installed models")
    check.add_argument("--base-url", default=OLLAMA_BASE_URL)

    infer = subparsers.add_parser("infer", help="Run batch inference")
    infer.add_argument("--model", default="qwen2.5:7b")
    infer.add_argument("--input", default="data/examples.jsonl")
    infer.add_argument("--output", default="outputs/predictions.jsonl")
    infer.add_argument("--mode", choices=["four-step", "one-shot"], default="four-step")

    evaluate = subparsers.add_parser("evaluate", help="Run public dataset evaluation")
    evaluate.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    evaluate.add_argument("--limit", type=int, default=120)
    evaluate.add_argument("--mode", choices=["four-step", "one-shot"], default="four-step")
    evaluate.add_argument("--dataset", choices=["public-aste", "examples"], default="public-aste")

    flips = subparsers.add_parser("evaluate-flips", help="Run sentiment flip evaluation")
    flips.add_argument("--model", default="heuristic-demo")
    flips.add_argument("--mode", choices=["four-step", "one-shot"], default="four-step")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "check-ollama":
        client = OllamaClient(base_url=args.base_url)
        print(json.dumps({"available": client.is_available(), "models": client.list_models()}, ensure_ascii=False, indent=2))
    elif args.command == "infer":
        predictions = run_inference(args.model, args.input, args.output, args.mode)
        print(json.dumps({"output": args.output, "count": len(predictions)}, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        result = evaluate_models(args.models, limit=args.limit, mode=args.mode, dataset=args.dataset)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    elif args.command == "evaluate-flips":
        result = evaluate_flip_examples(args.model, args.mode)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

