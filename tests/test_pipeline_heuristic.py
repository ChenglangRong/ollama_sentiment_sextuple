from app.cli import build_parser
from app.pipeline import SextuplePipeline


def test_empty_input_returns_valid_empty_result_with_model_metadata():
    result = SextuplePipeline().run("", model="heuristic-demo", doc_id="empty", mode="four-step")

    assert result["doc_id"] == "empty"
    assert result["model"] == "heuristic-demo"
    assert result["mode"] == "four-step"
    assert result["valid_json"] is True
    assert result["sextuples"] == []
    assert result["sentiment_flips"] == []


def test_heuristic_pipeline_detects_a_simple_flip_offline():
    dialogue = (
        "Alice: budget hotels have dirty cleanliness.\n"
        "Alice: budget hotels have high cleanliness standards now."
    )

    result = SextuplePipeline().run(dialogue, model="heuristic-demo", doc_id="flip_case", mode="four-step")

    assert result["model"] == "heuristic-demo"
    assert result["valid_json"] is True
    assert len(result["sextuples"]) >= 2
    assert any(
        flip["from_sentiment"] == "negative" and flip["to_sentiment"] == "positive"
        for flip in result["sentiment_flips"]
    )


def test_cli_parser_accepts_required_final_project_commands():
    parser = build_parser()

    check = parser.parse_args(["check-ollama"])
    evaluate = parser.parse_args(["evaluate", "--models", "qwen2.5:7b", "qwen2.5:3b", "--limit", "120", "--mode", "four-step"])

    assert check.command == "check-ollama"
    assert evaluate.command == "evaluate"
    assert evaluate.models == ["qwen2.5:7b", "qwen2.5:3b"]
    assert evaluate.limit == 120
    assert evaluate.mode == "four-step"
