from app.schemas import normalize_sextuple, validate_pipeline_result


def test_normalize_sextuple_maps_chinese_sentiment_and_bounds_confidence():
    row = normalize_sextuple(
        {
            "speaker": "张三",
            "target": "房间",
            "aspect": "卫生",
            "opinion": "干净",
            "sentiment": "积极",
            "confidence": 2,
        },
        doc_id="zh_case",
    )

    assert row["doc_id"] == "zh_case"
    assert row["speaker"] == "张三"
    assert row["sentiment"] == "positive"
    assert row["confidence"] == 1.0
    assert row["sentiment_flip"] == "None"


def test_validate_pipeline_result_normalizes_bad_flip_container():
    result = validate_pipeline_result(
        {
            "doc_id": "d1",
            "sextuples": [{"target": "service", "opinion": "slow", "sentiment": "negative"}],
            "sentiment_flips": {"bad": "container"},
            "valid_json": 1,
        }
    )

    assert result["valid_json"] is True
    assert result["sentiment_flips"] == []
    assert result["sextuples"][0]["sentiment"] == "negative"
