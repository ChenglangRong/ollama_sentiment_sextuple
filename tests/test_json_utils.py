from app.json_utils import extract_json_object


def test_extract_json_object_repairs_fences_and_trailing_commas():
    text = '```json\n{"sextuples": [{"target": "room", "sentiment": "positive",}],}\n```'

    parsed, valid, error = extract_json_object(text)

    assert valid is True
    assert error == "repaired common JSON issues"
    assert parsed["sextuples"][0]["target"] == "room"


def test_extract_json_object_recovers_completed_arrays_from_truncated_output():
    text = '{"doc_id":"d1","sextuples":[{"target":"service","opinion":"good","sentiment":"positive"}],'

    parsed, valid, error = extract_json_object(text)

    assert valid is True
    assert "recovered partial JSON" in error
    assert parsed["doc_id"] == "d1"
    assert parsed["sextuples"][0]["opinion"] == "good"
