from app.metrics import score_flip, score_triplets


def test_score_triplets_uses_soft_span_matching():
    predicted = [{"target": "battery", "opinion": "excellent", "sentiment": "positive"}]
    gold = [{"target": "battery life", "opinion": "excellent", "sentiment": "positive"}]

    scores = score_triplets(predicted, gold)

    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_score_flip_requires_same_speaker_target_aspect_and_direction():
    predicted = [{"speaker": "Alice", "target": "service", "aspect": "speed", "from_sentiment": "negative", "to_sentiment": "positive"}]
    gold = [{"speaker": "Alice", "target": "service", "aspect": "speed", "from_sentiment": "negative", "to_sentiment": "positive"}]

    scores = score_flip(predicted, gold)

    assert scores["flip_precision"] == 1.0
    assert scores["flip_recall"] == 1.0
    assert scores["flip_f1"] == 1.0
