from scripts.score_startup import score_startup, WEIGHTS


def test_all_100_scores_to_1000():
    record = {
        "startup_id": "x",
        "name": "X",
        "as_of": "2026-09-29T00:00:00Z",
        "dimensions": {
            name: {"score": 100, "evidence_ids": ["e1"]}
            for name in WEIGHTS
        },
    }
    result = score_startup(record)
    assert result["score"] == 1000.0
    assert result["confidence"] == 100.0


def test_all_50_scores_to_500():
    record = {
        "startup_id": "x",
        "name": "X",
        "as_of": "2026-09-29T00:00:00Z",
        "dimensions": {name: {"score": 50} for name in WEIGHTS},
    }
    result = score_startup(record)
    assert result["score"] == 500.0
    assert result["provisional"] is True
