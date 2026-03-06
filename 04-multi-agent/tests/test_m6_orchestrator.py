from multi_agent.m6_orchestrator import _clamp_confidence, _parse_json_like


def test_parse_json_like_handles_fenced_json() -> None:
    raw = "```json\n{\"action\": \"compose_answer\", \"claim_ids\": [1]}\n```"
    data = _parse_json_like(raw)
    assert data["action"] == "compose_answer"
    assert data["claim_ids"] == [1]


def test_clamp_confidence_bounds_values() -> None:
    assert _clamp_confidence(1.7) == 1.0
    assert _clamp_confidence(-0.1) == 0.0
    assert _clamp_confidence("0.4") == 0.4
