import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.grounding import GroundingOrchestrator


def test_sam_cache_key_is_deterministic_and_versioned():
    orchestrator = GroundingOrchestrator()
    first = orchestrator._cache_key("video-a", 1.0, 8.0, ["book", "person"], 4.0, 32)
    second = orchestrator._cache_key("video-a", 1.0, 8.0, ["person", "book"], 4.0, 32)
    changed_sampling = orchestrator._cache_key("video-a", 1.0, 8.0, ["book", "person"], 2.0, 32)
    assert first == second
    assert first != changed_sampling


def test_empty_grounding_request_is_a_noop():
    result = GroundingOrchestrator().ground_candidates("video-a", [], [], ["person"], limit=3)
    assert result == ([], [], {})
