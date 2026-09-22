import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference.contracts import EmbedTextRequest, GroundRequest, VerifyRequest
from app.retrieval.query_router import ClassroomQueryRouter
from app.retrieval.search_engine import HybridMomentSearchEngine
from inference_worker.main import _run
from inference_worker.model_manager import model_manager


@pytest.mark.parametrize(
    ("query", "prompts", "ambiguous"),
    [
        ("พระ", ["Buddha statue", "Buddhist monk"], True),
        ("พระพุทธรูป", ["Buddha statue"], False),
        ("พระสงฆ์", ["Buddhist monk"], False),
        ("แมวกวัก", ["cat statue"], False),
        ("รถเลี้ยวขวา", ["car or vehicle"], False),
    ],
)
def test_cascade_planner_canonical_prompts(query, prompts, ambiguous):
    plan = ClassroomQueryRouter().route(query)
    assert plan["policy"] == "stored_caption_then_qwen"
    assert plan["sam_prompts"] == prompts
    assert plan["ambiguous"] is ambiguous
    assert plan["planner_version"] == "caption-cascade-v1"


def test_action_without_subject_still_grounds_person_first():
    plan = ClassroomQueryRouter().route("เดินไปทางซ้าย")
    assert plan["sam_prompts"] == ["person"]
    assert "action" in plan["semantic_requirements"]


def test_release_after_contract_is_backward_compatible():
    assert EmbedTextRequest(texts=["cat"]).release_after is False
    assert GroundRequest(prompts=["cat"], release_after=True).release_after is True
    verify = VerifyRequest(
        query="cat",
        semantic_requirements=["object_presence"],
        grounding_evidence=[{"concept": "cat"}],
        release_after=True,
    )
    assert verify.release_after is True
    assert verify.grounding_evidence[0]["concept"] == "cat"


def test_accurate_fusion_contract():
    fuse = HybridMomentSearchEngine._fuse_accurate_score
    assert fuse(0.5, sam_score=0.8) == pytest.approx(0.635)
    assert fuse(0.5, qwen_confidence=0.9) == pytest.approx(0.66)
    assert fuse(0.5, sam_score=0.8, qwen_confidence=0.9) == pytest.approx(0.695)
    assert fuse(0.5, qwen_confidence=0.9, qwen_present=False) == pytest.approx(0.1)


def test_worker_release_after_unloads_on_success(monkeypatch):
    calls = []

    class Service:
        def infer(self, request):
            calls.append("infer")
            return {"ok": True}

    service = Service()

    def fake_get(name, factory):
        model_manager.services[name] = service
        model_manager.states[name] = "ready"
        return service

    def fake_unload(name):
        calls.append(f"unload:{name}")
        model_manager.services.pop(name, None)
        model_manager.states[name] = "unloaded"

    monkeypatch.setattr(model_manager, "get", fake_get)
    monkeypatch.setattr(model_manager, "unload", fake_unload)
    result = asyncio.run(_run("sam", lambda: service, "infer", SimpleNamespace(release_after=True)))
    assert result == {"ok": True}
    assert calls == ["infer", "unload:sam"]
    assert model_manager.states["sam"] == "unloaded"
