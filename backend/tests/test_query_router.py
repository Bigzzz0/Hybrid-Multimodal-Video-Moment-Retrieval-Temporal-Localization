import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.query_router import ClassroomQueryRouter


def test_router_sends_thai_object_query_to_sam():
    routed = ClassroomQueryRouter().route("คนเสื้อสีแดงอยู่ทางซ้าย")
    assert routed["route"] == "object_grounding"
    assert "person" in routed["object_prompts"]


def test_router_sends_english_action_query_to_qwen():
    routed = ClassroomQueryRouter().route("a student is walking into the classroom")
    assert routed["route"] == "action_relation"
    assert routed["object_prompts"] == [] or routed["object_prompts"] == ["person"]


def test_router_sends_mixed_query_to_sam_then_qwen():
    routed = ClassroomQueryRouter().route("คนถือกระเป๋าเดินไปหน้าห้อง")
    assert routed["route"] == "mixed"
    assert "person carrying a backpack or school bag" in routed["object_prompts"]
    assert routed["router_version"] == "classroom-router-v1"


def test_router_sends_standalone_cat_query_to_sam():
    routed = ClassroomQueryRouter().route("แมว")
    assert routed["route"] == "object_grounding"
    assert routed["object_prompts"] == ["cat"]
