from __future__ import annotations

import re
from typing import Dict, List


class ClassroomQueryRouter:
    """Deterministic Thai/English routing for the expensive second stage."""

    OBJECT_TERMS = {
        "คน", "คนที่", "เสื้อ", "สี", "กระเป๋า", "หนังสือ", "โทรศัพท์", "มือถือ", "โน้ตบุ๊ก", "คอม", "โต๊ะ", "เก้าอี้", "วัตถุ", "แมว", "สุนัข", "หมา",
        "person", "people", "shirt", "color", "bag", "backpack", "book", "phone", "mobile", "laptop", "computer", "table", "chair", "object", "cat", "kitten", "dog", "puppy",
    }
    SPATIAL_TERMS = {
        "ซ้าย", "ขวา", "หน้า", "หลัง", "บน", "ใต้", "ข้าง", "ใกล้", "จำนวน", "กี่คน", "ซ่อน", "อยู่ตรงไหน",
        "left", "right", "front", "behind", "under", "next", "near", "how many", "where", "position", "location",
    }
    ACTION_TERMS = {
        "เดิน", "วิ่ง", "ยืน", "นั่ง", "ล้ม", "ลุก", "ยกมือ", "เขียน", "หยิบ", "ถือ", "วาง", "ส่ง", "คุย", "เข้า", "ออก", "เคลื่อน", "ก่อน", "หลังจาก",
        "walk", "walking", "run", "running", "stand", "standing", "sit", "sitting", "fall", "falling", "raise", "write", "pick", "hold", "put", "give", "talk", "enter", "leave", "move", "before", "after",
    }
    RELATION_TERMS = {
        "คุยกับ", "ยื่นให้", "นั่งข้าง", "อยู่หน้า", "อยู่หลัง", "ถือ", "พร้อมกับ", "ระหว่าง",
        "talking to", "giving", "next to", "in front of", "behind", "holding", "with", "between",
    }

    def route(self, query: str) -> Dict[str, object]:
        value = query.casefold().strip()
        tokens = set(re.findall(r"[\wก-๙]+", value))
        object_hits = sorted(token for token in self.OBJECT_TERMS if token in value or token in tokens)
        spatial_hits = sorted(token for token in self.SPATIAL_TERMS if token in value or token in tokens)
        action_hits = sorted(token for token in self.ACTION_TERMS if token in value or token in tokens)
        relation_hits = sorted(token for token in self.RELATION_TERMS if token in value)

        has_object = bool(object_hits or spatial_hits)
        has_action = bool(action_hits or relation_hits)
        route = "mixed" if has_object and has_action else "object_grounding" if has_object else "action_relation"
        return {
            "route": route,
            "object_prompts": self._object_prompts(query, object_hits),
            "action_text": query,
            "signals": sorted(set(object_hits + spatial_hits + action_hits + relation_hits)),
            "router_version": "classroom-router-v1",
        }

    @staticmethod
    def _object_prompts(query: str, hits: List[str]) -> List[str]:
        prompts: List[str] = []
        text = query.casefold()
        if any(term in text for term in ("กระเป๋า", "bag", "backpack")):
            prompts.append("person carrying a backpack or school bag")
        if any(term in text for term in ("โทรศัพท์", "มือถือ", "phone", "mobile")):
            prompts.append("mobile phone")
        if any(term in text for term in ("หนังสือ", "book")):
            prompts.append("book")
        if any(term in text for term in ("โน้ตบุ๊ก", "คอม", "laptop", "computer")):
            prompts.append("laptop or computer")
        if any(term in text for term in ("แมว", "cat", "kitten")):
            prompts.append("cat")
        if any(term in text for term in ("สุนัข", "หมา", "dog", "puppy")):
            prompts.append("dog")
        if any(term in text for term in ("คน", "person", "people", "เสื้อ", "shirt")):
            prompts.append("person")
        return list(dict.fromkeys(prompts)) or ["person"]


query_router = ClassroomQueryRouter()
