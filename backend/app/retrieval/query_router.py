from __future__ import annotations

import re
from typing import Dict, List


class ClassroomQueryRouter:
    """Build a deterministic execution plan for the sequential Accurate cascade."""

    ACTION_TERMS = {
        "เดิน", "วิ่ง", "ยืน", "นั่ง", "ล้ม", "ลุก", "ยกมือ", "เขียน", "หยิบ", "ถือ", "วาง", "ส่ง", "คุย", "เข้า", "ออก", "เคลื่อน", "เลี้ยว",
        "walk", "walking", "run", "running", "stand", "standing", "sit", "sitting", "fall", "falling", "raise", "write", "pick", "hold", "holding", "put", "give", "talk", "enter", "leave", "move", "turn",
    }
    RELATION_TERMS = {
        "คุยกับ", "ยื่นให้", "นั่งข้าง", "อยู่หน้า", "อยู่หลัง", "พร้อมกับ",
        "talking to", "giving", "next to", "in front of", "behind", "with",
    }
    SPATIAL_TERMS = {
        "ซ้าย", "ขวา", "หน้า", "หลัง", "บน", "ใต้", "ข้าง", "ใกล้", "อยู่ตรงไหน",
        "left", "right", "front", "behind", "under", "next", "near", "where", "position", "location",
    }
    COUNT_TERMS = {"จำนวน", "กี่คน", "กี่", "how many", "count"}
    TEMPORAL_TERMS = {"ก่อน", "หลังจาก", "ระหว่าง", "เริ่ม", "หยุด", "before", "after", "during", "start", "stop"}

    PROMPT_GROUPS = (
        (("พระพุทธรูป", "buddha statue"), "Buddha statue"),
        (("พระสงฆ์", "buddhist monk", "monk"), "Buddhist monk"),
        (("รูปปั้นแมว", "แมวกวัก", "cat statue", "lucky cat", "maneki-neko"), "cat statue"),
        (("รถยนต์", "รถ", "car", "vehicle"), "car or vehicle"),
        (("กระเป๋า", "bag", "backpack"), "backpack or school bag"),
        (("โทรศัพท์", "มือถือ", "phone", "mobile", "smartphone"), "mobile phone"),
        (("หนังสือ", "book"), "book"),
        (("โน้ตบุ๊ก", "คอมพิวเตอร์", "laptop", "computer"), "laptop or computer"),
        (("สุนัข", "หมา", "dog", "puppy"), "dog"),
        (("แมว", "cat", "kitten"), "cat"),
        (("คน", "บุคคล", "person", "people", "shirt", "เสื้อ"), "person"),
    )

    def route(self, query: str) -> Dict[str, object]:
        text = query.casefold().strip()
        semantic_requirements: List[str] = []
        for name, terms in (
            ("action", self.ACTION_TERMS),
            ("relation", self.RELATION_TERMS),
            ("count", self.COUNT_TERMS),
            ("spatial", self.SPATIAL_TERMS),
            ("temporal", self.TEMPORAL_TERMS),
        ):
            if any(term in text for term in terms):
                semantic_requirements.append(name)

        ambiguous = False
        ambiguity_reasons: List[str] = []
        prompts: List[str] = []
        explicit_object = False
        if re.search(r"(^|\s)พระ($|\s)", text):
            prompts.extend(["Buddha statue", "Buddhist monk"])
            ambiguous = True
            ambiguity_reasons.append("thai_term_phra_can_mean_statue_or_monk")
            explicit_object = True
        else:
            for aliases, prompt in self.PROMPT_GROUPS:
                if any(alias in text for alias in aliases):
                    prompts.append(prompt)
                    explicit_object = True
            if "cat statue" in prompts and "cat" in prompts:
                prompts.remove("cat")

        has_human_action = any(term in text for term in self.ACTION_TERMS) and not any(
            term in text for term in ("รถ", "car", "vehicle")
        )
        if not prompts and has_human_action:
            prompts.append("person")

        if not prompts and re.search(r"[a-z]", text):
            words = re.findall(r"[a-z][a-z0-9-]*", text)
            stop = {"a", "an", "the", "is", "are", "to", "of", "in", "on", "at", "find", "show", "me"}
            noun_phrase = " ".join(word for word in words if word not in stop)
            if noun_phrase:
                prompts.append(noun_phrase)

        if not prompts:
            prompts.append(query.strip() or "object")
            ambiguous = True
            ambiguity_reasons.append("untranslated_or_unknown_query")

        prompts = list(dict.fromkeys(prompts))[:3]
        has_action_semantics = bool(set(semantic_requirements) & {"action", "relation", "temporal"})
        legacy_route = "mixed" if has_action_semantics and explicit_object else "action_relation" if has_action_semantics else "object_grounding"
        legacy_prompts = [
            "person carrying a backpack or school bag" if prompt == "backpack or school bag" else prompt
            for prompt in prompts
        ]
        return {
            "policy": "always_sam_then_qwen",
            "sam_prompts": prompts,
            "semantic_requirements": semantic_requirements,
            "ambiguous": ambiguous,
            "ambiguity_reasons": ambiguity_reasons,
            "planner_version": "cascade-planner-v2",
            "route": legacy_route,
            "object_prompts": legacy_prompts,
            "action_text": query,
            "signals": semantic_requirements,
            "router_version": "classroom-router-v1",
        }


query_router = ClassroomQueryRouter()
