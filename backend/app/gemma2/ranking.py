import numpy as np

def rank_channels(query, channels):
    ranked, evidence = {}, {}
    for kind, (ids, matrix) in channels.items():
        values = matrix @ query
        # Multiple image/caption vectors refer to the same temporal window.
        scores = {}
        for window_id, score in zip(ids, values):
            key = int(window_id)
            scores[key] = max(scores.get(key, -1.0), float(score))
        ranked[kind] = sorted(scores, key=lambda key: (-scores[key], key))
        for key, score in scores.items():
            evidence.setdefault(key, {})[kind] = score
    return ranked, evidence

def rrf(rankings, k=60):
    fused = {}
    for kind, ranking in rankings.items():
        for position, key in enumerate(ranking, 1):
            fused[key] = fused.get(key, 0.0) + 1 / (k + position)
    return sorted(fused, key=lambda key: (-fused[key], key)), fused

EVENTS = {
    "shouting": {
        "label": "ตะโกน / กรีดร้อง", "channels": ("audio",),
        "positive": ["เสียงคนตะโกนหรือกรีดร้องดัง", "A person shouting loudly or screaming"],
        "negative": ["เสียงคนพูดคุยตามปกติ", "People talking calmly at normal volume", "Music playing without a person screaming"],
    },
    "pushing_punching": {
        "label": "ผลัก / ชก", "channels": ("video", "image"),
        "positive": ["คนใช้มือผลักหรือชกคนอื่น", "A person forcefully pushing or punching another person"],
        "negative": ["คนจับมือทักทายกัน", "People shaking hands or gently helping another person", "People walking past one another"],
    },
    "falling": {
        "label": "ล้ม", "channels": ("video", "image"),
        "positive": ["คนเสียการทรงตัวล้มลงบนพื้น", "A person losing balance and falling to the floor"],
        "negative": ["คนนั่งลงบนเก้าอี้ตามปกติ", "A person sitting down normally or bending to pick up an object", "A person lying down intentionally"],
    },
}

def screen(channels, prompt_vectors, kinds, threshold=0.35, margin=0.02):
    candidates = []
    for kind in kinds:
        spec = EVENTS[kind]
        positives, negatives = prompt_vectors[kind]
        evidence = {}
        for modality in spec["channels"]:
            if modality not in channels:
                continue
            ids, matrix = channels[modality]
            positive = (matrix @ positives.T).max(axis=1)
            negative = (matrix @ negatives.T).max(axis=1)
            for slot, window_id in enumerate(ids):
                score = float(positive[slot])
                delta = float(positive[slot] - negative[slot])
                if score >= threshold and delta >= margin:
                    item = {"similarity": score, "normal_similarity": float(negative[slot]), "margin": delta}
                    previous = evidence.setdefault(int(window_id), {}).get(modality)
                    if previous is None or score > previous["similarity"]:
                        evidence[int(window_id)][modality] = item
        candidates.extend({"kind": kind, "window_id": key, "evidence": value} for key, value in evidence.items())
    return candidates

def merge_candidates(candidates, timeline, gap=3.0, max_seconds=None):
    merged = []
    lookup = {row["id"]: row for row in timeline}
    for kind in EVENTS:
        rows = sorted([item for item in candidates if item["kind"] == kind], key=lambda item: lookup[item["window_id"]]["t_start"])
        for row in rows:
            window = lookup[row["window_id"]]
            if max_seconds is not None and window["t_end"] - window["t_start"] > max_seconds + 1e-6:
                raise ValueError("Candidate window exceeds verification duration budget")
            if (merged and merged[-1]["kind"] == kind
                    and window["t_start"] <= merged[-1]["t_end"] + gap
                    and (max_seconds is None or max(merged[-1]["t_end"], window["t_end"]) - merged[-1]["t_start"] <= max_seconds + 1e-6)):
                previous = merged[-1]
                previous["t_end"] = max(previous["t_end"], window["t_end"])
                previous["window_ids"].append(row["window_id"])
                previous["evidence"].append(row["evidence"])
            else:
                merged.append({"kind": kind, "t_start": window["t_start"], "t_end": window["t_end"], "window_ids": [row["window_id"]], "evidence": [row["evidence"]]})
    return merged
