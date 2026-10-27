import math
import numpy as np

def iou(a, b):
    overlap = max(0., min(a[1], b[1])-max(a[0], b[0]))
    union = max(a[1], b[1])-min(a[0], b[0])
    return overlap/union if union > 0 else 0.

def wilson(successes, count):
    if not count:
        return None
    p, z = successes/count, 1.96
    center = (p + z*z/(2*count)) / (1+z*z/count)
    width = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0., center-width), min(1., center+width)]

def retrieval(records, threshold=.3):
    result = {}
    for language in ("th", "en", "all"):
        selected = [row for row in records if language == "all" or row["language"] == language]
        positive = [row for row in selected if row.get("truth")]
        negative = [row for row in selected if not row.get("truth")]
        metrics = {"queries": len(selected), "answerable": len(positive), "no_match_queries": len(negative)}
        for k in (1, 5):
            hit = sum(any(max(iou((p["t_start"], p["t_end"]), target) for target in row["truth"]) >= threshold
                          for p in row["moments"][:k]) for row in positive)
            metrics[f"recall@{k}"] = hit/len(positive) if positive else None
            metrics[f"recall@{k}_95ci"] = wilson(hit, len(positive))
        metrics["mean_top1_tiou"] = float(np.mean([max((iou((row["moments"][0]["t_start"], row["moments"][0]["t_end"]), target) for target in row["truth"]), default=0.) if row["moments"] else 0. for row in positive])) if positive else None
        metrics["no_match_false_positive_rate"] = sum(bool(row["moments"]) for row in negative)/len(negative) if negative else None
        result[language] = metrics
    return result

def event_metrics(predicted, truth, normal_hours, threshold=.3):
    # Greedy highest-IoU one-to-one matching: duplicate detections are false positives.
    pairs = []
    for p, prediction in enumerate(predicted):
        for t, target in enumerate(truth):
            if prediction["kind"] == target["kind"] and prediction["video_id"] == target["video_id"]:
                score = iou((prediction["t_start"], prediction["t_end"]), (target["t_start"], target["t_end"]))
                if score >= threshold:
                    pairs.append((score, p, t))
    matched_p, matched_t = set(), set()
    for score, p, t in sorted(pairs, reverse=True):
        if p not in matched_p and t not in matched_t:
            matched_p.add(p); matched_t.add(t)
    tp = len(matched_t)
    false_on_normal = sum(bool(row.get("normal_recording")) for row in predicted)
    return {"tp": tp, "fp": len(predicted)-tp, "fn": len(truth)-tp,
            "recall": tp/len(truth) if truth else None, "precision": tp/len(predicted) if predicted else None,
            "recall_95ci": wilson(tp, len(truth)), "precision_95ci": wilson(tp, len(predicted)),
            "normal_hours": normal_hours, "false_alerts_per_hour": false_on_normal/normal_hours if normal_hours > 0 else None}

def gates(screening, final, performance):
    targets = {"screen_recall": (screening.get("recall"), .95, "min"), "final_recall": (final.get("recall"), .85, "min"),
        "precision": (final.get("precision"), .80, "min"), "false_alerts_per_hour": (final.get("false_alerts_per_hour"), 2., "max"),
        "allocated_gib": (performance.get("allocated_gib"), 6., "max"), "fast_p95_seconds": (performance.get("fast_p95_seconds"), 2., "max"),
        "qwen_call_reduction": (performance.get("qwen_call_reduction"), .5, "min")}
    checks = {name: {"value": value, "target": target, "passed": value is not None and (value >= target if direction == "min" else value <= target)} for name, (value, target, direction) in targets.items()}
    return {"passed": all(row["passed"] for row in checks.values()), "checks": checks}
