"""Offline audit of existing Qwen evidence for action-selected candidates.

No new Qwen inference, no assertion of end-to-end runtime/call savings. Requires
identical source-window and retry-span bounds; Qwen still saw only eight frames.
"""
import json
from pathlib import Path

import numpy as np

from .benchmark_action_expert import attempts, digest
from .validate_dense_verification import metrics


def cached_rows(action_row, cached):
    if not cached or not np.allclose(cached[0]["bounds"], action_row["window_bounds"], rtol=0, atol=1e-6):
        raise ValueError("Cached Qwen window mismatch")
    for row in cached[1:]:
        if not np.allclose(row["bounds"], action_row["evidence"]["bounds"], rtol=0, atol=1e-6):
            raise ValueError("Cached Qwen retry span mismatch")
    return cached


def main():
    root = Path("scratch/event-retrieval")
    target = root/"action-expert-cached-cascade.json"
    if target.exists():
        raise ValueError("Preserve existing audit")
    action_path = root/"action-expert-validation/results.json"
    qwen_path = root/"motion-zoom-validation/results.json"
    lock_path = root/"action-expert-recall-lock.json"
    action = json.loads(action_path.read_text(encoding="utf-8"))
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not action["complete"] or not qwen["complete"] or action["protocol"]["sequences"] != qwen["sequences"]:
        raise ValueError("Incomplete/mismatched validation artifacts")
    selected = attempts(action["rows"], lock["threshold"])
    kept = {}
    for identifier, values in selected.items():
        if values[0]["reply"]["verdict"] and values[0]["reply"]["verdict"]["event_present"]:
            kept[identifier] = cached_rows(action["rows"][identifier], qwen["attempts"][identifier])
    source = json.loads((root/"continuous-ut/results.json").read_text(encoding="utf-8"))
    result = dict(scope="Offline matched cached Qwen controls/retries on selected candidates; no fresh Qwen calls",
        action_sha256=digest(action_path), qwen_sha256=digest(qwen_path), lock_sha256=digest(lock_path),
        selected_candidates=len(kept), candidate_ids=list(kept),
        cached_qwen_metrics=metrics(source, kept, (13, 14, 15, 16), True),
        full_cached_qwen_metrics=metrics(source, qwen["attempts"], (13, 14, 15, 16), True))
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
