"""Reproducible post-hoc sampling analysis. No model calls or label fitting."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .continuous_screening import measure
from .verify_continuous_candidates import summarize_policy


def is_positive(reply):
    return reply and reply.get("status") == "verified" and (reply.get("verdict") or {}).get("event_present") is True


def cascade(first, second, ids, locations):
    decisions, intervals = {}, {}
    calls = 0
    for identifier in ids:
        original, zoom = first.get(identifier), second.get(identifier)
        calls += (original or {}).get("calls", 0)
        start, end = locations[identifier]["bounds"]
        if is_positive(original):
            decisions[identifier] = "positive"; intervals[identifier] = [start, end]
        else:
            calls += (zoom or {}).get("calls", 0)
            if is_positive(zoom):
                span = min(3., end-start); left = (start+end-span)/2
                decisions[identifier] = "positive"; intervals[identifier] = [left, left+span]
            elif any(not reply or reply.get("status") != "verified" or reply.get("verdict") is None for reply in (original, zoom)):
                decisions[identifier] = "pending"
            else:
                decisions[identifier] = "negative"
    return decisions, intervals, calls


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", default="scratch/event-retrieval/continuous-ut")
    root = Path(parser.parse_args().output)
    source = json.loads((root/"results.json").read_text())
    uniform = json.loads((root/"qwen-results.json").read_text(encoding="utf-8"))
    central = json.loads((root/"qwen-central-results.json").read_text(encoding="utf-8"))
    if not uniform["complete"] or not central["complete"]:
        raise ValueError("Both paired runs must be complete")
    if uniform["policy_ids"] != central["policy_ids"] or uniform["policies"] != central["policies"]:
        raise ValueError("Candidate sets or thresholds differ")
    output = dict(scope="Post-test exploratory ablation; neither a fresh blind test nor production activation", runs={}, cascade={})
    for name, run in (("uniform", uniform), ("central", central)):
        output["runs"][name] = dict(elapsed_seconds=run["elapsed_seconds"], per_policy={
            method: summarize_policy(source, run["results"], run["locations"], ids, name) for method, ids in run["policy_ids"].items()})
    first = {reply["id"]: reply for reply in uniform["results"]}
    second = {reply["id"]: reply for reply in central["results"]}
    labels = {int(seq): rows for seq, rows in source["annotations"].items()}
    for method, ids in uniform["policy_ids"].items():
        decisions, intervals, calls = cascade(first, second, ids, uniform["locations"])
        records = []
        for seq in range(17, 21):
            bounds = copy.deepcopy(source["scores"][str(seq)]["bounds"])
            scores = np.full(len(bounds), -1.)
            for identifier, interval in intervals.items():
                where = uniform["locations"][identifier]
                if where["seq"] == seq:
                    bounds[where["window"]] = interval; scores[where["window"]] = 1.
            records.append((seq, {"bounds": np.array(bounds)}, {"positive": scores}))
        output["cascade"][method] = dict(coverage=measure(records, "positive", 0., labels),
            pending=sum(value == "pending" for value in decisions.values()), derived_private_requests=calls,
            cost_scope="Derived from paired cached replies; adaptive cascade wall time was not measured", decisions=decisions)
        print("CASCADE", method, output["cascade"][method]["coverage"]["covered_events"], "/ 8", "requests", calls, flush=True)
    (root/"sampling-analysis.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
