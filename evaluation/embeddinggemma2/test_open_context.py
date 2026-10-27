import copy

from evaluation.embeddinggemma2.ablate_open_context import contextualize, RULE


def test_context_ablation_changes_only_context_and_never_mutates_baseline():
    baseline = [{"id": "unit", "frame_paths": ["frame"], "timestamps": [3.],
                 "reference_frames": [{"path": "normal"}], "scene_context": "A corridor"}]
    original = copy.deepcopy(baseline)
    changed = contextualize(baseline)
    assert baseline == original
    assert changed[0]["scene_context"] == "A corridor " + RULE
    assert {k: v for k, v in changed[0].items() if k != "scene_context"} == {
        k: v for k, v in baseline[0].items() if k != "scene_context"}
    changed[0]["reference_frames"][0]["path"] = "different"
    assert baseline == original
