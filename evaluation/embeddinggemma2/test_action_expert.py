import numpy as np
import pytest

from evaluation.embeddinggemma2.action_expert import action_scores, inputs, sample_indices
from evaluation.embeddinggemma2.benchmark_action_expert import attempts
from evaluation.embeddinggemma2.audit_action_cascade import cached_rows


def test_person_classes_are_one_based_and_exclude_object_actions():
    predictions = np.zeros((2, 80), dtype=np.float32)
    predictions[0, 33] = 1.  # hit object, must not be fight/hit person
    predictions[0, 45] = 1.  # push object
    predictions[0, 63] = .7
    predictions[1, 75] = .8
    predictions[1, 69] = .6
    scores = action_scores(predictions)
    assert scores["target"] == pytest.approx(.8)
    assert scores["normal"] == pytest.approx(.6)
    assert scores["margin"] == pytest.approx(.2)


def test_no_silent_frame_duplication_or_outside_clip_sampling():
    indices = sample_indices(1.01, 3.01, 30., 100)
    assert len(indices) == len(set(indices)) == 32
    assert np.all(indices/30 >= 1.01) and np.all(indices/30 < 3.01)
    with pytest.raises(ValueError, match="insufficient_unique_frames"):
        sample_indices(0., .5, 30., 100)
    with pytest.raises(ValueError):
        sample_indices(0., 2., float("nan"), 100)


def test_pathways_rgb_normalization_and_roi_scaling():
    frames = np.zeros((32, 64, 96, 3), dtype=np.uint8)
    frames[..., 2] = 255  # red in BGR
    pathways, rois = inputs(frames, [[10, 20, 30, 40]])
    slow, fast = pathways
    assert tuple(slow.shape) == (1, 3, 8, 256, 384)
    assert tuple(fast.shape) == (1, 3, 32, 256, 384)
    assert rois.tolist() == [[0, 40, 80, 160, 240]]
    assert fast[0, 0, 0, 0, 0].item() == pytest.approx((1-.45)/.225)
    assert fast[0, 2, 0, 0, 0].item() == pytest.approx(-.45/.225)
    assert np.allclose(slow.numpy(), fast[:, :, np.linspace(0, 31, 8).astype(int)].numpy())


def test_invalid_scores_or_boxes_fail_explicitly():
    with pytest.raises(ValueError):
        action_scores(np.full((1, 80), np.nan))
    with pytest.raises(ValueError):
        action_scores(np.zeros((1, 81)))
    with pytest.raises(ValueError):
        inputs(np.zeros((32, 64, 96, 3), dtype=np.uint8), [[100, 20, 30, 40]])


def test_unverified_remains_unknown_and_normal_margin_blocks_target():
    rows = {
        "candidate-1-0": dict(window_bounds=[0, 6], evidence=dict(status="unverified", bounds=[1, 3])),
        "candidate-1-1": dict(window_bounds=[3, 9], evidence=dict(status="scored", bounds=[4, 6],
            scores=dict(target=.9, margin=-.1))),
        "candidate-1-2": dict(window_bounds=[6, 12], evidence=dict(status="scored", bounds=[7, 9],
            scores=dict(target=.8, margin=.1))),
    }
    results = attempts(rows, .7)
    assert results["candidate-1-0"][0]["reply"]["verdict"] is None
    assert not results["candidate-1-1"][0]["reply"]["verdict"]["event_present"]
    assert results["candidate-1-2"][0]["reply"]["verdict"]["event_present"]
    assert results["candidate-1-2"][0]["bounds"] == [6, 12]
    assert attempts(rows, .7, True)["candidate-1-2"][0]["bounds"] == [7, 9]


def test_cache_cannot_be_joined_across_different_temporal_inputs():
    action = dict(window_bounds=[0, 6], evidence=dict(bounds=[2, 4]))
    cached = [dict(bounds=[0, 6]), dict(bounds=[2, 4])]
    assert cached_rows(action, cached) is cached
    with pytest.raises(ValueError, match="retry span mismatch"):
        cached_rows(action, [dict(bounds=[0, 6]), dict(bounds=[3, 5])])
    with pytest.raises(ValueError, match="window mismatch"):
        cached_rows(action, [dict(bounds=[1, 7])])
