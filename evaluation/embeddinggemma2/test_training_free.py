import numpy as np
import pytest

from .training_free import audit, calibrate, metrics, neighbor_margin, sequence_feature
from . import training_free
from app.gemma2.config import IndexConfig


def test_reference_splits_reject_same_recording_or_bytes():
    base = dict(id="one", recording_id="source", sha256="aaa", split="development")
    for other in (dict(id="two", recording_id="source", sha256="bbb", split="test"),
                  dict(id="two", recording_id="other", sha256="aaa", split="validation")):
        with pytest.raises(ValueError, match="leakage"):
            audit([base, other])


def test_threshold_selection_never_uses_test():
    with pytest.raises(ValueError, match="validation only"):
        calibrate([dict(split="test", label=True)], [.5])


def test_calibration_keeps_recall_before_reducing_false_positives():
    rows = [dict(split="validation", label=x) for x in (True, True, False, False)]
    threshold, result = calibrate(rows, [.8, .4, .6, .1])
    assert threshold == .4
    assert result["tp"] == 2 and result["fp"] == 1


def test_order_feature_changes_when_action_reversed_and_handles_short_clip():
    frames = np.eye(3)
    assert not np.allclose(sequence_feature(frames), sequence_feature(frames[::-1]))
    assert np.linalg.norm(sequence_feature(frames[:1])) == pytest.approx(1)


def test_reference_class_size_does_not_change_equal_neighbors():
    query = np.array([1., 0.])
    positive = np.array([[1., 0.]])
    normal = np.array([[0., 1.]])
    assert neighbor_margin(query, positive, normal) == neighbor_margin(query, np.repeat(positive, 20, axis=0), normal)


def test_empty_predictions_have_zero_precision_and_correct_false_negatives():
    assert metrics([True, False], [.2, .1], .5) == dict(tp=0, fp=0, fn=1, tn=1, recall=0., precision=0., selected=0, total=2)


def test_decode_uses_actual_container_duration_not_stale_manifest(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(training_free, "file_hash", lambda _: "abc")
    monkeypatch.setattr(training_free, "metadata", lambda _: {"duration": 1.9})
    def extract(*args):
        calls.append(args[2])
        return dict(frame_paths=["one", "two"], image_slots=[0, 1], timestamps=[0., 1.])
    monkeypatch.setattr(training_free, "extract", extract)
    class Client:
        last_telemetry = {}
        def embed(self, kind, config, **inputs):
            return np.ones((2 if kind == "image" else 1, 768))/np.sqrt(768)
    result = training_free.encode(dict(path="dummy", sha256="abc", duration=2.033), tmp_path, Client(), IndexConfig(), [])
    assert calls == [1.9]
    assert result["actual_duration"] == 1.9
