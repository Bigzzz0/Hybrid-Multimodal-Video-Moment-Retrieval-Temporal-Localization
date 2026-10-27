from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.event_retrieval.metrics import event_metrics, gates, retrieval
from evaluation.event_retrieval.select_validation import select
from evaluation.event_retrieval.dataset import validate

def test_duplicate_detections_are_false_positives():
    event = {"kind": "falling", "video_id": "v", "t_start": 1, "t_end": 4}
    result = event_metrics([event, event], [event], 0)
    assert result["tp"] == 1 and result["fp"] == 1 and result["precision"] == .5
    assert result["false_alerts_per_hour"] is None

def test_no_match_queries_and_th_en_reported_separately():
    results = retrieval([{"language": "th", "truth": [[1, 4]], "moments": [{"t_start": 1, "t_end": 4}]},
                         {"language": "en", "truth": [], "moments": [{"t_start": 8, "t_end": 10}]}])
    assert results["th"]["recall@1"] == 1
    assert results["en"]["no_match_false_positive_rate"] == 1

def test_missing_measurements_cannot_pass_gates():
    assert not gates({}, {}, {})["passed"]

def test_test_split_cannot_select_thresholds():
    with pytest.raises(ValueError, match="validation"):
        select([{"split": "test"}])

def test_empty_manifest_reports_all_missing_data():
    result = validate({"videos": []})
    assert not result["ready"] and result["videos"] == 0
    assert "Need 60.0 more continuous normal minutes in test" in result["errors"]
