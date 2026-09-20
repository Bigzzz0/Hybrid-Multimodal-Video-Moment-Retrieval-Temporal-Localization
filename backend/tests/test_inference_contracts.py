import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference.contracts import CaptionRequest, GroundResponse, VerifyResponse


def test_worker_contracts_are_path_based_and_validate():
    request = CaptionRequest(frame_paths=[r"C:\frames\one.jpg"], timestamps=[1.5])
    assert request.frame_paths[0].endswith("one.jpg")
    assert request.timestamps == [1.5]

    verify = VerifyResponse(
        event_present=True,
        start_frame_index=0,
        end_frame_index=1,
        confidence=0.9,
        reason="visible action",
    )
    assert verify.confidence == 0.9
    assert "base64" not in request.model_dump_json().lower()


def test_ground_response_defaults_to_no_detection_not_error():
    response = GroundResponse()
    assert response.tracks == []
    assert response.status == "unavailable"
