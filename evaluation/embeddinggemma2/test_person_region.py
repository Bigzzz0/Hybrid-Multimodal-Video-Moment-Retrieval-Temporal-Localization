import cv2
import numpy as np
import pytest

from .person_region import region_for_detections, prepare_region, filter_person_boxes


def test_union_keeps_every_detected_person_and_stays_fixed_over_time():
    detections = [[[100, 100, 60, 180]], [[250, 110, 70, 170], [120, 100, 60, 180]]]
    rect, fallback = region_for_detections(detections, 720, 480)
    assert fallback is None
    for boxes in detections:
        for x, y, w, h in boxes:
            assert rect[0] <= x and rect[1] <= y and rect[2] >= x+w and rect[3] >= y+h
    assert rect[2]-rect[0] >= 360 and rect[3]-rect[1] >= 240


def test_missing_or_wide_detections_use_whole_frame():
    assert region_for_detections([[], []], 720, 480)[0] == [0, 0, 720, 480]
    assert region_for_detections([[[0, 0, 700, 460]]]*2, 720, 480)[1] == "region_too_wide"
    assert region_for_detections([[[100, 100, 50, 200]]], 720, 480)[1] == "insufficient_detection_frames"


def test_detector_box_outside_canvas_is_rejected():
    with pytest.raises(ValueError, match="detector box"):
        region_for_detections([[[700, 0, 60, 180]]]*2, 720, 480)


def test_ssdlite_keeps_only_people_above_fixed_threshold_and_clips_to_canvas():
    prediction = dict(boxes=np.array([[-1., 1.5, 60., 90.], [0., 0., 20., 20.], [0., 0., 20., 20.]]),
                      labels=[1, 2, 1], scores=[.8, .9, .2])
    boxes, scores = filter_person_boxes(prediction, 50, 80)
    assert boxes == [[0, 1, 50, 79]] and scores == [.8]


def test_crop_preserves_frame_order_count_and_original_canvas(tmp_path):
    paths = []
    for index in (20, 80):
        path = tmp_path/f"input-{index}.png"
        cv2.imwrite(str(path), np.full((480, 720, 3), index, dtype=np.uint8)); paths.append(str(path))
    class Detector:
        def detectMultiScale(self, frame, **kwargs):
            return [[200, 120, 80, 180]], [1.]
    cropped, provenance = prepare_region(paths, tmp_path/"crop", Detector())
    assert provenance["fallback"] is None and len(cropped) == 2
    for index, path in enumerate(cropped):
        frame = cv2.imread(path)
        assert frame.shape == (480, 720, 3) and np.all(frame == (20, 80)[index])
    with pytest.raises(ValueError, match="Unreadable"):
        prepare_region([tmp_path/"absent.png"], tmp_path/"missing", Detector())
