import numpy as np
import pytest

from .benchmark_vadclip import blocks, calibration_threshold, threshold_events


def test_tail_timestamps_and_merge():
    bounds = blocks(35, 30.)
    assert bounds[-1] == (32, 35, 32/30, 35/30)
    pred = threshold_events(bounds, [.8, .9, .1], .5, 35/30)
    assert len(pred) == 1
    assert pred[0]['t_start'] == 0
    assert pred[0]['t_end'] == 32/30


def test_no_silent_nonfinite_or_missing_scores():
    for values in [[np.nan], []]:
        with pytest.raises(ValueError):
            threshold_events(blocks(16, 30), values, .5, 16/30)


def test_calibration_uses_higher_quantile_and_refuses_empty():
    assert calibration_threshold([.1, .2, .3]) == .3
    with pytest.raises(ValueError):
        calibration_threshold([])


def test_does_not_bridge_negative_block_or_extend_tail():
    pred = threshold_events(blocks(35, 30), [.8, .1, .8], .5, 35/30)
    assert len(pred) == 2
    assert pred[-1]['t_end'] == 35/30
