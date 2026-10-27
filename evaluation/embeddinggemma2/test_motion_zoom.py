import numpy as np
import pytest

from .motion_zoom import choose_motion_span


def test_motion_span_tracks_a_burst_without_label_input():
    stamps = np.arange(0., 6., .125); energies = np.zeros(len(stamps))
    energies[(stamps >= 3.5) & (stamps < 4.)] = 20.
    left, right = choose_motion_span(stamps, energies, 0., 6.)
    burst = stamps[energies > 0]
    assert right-left == 2. and left <= burst.min() and right >= burst.max()


def test_quiet_frames_do_not_trigger_a_spurious_motion_retry():
    assert choose_motion_span([0., 1.], [0., 0.], 0., 2.) is None


def test_short_tail_and_edge_bursts_stay_inside_source_interval():
    assert choose_motion_span([9., 9.25, 9.5], [0., 20., 2.], 9., 9.75) == [9., 9.75]
    left, right = choose_motion_span(np.arange(0., 6., .125), np.full(48, 10.), 0., 6.)
    assert 0. <= left < right <= 6.


def test_invalid_trace_is_not_silently_resampled():
    for stamps, values in [([0., 0.], [1., 2.]), ([0., 1.], [float("nan"), 2.]), ([0., 1.], [2.])]:
        with pytest.raises(ValueError):
            choose_motion_span(stamps, values, 0., 6.)


def test_retry_runner_cannot_read_test_or_mislabel_validation(tmp_path):
    from .compare_motion_zoom import run
    with pytest.raises(ValueError, match="approved control"):
        run(dict(complete=True, sequences=[17, 18, 19, 20]), {}, tmp_path)
    with pytest.raises(ValueError, match="Mismatched"):
        run(dict(complete=True, sequences=[13, 14, 15, 16], split="development"), {}, tmp_path)
