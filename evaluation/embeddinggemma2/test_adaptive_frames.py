import numpy as np
import pytest
from .research_adaptive_frames import select_indices
from .research_evidence_frames import evidence_payload, EVIDENCE


def test_static_coverage():
    ids=select_indices(np.arange(48)/4,np.zeros(48))
    assert len(set(ids))==8
    assert ids==sorted(ids)
    assert ids[0]<6 and ids[-1]>=42


def test_motion_increases_local_allocation_without_losing_full_coverage():
    motion=np.zeros(48);motion[30:36]=10
    ids=select_indices(np.arange(48)/4,motion)
    assert sum(30<=i<36 for i in ids)>=3
    assert ids[0]<12 and ids[-1]>=36
    assert len(set(ids))==8


def test_short_tail_with_eight_candidates_keeps_all():
    assert select_indices(np.linspace(12,13,8,endpoint=False),np.arange(8))==list(range(8))


@pytest.mark.parametrize('motion',[[float('nan')]*8,[-1]*8,[1]*7])
def test_bad_scores_fail(motion):
    with pytest.raises(ValueError):select_indices(np.arange(8),motion)


def test_evidence_changes_only_context_and_does_not_mutate_source():
    source=[{'id':'x','scene_context':'routine','frame_paths':['one'],'timestamps':[1.],
             'reference_frames':[{'path':'reference'}]}]
    out=evidence_payload(source)
    assert source[0]['scene_context']=='routine'
    assert out[0]['scene_context']=='routine'+EVIDENCE
    assert {k:v for k,v in out[0].items() if k!='scene_context'}=={k:v for k,v in source[0].items() if k!='scene_context'}
    out[0]['frame_paths'].append('two')
    assert source[0]['frame_paths']==['one']
