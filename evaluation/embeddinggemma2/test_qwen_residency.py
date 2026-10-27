from types import SimpleNamespace
import pytest
from .research_qwen_residency import release_policy


def test_retention_releases_once_and_restores_hook():
    calls=[];original=lambda:calls.append('release');module=SimpleNamespace(release_qwen=original)
    with release_policy(module,True) as counts:
        module.release_qwen();module.release_qwen()
        assert calls==[]
    assert calls==['release'] and counts['deferred_releases']==2
    assert module.release_qwen is original


def test_exception_also_releases_and_restores_hook():
    calls=[];original=lambda:calls.append('release');module=SimpleNamespace(release_qwen=original)
    with pytest.raises(RuntimeError):
        with release_policy(module,True):
            module.release_qwen();raise RuntimeError('cancelled or failure')
    assert calls==['release'] and module.release_qwen is original


def test_baseline_keeps_every_release():
    calls=[];module=SimpleNamespace(release_qwen=lambda:calls.append('release'))
    with release_policy(module,False) as counts:
        module.release_qwen();module.release_qwen()
    assert len(calls)==2 and counts['deferred_releases']==0
