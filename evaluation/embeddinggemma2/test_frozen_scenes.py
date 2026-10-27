from pathlib import Path
from . import evaluate_frozen_scenes as runner


def test_payload_does_not_depend_on_validation_annotations(monkeypatch):
    monkeypatch.setattr(runner.n,'sample_frames',lambda path,start,end,duration,folder:
        [{'path':str(folder/'frame.jpg'),'timestamp':start}, {'path':str(folder/'next.jpg'),'timestamp':end-.01}])
    monkeypatch.setattr(runner,'reference_frames',lambda service,refs:refs)
    row={'id':'new-scene','path':'video.mp4','duration':15.,'events':[]}
    windows=[{'id':0,'t_start':0.,'t_end':6.},{'id':1,'t_start':12.,'t_end':15.}]
    evidence={0:{'image':[{'references':[{'key':'a'},{'key':'b'}]}]},1:{'video':[{'references':[{'key':'c'}]}]}}
    baseline=runner.build_payload(None,row,windows,evidence,Path('frames'))
    row['events']=[{'t_start':13.,'t_end':14.}]
    changed=runner.build_payload(None,row,windows,evidence,Path('frames'))
    assert baseline==changed
    payload,intervals=baseline
    assert len(payload)==2
    assert payload[0]['reference_frames']==[{'key':'a'},{'key':'b'}]
    assert payload[1]['reference_frames']==[{'key':'c'}]
    assert intervals['new-scene-1']=={'video_id':'new-scene','t_start':12.,'t_end':15.}
    assert all(p['scene_context']==runner.CONTEXT+' '+runner.RULE for p in payload)
    assert all(p['generate_caption'] is False for p in payload)
