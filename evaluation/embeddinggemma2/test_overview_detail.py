import cv2
import numpy as np
import pytest
from types import SimpleNamespace
from .overview_detail import rectangles,compose_detail,detect_tiled,normal_storyboard
from .compare_overview_detail import qualifies


def test_fixed_regions_cover_detections_and_do_not_escape_canvas():
    boxes=[[[20,50,35,120],[450,40,40,110]],[[40,40,35,130],[430,45,40,110]]]
    rects=rectangles(boxes,640,360)
    assert len(rects)==2
    for frames in boxes:
        for x,y,w,h in frames:
            assert any(a<=x and b<=y and c>=x+w and d>=y+h for a,b,c,d in rects)
    assert all(0<=a<c<=640 and 0<=b<d<=360 for a,b,c,d in rects)


def test_fallback_and_same_canvas():
    image=np.full((360,640,3),120,dtype=np.uint8)
    assert rectangles([[],[]],640,360)==[]
    assert np.array_equal(compose_detail(image,[]),image)
    result=compose_detail(image,[[0,0,100,200],[400,30,600,300]])
    assert result.shape==image.shape
    assert np.any(result[:,:320]==120)


def test_invalid_boxes_fail():
    with pytest.raises(ValueError):rectangles([[[600,0,80,100]]]*2,640,360)


def test_tiling_uses_pixels_and_clips_regions():
    class Detector:
        def detectMultiScale(self,frame):return [[0,0,10,20]],[.8]
    boxes,scores=detect_tiled(np.zeros((360,640,3),np.uint8),Detector())
    assert len(boxes)==len(scores)
    assert all(0<=x<x+w<=640 and 0<=y<y+h<=360 for x,y,w,h in boxes)


def test_only_development_can_be_normal_storyboard(tmp_path):
    with pytest.raises(ValueError,match='development'):
        normal_storyboard({'video_id':'held-out'},None,None,set(),tmp_path)


def test_short_normal_reference_falls_back_without_fabricating_motion(tmp_path):
    store=SimpleNamespace(get=lambda *args:{'fingerprint':'f','captions_hash':'c'},
                          windows=lambda ns:[{'id':0,'frame_paths':['single.jpg']}])
    config=SimpleNamespace(namespace=lambda *args:'ns')
    ref={'video_id':'dev','path':'single.jpg','timestamp':1.}
    result,log=normal_storyboard(ref,store,config,{'dev'},tmp_path)
    assert result==ref and log['fallback']=='fewer_than_four_normal_frames'


def test_promotion_requires_preserving_existing_events_and_no_extra_false_alerts():
    base={'tp':3,'fp':2,'uncertain_units':0};better={'tp':4,'fp':2,'uncertain_units':0}
    assert qualifies(base,better,{('a',0)},{('a',0),('b',0)})
    assert not qualifies(base,better,{('a',0)},{('b',0),('c',0)})
    assert not qualifies(base,{**better,'fp':3},{('a',0)},{('a',0),('b',0)})
    assert not qualifies(base,{**better,'uncertain_units':1},{('a',0)},{('a',0),('b',0)})


def test_storyboard_keeps_normal_recording_time_order_and_original_canvas(tmp_path):
    paths=[]
    for i in range(6):
        path=tmp_path/f'normal-{i}.png'
        assert cv2.imwrite(str(path),np.full((360,640,3),20+i*35,np.uint8))
        paths.append(str(path))
    window={'id':7,'frame_paths':paths,'timestamps':[0.,1.,2.,3.,4.,5.]}
    store=SimpleNamespace(get=lambda *args:{'fingerprint':'f','captions_hash':'c'},windows=lambda ns:[window])
    config=SimpleNamespace(namespace=lambda *args:'ns')
    ref={'video_id':'dev-normal','path':paths[2],'timestamp':2.}
    result,log=normal_storyboard(ref,store,config,{'dev-normal'},tmp_path/'storyboards')
    image=cv2.imread(result['path'])
    assert image.shape==(360,640,3)
    assert result['video_id']=='dev-normal'
    assert log['slots']==[0,1,3,5] and log['timestamps']==[0.,1.,3.,5.]
    assert log['paths']==[paths[i] for i in (0,1,3,5)]
    assert [image[y,x,0] for y,x in ((100,150),(100,470),(280,150),(280,470))]==[20,55,125,195]
