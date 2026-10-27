"""CPU-only synchronized overview/details and normal-motion storyboards.

All proposals come from pixels, never anomaly labels or annotation boxes.
"""
from pathlib import Path
import math
import cv2
import numpy as np

DETAIL_RULE=' Labeled OVERVIEW and DETAIL panels, if present, are synchronized views of the same instant. They repeat the same people, not additional actors. Compare motion across chronological frames, not across panels. The overview preserves scene context.'
NORMAL_RULE=' A NORMAL reference image labeled with multiple times is a chronological storyboard from a separate verified-normal recording. Compare the visible action pattern as well as the scene. Reference storyboard tiles are never frames of the current event.'


def overlap(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    intersection=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return intersection/(aw*ah+bw*bh-intersection)


def detect_tiled(frame,detector):
    h,w=frame.shape[:2];boxes=[];scores=[]
    # Full view plus overlapping 2x2 tiles helps small people without a GPU detector.
    tile_w,tile_h=math.ceil(w*.6),math.ceil(h*.6)
    regions=[(0,0,w,h)]+[(x,y,x+tile_w,y+tile_h) for y in (0,h-tile_h) for x in (0,w-tile_w)]
    for x,y,right,bottom in regions:
        found,confidence=detector.detectMultiScale(frame[y:bottom,x:right])
        for box,score in zip(found,confidence):
            bx,by,bw,bh=map(int,box);boxes.append([bx+x,by+y,bw,bh]);scores.append(float(score))
    kept=[]
    for i in sorted(range(len(boxes)),key=lambda i:(-scores[i],i)):
        if all(overlap(boxes[i],boxes[j])<.5 for j in kept):kept.append(i)
    return [boxes[i] for i in kept],[scores[i] for i in kept]


def rectangles(detections,width,height):
    if sum(bool(x) for x in detections)<2:return []
    boxes=[b for frame in detections for b in frame]
    for b in boxes:
        if len(b)!=4 or not np.isfinite(b).all() or min(b[2:])<=0 or min(b[:2])<0 or b[0]+b[2]>width or b[1]+b[3]>height:
            raise ValueError('Invalid person detection')
    centers=np.array([b[0]+b[2]/2 for b in boxes]);middle=float(np.median(centers))
    groups=([boxes] if centers.max()-centers.min()<.1*width else
            [[b for b in boxes if b[0]+b[2]/2<=middle],[b for b in boxes if b[0]+b[2]/2>middle]])
    result=[]
    for group in groups:
        if not group:continue
        left=min(b[0] for b in group);top=min(b[1] for b in group)
        right=max(b[0]+b[2] for b in group);bottom=max(b[1]+b[3] for b in group)
        cx,cy=(left+right)/2,(top+bottom)/2
        rw=min(width,max((right-left)*1.4,width*.12));rh=min(height,max((bottom-top)*1.4,height*.25))
        x=max(0.,min(cx-rw/2,width-rw));y=max(0.,min(cy-rh/2,height-rh))
        result.append([int(math.floor(x)),int(math.floor(y)),int(math.ceil(x+rw)),int(math.ceil(y+rh))])
    return result


def panel(canvas,image,rect,title):
    x1,y1,x2,y2=rect;h,w=image.shape[:2];space_w=x2-x1;space_h=y2-y1-22
    if min(space_w,space_h)<=0:raise ValueError('Panel too small')
    scale=min(space_w/w,space_h/h);nw=max(1,int(round(w*scale)));nh=max(1,int(round(h*scale)))
    small=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    x=x1+(space_w-nw)//2;y=y1+22+(space_h-nh)//2;canvas[y:y+nh,x:x+nw]=small
    cv2.putText(canvas,title,(x1+6,y1+16),cv2.FONT_HERSHEY_SIMPLEX,.44,(240,240,240),1,cv2.LINE_AA)


def compose_detail(frame,regions):
    h,w=frame.shape[:2]
    if not regions:return frame.copy()
    canvas=np.full_like(frame,24);split=w//2
    panel(canvas,frame,[0,0,split,h],'OVERVIEW - same instant')
    for i,(x1,y1,x2,y2) in enumerate(regions):
        top=i*h//len(regions);bottom=(i+1)*h//len(regions)
        panel(canvas,frame[y1:y2,x1:x2],[split,top,w,bottom],f'DETAIL {i+1} - same instant')
    return canvas


def prepare_detail(event,folder,detector,cache):
    frames=[];detections=[];scores=[]
    for path in event['frame_paths']:
        frame=cv2.imread(path)
        if frame is None:raise ValueError('Missing current frame')
        frames.append(frame)
        if path not in cache:cache[path]=detect_tiled(frame,detector)
        found,confidence=cache[path];detections.append(found);scores.append(confidence)
    h,w=frames[0].shape[:2]
    if any(f.shape!=frames[0].shape for f in frames):raise ValueError('Inconsistent frame canvas')
    rects=rectangles(detections,w,h);paths=[];folder=Path(folder)
    if rects:
        folder.mkdir(parents=True,exist_ok=True)
        for i,frame in enumerate(frames):
            target=folder/f'{i:02}.png'
            if not cv2.imwrite(str(target),compose_detail(frame,rects)):raise OSError('Cannot save detail view')
            paths.append(str(target.resolve()))
    else:paths=list(event['frame_paths'])
    return paths,{'rectangles':rects,'detections':detections,'scores':scores,'canvas':[w,h],
                 'fallback':None if rects else 'fewer_than_two_detection_frames','source_frames':event['frame_paths']}


def normal_storyboard(ref,store,config,development_ids,folder):
    if ref['video_id'] not in development_ids:raise ValueError('Reference not from development')
    video=store.get('videos',ref['video_id'])
    namespace=config.namespace(video['fingerprint'],video['captions_hash'])
    window=next((w for w in store.windows(namespace) if ref['path'] in w['frame_paths']),None)
    if window is None:raise ValueError('Cannot establish reference window')
    slots=np.linspace(0,len(window['frame_paths'])-1,4).astype(int).tolist()
    if len(set(slots))<4:
        return dict(ref),{'video_id':ref['video_id'],'window_id':window['id'],
                          'fallback':'fewer_than_four_normal_frames','paths':window['frame_paths']}
    images=[cv2.imread(window['frame_paths'][i]) for i in slots]
    if any(image is None for image in images):raise ValueError('Missing normal reference frame')
    h,w=images[0].shape[:2];canvas=np.full_like(images[0],24)
    for j,(slot,image) in enumerate(zip(slots,images)):
        x=j%2;y=j//2
        panel(canvas,image,[x*w//2,y*h//2,(x+1)*w//2,(y+1)*h//2],f'NORMAL t={window["timestamps"][slot]:.2f}s')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);target=folder/f'{ref["video_id"]}-{window["id"]}.png'
    if not cv2.imwrite(str(target),canvas):raise OSError('Cannot save normal storyboard')
    return {**ref,'path':str(target.resolve())},{'video_id':ref['video_id'],'window_id':window['id'],'slots':slots,
        'timestamps':[window['timestamps'][i] for i in slots],'paths':[window['frame_paths'][i] for i in slots],'canvas':[w,h]}
