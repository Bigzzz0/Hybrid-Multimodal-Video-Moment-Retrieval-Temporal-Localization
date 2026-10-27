"""VideoPatchCore core functions on UBnormal, with declared resource adaptations.

Official RN101 layer2/layer3, spatial/temporal/high-level functions and greedy
coreset. SSDLite CPU replaces YOLOv5l; 2 FPS / four frames / one-second stride;
512-patch cap per scene/channel. Not a reproduction of published benchmark AUC.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from app.gemma2.store import file_hash
from .open_anomaly_eval import audit
from .person_region import SSDLitePersonDetector
from .research_screening import cached_features, timeline

def source_windows(database, rows):
    import sqlite3
    db=sqlite3.connect(f"file:{Path(database).resolve().as_posix()}?mode=ro",uri=True)
    result={}
    try:
        for row in rows:
            namespace=db.execute('select namespace from active where video_id=?',(row['id'],)).fetchone()[0]
            unique={}
            for data, in db.execute('select data from windows where namespace=? order by id',(namespace,)):
                w=json.loads(data)
                unique.update({round(t,6):p for t,p in zip(w['timestamps'],w['frame_paths'])})
            result[row['id']]=sorted(unique.items())
    finally:db.close()
    return result

def padded_box(box,width,height):
    x,y,w,h=map(float,box)
    center_x,center_y=x+w/2,y+h/2
    side=max(w*2,h*1.4)
    return [max(0,int(center_x-side/2)),max(0,int(center_y-side/2)),
            min(width,int(center_x+side/2)),min(height,int(center_y+side/2))]

def preprocess(frames,device):
    import torch
    # Deliberately match upstream dataset np_load_frame (BGR, -1..1), not CLIP RGB transform.
    array=np.stack([cv2.resize(f,(224,224)).transpose(2,0,1) for f in frames]).astype(np.float32)/127.5-1
    return torch.from_numpy(array).to(device)

def score_vpc(rows,root):
    """Build per-scene dev-only banks, no validation-driven feature normalization."""
    import torch
    from scipy.spatial.distance import cdist
    sys.path.insert(0,str(Path('scratch/research-baselines/VideoPatchCore/Memorization').resolve()))
    from functions.memory_func import get_coreset
    refs=[r for r in rows if r['split']=='development']
    validation=[r for r in rows if r['split']=='validation']
    banks={}; all_scores={}; stats={}
    started=time.perf_counter()
    for scene in sorted({r['scene_id'] for r in rows}):
        raw={key:[] for key in ('spatial','temporal','global')}
        for row in refs:
            if row['scene_id']!=scene:continue
            with np.load(root/(row['id']+'.npz'),allow_pickle=False) as f:
                for key in raw:raw[key].append(f[key])
        for key,parts in raw.items():
            features=np.concatenate(parts)
            # Fixed, label-free memory budget; deterministic greedy farthest-first.
            indices=get_coreset(torch.from_numpy(features).float(),l=min(512,len(features)),device='cpu').numpy()
            banks[(scene,key)]=features[indices]
        def raw_score(path):
            with np.load(path,allow_pickle=False) as f:
                offsets=f['offsets']; result=[]
                for i in range(len(f['intervals'])):
                    scores=[]
                    for j,key in enumerate(('spatial','temporal','global')):
                        a,b=int(offsets[i,j,0]),int(offsets[i,j,1])
                        patches=f[key][a:b]
                        scores.append(float(np.max(np.min(cdist(patches,banks[(scene,key)]),axis=1)))**2 if len(patches) else 0.)
                    result.append(scores)
                return f['intervals'].copy(),np.asarray(result)
        dev_values=np.concatenate([raw_score(root/(r['id']+'.npz'))[1] for r in refs if r['scene_id']==scene])
        # Only development calibrates scale; avoid per-test-video normalization.
        center=np.median(dev_values,axis=0)
        scale=np.maximum(np.quantile(dev_values,.9,axis=0)-np.quantile(dev_values,.1,axis=0),1e-6)
        stats[scene]={'center':center.tolist(),'scale':scale.tolist(),
                      'bank_sizes':{k:len(banks[(scene,k)]) for k in raw}}
        for row in validation:
            if row['scene_id']!=scene:continue
            intervals,values=raw_score(root/(row['id']+'.npz'))
            standardized=(values-center)/scale
            variants={'vpc_spatial':standardized[:,0],'vpc_temporal':standardized[:,1],
                'vpc_global':standardized[:,2],
                'vpc_three_memory':.5*(.5*standardized[:,0]+.5*standardized[:,1])+.5*standardized[:,2]}
            for key,signal in variants.items():
                all_scores.setdefault(key,{})[row['id']]=timeline(signal,intervals,row['duration'])
    (root/'memory-stats.json').write_text(json.dumps({'scenes':stats,'seconds':time.perf_counter()-started},indent=2),encoding='utf-8')
    return all_scores

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    rows=json.loads(manifest_path.read_text(encoding='utf-8'))['videos'];audit(rows)
    source=Path('scratch/research-baselines/VideoPatchCore/Memorization').resolve()
    sys.path.insert(0,str(source));sys.path.insert(0,str(Path('scratch/research-baselines/CLIP').resolve()))
    import clip
    import torch
    from functions.feature_func import MakeLocallyAwareFeature,MakeGloballyAwareFeature,MakeHighlevelFeature,MakeSpatialFeature,MakeTemporalFeature
    protocol={'method':'VideoPatchCore-adapted','vpc_revision':'eefdc8d7a84f8eb06a561e3c42ab26a1b944f9b6',
        'clip_revision':'d05afc436d78f1c48dc0dbf8e5980a9d471f35f6','manifest_sha256':file_hash(manifest_path),
        'code_sha256':file_hash(Path(__file__)), 'weights_sha256':file_hash(Path('scratch/research-baselines/weights/RN101.pt')),
        'source_hashes':{str(p):file_hash(p) for p in [source/'functions/feature_func.py',source/'functions/pooling.py',source/'functions/memory_func.py']},
        'adaptations':['UBnormal instead of original datasets','SSDLite CPU COCO person instead of YOLOv5l',
            '2FPS four frames in two seconds, stride one second; cached JPEGs','FP32 CNN on RTX5070',
            'Greedy memory cap512 per channel per scene; 64 local channels',
            'Dev-only robust scaling instead of normalization on each test video',
            'Fixed fusion and sigma0/1; no upstream GT-driven best-AUC search'],
        'test_opened':False,'production_activation':False}
    p=root/'protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=protocol:raise ValueError('Fresh output required')
    p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Use shared broker to release all worker GPU models before this isolated run.
    import requests
    requests.post('http://127.0.0.1:8012/unload',timeout=60).raise_for_status()
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    started=time.perf_counter()
    model,_=clip.load('RN101',device=device,jit=False,download_root='scratch/research-baselines/weights')
    model=model.float().eval()
    maps=[]
    hooks=[model.visual.layer2[-1].register_forward_hook(lambda m,a,o:maps.append(o.float())),
           model.visual.layer3[-1].register_forward_hook(lambda m,a,o:maps.append(o.float()))]
    detector=SSDLitePersonDetector(Path('scratch/event-retrieval/person-region-ssdlite-dev/model-cache'))
    load_seconds=time.perf_counter()-started
    frames=source_windows('scratch/event-retrieval/open-anomaly-multiscene-v2/store/events.sqlite3',rows)
    records=[]
    import psutil,threading,subprocess
    stop=threading.Event();resources={'rss_peak_bytes':0,'whole_gpu_sampled_peak_bytes':0}
    def sample():
        while not stop.is_set():
            resources['rss_peak_bytes']=max(resources['rss_peak_bytes'],psutil.Process().memory_info().rss)
            try:
                used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=3).strip().splitlines()[0])
                resources['whole_gpu_sampled_peak_bytes']=max(resources['whole_gpu_sampled_peak_bytes'],used*1024**2)
            except Exception:pass
            stop.wait(.5)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            for row in rows:
                cache=root/(row['id']+'.npz')
                if cache.exists():
                    print('CACHED',row['id'],flush=True);continue
                before=time.perf_counter(); values={'spatial':[],'temporal':[],'global':[]};offsets=[];intervals=[];counts=[]
                slots=frames[row['id']];times=np.array([t for t,p in slots])
                canvas=[cv2.imread(p) for t,p in slots]
                if any(f is None for f in canvas):raise ValueError('Missing cached frame')
                for start in np.arange(0,row['duration'],1.):
                    indices=[int(np.argmin(abs(times-t))) for t in start+np.arange(4)*.5]
                    batch=[canvas[i] for i in indices]; height,width=batch[0].shape[:2]
                    detected=detector.detectMultiScale(batch[len(batch)//2])[0]
                    boxes=[padded_box(box,width,height) for box in detected]
                    boxes=[b for b in boxes if b[2]>b[0] and b[3]>b[1]]
                    maps.clear(); model.visual(preprocess(batch,device))
                    high=MakeHighlevelFeature(MakeGloballyAwareFeature(maps)).cpu().numpy()
                    spatial=[];temporal=[]
                    # One object at a time bounds GPU memory; same box across four frames.
                    for x1,y1,x2,y2 in boxes:
                        crops=[f[y1:y2,x1:x2] for f in batch]
                        maps.clear();model.visual(preprocess(crops,device))
                        local=MakeLocallyAwareFeature(maps,pool_size=64,len_frame=4)
                        spatial.append(MakeSpatialFeature(local,kernel_size=(1,4,4)).cpu().numpy())
                        temporal.append(MakeTemporalFeature(local,mode='train' if row['split']=='development' else 'test').cpu().numpy())
                    current={'spatial':np.concatenate(spatial) if spatial else np.empty((0,64)),
                             'temporal':np.concatenate(temporal) if temporal else np.empty((0,64)), 'global':high}
                    offset=[]
                    for key in values:
                        begin=sum(len(x) for x in values[key]);values[key].append(current[key]);offset.append([begin,begin+len(current[key])])
                    offsets.append(offset);intervals.append([start,min(start+2,row['duration'])]);counts.append(len(boxes))
                if device=='cuda':torch.cuda.synchronize()
                feature={key:np.concatenate(parts).astype(np.float32) for key,parts in values.items()}
                np.savez_compressed(cache,**feature,offsets=offsets,intervals=intervals,object_counts=counts)
                record={'id':row['id'],'seconds':time.perf_counter()-before,'blocks':len(intervals),'object_counts':counts,
                        'unique_frames':len(slots),'sha256':file_hash(cache)}
                records.append(record)
                (root/'partial.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                print('VPC',row['id'],len(intervals),'blocks',round(record['seconds'],2),'s',flush=True)
    finally:
        stop.set();thread.join(timeout=4)
        for hook in hooks:hook.remove()
        resources['torch_allocated_peak_bytes']=torch.cuda.max_memory_allocated() if device=='cuda' else None
        resources['torch_reserved_peak_bytes']=torch.cuda.max_memory_reserved() if device=='cuda' else None
        report={'protocol':protocol,'load_seconds':load_seconds,'records':records,'resources':resources,
                'detector':detector.metadata,'feature_seconds':sum(r['seconds'] for r in records),
                'status':'complete' if len(list(root.glob('*.npz')))==len(rows) else 'partial'}
        (root/'resources.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
