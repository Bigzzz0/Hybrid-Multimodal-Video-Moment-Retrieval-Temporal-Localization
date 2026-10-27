"""Matched-budget temporal sampling experiment; selection never receives labels."""
import argparse
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash
from .compare_matched_gemma_gate import predictions
from .open_anomaly_eval import audit, metrics


def select_indices(times, motion, count=8):
    times = np.asarray(times, dtype=float)
    motion = np.asarray(motion, dtype=float)
    if len(times) < count or motion.shape != times.shape:
        raise ValueError('Insufficient/misaligned candidates')
    if not np.isfinite(times).all() or not np.isfinite(motion).all() or np.any(motion < 0):
        raise ValueError('Non-finite or negative sampling input')
    if np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must increase')
    # Half uniform mass preserves coverage; half follows observed image changes.
    total = motion.sum()
    weights = np.ones(len(times))/len(times)
    if total > 1e-12:
        weights = .5*weights + .5*motion/total
    cdf = np.cumsum(weights)
    chosen = []
    for q in (np.arange(count)+.5)/count:
        candidate = min(int(np.searchsorted(cdf, q)), len(times)-1)
        available = [i for i in range(len(times)) if i not in chosen]
        chosen.append(min(available, key=lambda i: (abs(i-candidate), i)))
    return sorted(chosen)


def adaptive(event, interval, video, folder):
    start, end = interval['t_start'], interval['t_end']
    cap = cv2.VideoCapture(video['path'])
    if not cap.isOpened(): raise ValueError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_count = max(8, int(np.ceil((end-start)*4)))
    ids = sorted(set(int(round(t*fps)) for t in np.linspace(start,end,sample_count,endpoint=False) if round(t*fps)/fps < end))
    images, times, motion = [], [], []
    previous = None
    try:
        wanted=set(ids)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ids[0])
        for index in range(ids[0],ids[-1]+1):
            if index not in wanted:
                if not cap.grab(): raise ValueError(f'Missing intervening frame {index}')
                continue
            ok, image = cap.read()
            if not ok: raise ValueError(f'Missing frame {index}')
            gray = cv2.cvtColor(cv2.resize(image, (160,90)), cv2.COLOR_BGR2GRAY).astype(float)
            motion.append(0. if previous is None else float(np.mean(np.abs(gray-previous))))
            previous = gray
            images.append(image); times.append(index/fps)
    finally:
        cap.release()
    picked = select_indices(times, motion)
    target = folder/event['id']; target.mkdir(parents=True, exist_ok=True)
    frames = [(t,p) for t,p in zip(event['timestamps'],event['frame_paths']) if not start <= t < end]
    expected = sum(start <= t < end for t in event['timestamps'])
    if expected != 8: raise ValueError('Baseline must have eight event frames')
    for i in picked:
        path = target/f'{ids[i]}.jpg'
        if not cv2.imwrite(str(path), images[i]): raise OSError('JPEG write failed')
        frames.append((times[i],str(path.resolve())))
    frames.sort()
    result = copy.deepcopy(event)
    result['timestamps'] = [t for t,p in frames]
    result['frame_paths'] = [p for t,p in frames]
    if len(result['frame_paths']) != len(event['frame_paths']): raise ValueError('Frame budget changed')
    if {k:v for k,v in result.items() if k not in ('timestamps','frame_paths')} != {k:v for k,v in event.items() if k not in ('timestamps','frame_paths')}:
        raise ValueError('Non-sampling payload changed')
    return result, {'times':times,'motion':motion,'selected_indices':picked}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    source=Path('scratch/event-retrieval/matched-gemma-gate-20261015-v1')
    original=json.loads((source/'requests.json').read_text())
    manifest_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    manifest=json.loads(manifest_path.read_text()); audit(manifest['videos'])
    rows=[v for v in manifest['videos'] if v['split']=='validation']; lookup={v['id']:v for v in rows}
    jobs=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
    if any(j['status'] in ('running','queued','cancelling') for j in jobs): raise ValueError('Demo job active')
    baseline=original['gemma_then_qwen']; intervals=original['intervals']
    protocol={'version':'adaptive-frames-v2','selection':'4FPS (at least eight candidates for short tails) grayscale mean absolute difference; 50% uniform + 50% motion CDF midpoint quantiles; eight event frames; preserve context, references, prompt and unit boundaries',
        'seed':None,'threshold_tuning':False,'labels_used_for_sampling':False,'test_opened':False,
        'source_sha256':file_hash(source/'requests.json'),'manifest_sha256':file_hash(manifest_path),
        'script_sha256':file_hash(Path(__file__)),'order':'interleaved paired units, alternating arm first',
        'timing_scope':'Qwen batch wall includes model lifecycle; candidate decoding separately; cached Gemma gates and normal references',
        'limitations':'Previously inspected synthetic validation, not independent test; motion is only sampling, not a gate'}
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol: raise ValueError('Use fresh output')
    pp.write_text(json.dumps(protocol,indent=2))
    started=time.perf_counter(); variants=[]; logs={}
    for event in baseline:
        interval=intervals[event['id']]
        variant, log=adaptive(event,interval,lookup[interval['video_id']],root/'frames')
        variants.append(variant);logs[event['id']]=log
    prep_seconds=time.perf_counter()-started
    hashes={p:file_hash(Path(p)) for e in baseline+variants for p in e['frame_paths']+[r['path'] for r in e['reference_frames']]}
    (root/'requests.json').write_text(json.dumps({'uniform':baseline,'adaptive':variants,'sampling':logs,'intervals':intervals,'frame_hashes':hashes,'preparation_seconds':prep_seconds},indent=2))
    print('PREPARED',len(variants),'paired units; decode seconds',prep_seconds,flush=True)
    statefile=root/'partial.json'
    arms=json.loads(statefile.read_text()) if statefile.exists() else {k:{'replies':[],'seconds':0.,'telemetry':[]} for k in ('uniform','adaptive')}
    client=WorkerClient()
    for offset in range(0,len(baseline),4):
        order=('uniform','adaptive') if (offset//4)%2==0 else ('adaptive','uniform')
        for name in order:
            events=baseline if name=='uniform' else variants
            done={r['id'] for r in arms[name]['replies']}
            batch=[e for e in events[offset:offset+4] if e['id'] not in done]
            if not batch: continue
            started=time.perf_counter()
            response=client.post('/verify-batch',{'events':batch},timeout=4*480+30)
            arms[name]['seconds']+=time.perf_counter()-started
            arms[name]['replies'].extend(response['results']);arms[name]['telemetry'].append(response.get('telemetry',{}))
            statefile.write_text(json.dumps(arms,indent=2))
            print(name,len(arms[name]['replies']),'/',len(events),flush=True)
    for name,arm in arms.items():
        if {r['id'] for r in arm['replies']}!={e['id'] for e in baseline} or len(arm['replies'])!=len(baseline): raise ValueError('Missing/duplicate reply')
        arm['predictions']=predictions(rows,arm['replies'],intervals)
        arm['metrics']=metrics(rows,arm['predictions'])
        arm['calls']=sum(r.get('calls',0) for r in arm['replies'])
    result={'protocol':protocol,'arms':arms,'preparation_seconds':prep_seconds,
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in hashes.items()),
        'code_unchanged':file_hash(Path(__file__))==protocol['script_sha256'],'test_opened':False}
    (root/'results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps({k:v['metrics'] for k,v in arms.items()}),flush=True)


if __name__=='__main__':main()
