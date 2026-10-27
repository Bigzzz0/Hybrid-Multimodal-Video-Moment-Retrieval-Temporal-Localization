"""Audited, isolated adaptations of PRISM/Flashback and VideoPatchCore.

Exploratory leave-one-scene-out validation, not official paper reproduction.
No test access, learned model weights, production defaults or demo DB writes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from app.gemma2.config import IndexConfig
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash
from .open_anomaly_eval import audit, metrics, bootstrap

NORMAL = ["People walking calmly on a pedestrian path.", "People standing still or waiting.",
          "People sitting on a bench.", "People talking while standing together.",
          "A person carrying a bag while walking normally.", "A person stopping briefly and then continuing to walk.",
          "An empty public walkway with no unusual activity.", "Routine movement of pedestrians through a public area."]
CATEGORIES = {
    "interaction": ["People fighting and hitting each other.", "A person pushing another person forcefully."],
    "collapse": ["A person falls down onto the ground.", "A person lying motionless on a public walkway."],
    "running": ["A person suddenly running through an area where people normally walk.", "People running away or chasing someone."],
    "play": ["People jumping and performing unusual body movements on a walkway.", "People dancing or playing in a pedestrian passage."],
    "vehicle": ["A vehicle driving into a pedestrian-only area.", "A bicycle or motorcycle moving between pedestrians."],
    "object": ["A person throwing an object across a public space.", "A large unattended object obstructing a walkway."],
    "environment": ["Smoke or fire spreading through a public area.", "An object falling from above near pedestrians."],
    "boundary": ["A person climbing over a barrier into a restricted area.", "A person entering an area in an unusual direction."],
}

def unit(array):
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Expected finite matrix")
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norm < 1e-10):
        raise ValueError("Zero vector")
    return array / norm

def semantic_scores(media, normal, categories):
    """PRISM source M3/M5/M6 and Flashback-like M2, on frozen Gemma features.

    Text-only covariance; whiten semantic axes, never video distribution.
    Small generic bank replaces original large pseudo-caption memory.
    """
    x, normal = unit(media), unit(normal)
    groups = [unit(group) for group in categories]
    anomalies = np.concatenate(groups)
    texts = np.concatenate([normal, anomalies])
    center = normal.mean(0)
    delta = texts-texts.mean(0)
    covariance = delta.T@delta/(len(texts)-1) + .01*np.eye(texts.shape[1])
    eig, vectors = np.linalg.eigh(covariance)
    whitening = (vectors * (1/np.sqrt(np.maximum(eig, 1e-7))))@vectors.T
    axes = unit(np.stack([group.mean(0)-center for group in groups]))
    white_axes = unit(axes@whitening)
    def weighted(logits, temperature):
        exponential = np.exp((logits-logits.max(1, keepdims=True))/temperature)
        weights = exponential/exponential.sum(1, keepdims=True)
        return (weights*logits).sum(1)
    logits = (x-center)@white_axes.T
    # Original paper's anomaly penalty and retrieval temperature.
    bank = np.concatenate([normal, anomalies*.95])
    similarities = x@bank.T
    indices = np.argsort(similarities, axis=1)[:, -min(10, len(bank)):]
    selected = np.take_along_axis(similarities, indices, axis=1)
    exp = np.exp((selected-selected.max(1, keepdims=True))/.01)
    labels = (indices >= len(normal)).astype(float)
    flashback = (exp*labels).sum(1)/exp.sum(1)
    return {"semantic_margin": (x@anomalies.T).max(1)-(x@normal.T).max(1),
            "flashback_small_bank": flashback, "prism_raw_axes": weighted((x-center)@axes.T,.2),
            "prism_whitened_axes": weighted(logits,.2)}

def timeline(scores, intervals, duration):
    """Project all scores to shared one-second bins; no label-guided boundaries."""
    edges = np.append(np.arange(0, duration, 1.), duration)
    centers = (edges[:-1]+edges[1:])/2
    result = np.empty(len(centers))
    for i, center in enumerate(centers):
        matches = [score for score,(start,end) in zip(scores,intervals) if start <= center < end]
        if not matches:
            closest = min(range(len(intervals)), key=lambda j: abs(sum(intervals[j])/2-center))
            matches = [scores[closest]]
        result[i] = max(matches)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite timeline")
    return edges, result

def predictions(edges, scores, threshold):
    records = []
    for i, selected in enumerate(np.asarray(scores) >= threshold):
        if selected:
            if records and abs(records[-1]["t_end"]-edges[i]) < 1e-6:
                records[-1]["t_end"] = float(edges[i+1])
            else:
                records.append({"t_start": float(edges[i]), "t_end": float(edges[i+1]), "assessment":"suspicious"})
    return records

def select_threshold(rows, data):
    """Calibration must be called only on the other scenes' validation rows."""
    values = sorted(set(float(s) for row in rows for s in data[row['id']][1]))
    # Exhaustive finite observed thresholds plus explicit select-none threshold.
    values.append(float(np.nextafter(values[-1], np.inf)))
    best = None
    for threshold in values:
        pred = {r['id']: predictions(*data[r['id']], threshold) for r in rows}
        measured = metrics(rows, pred)
        key = (measured['recall'] >= .95, measured['recall'], -measured['fp'],
               -measured['positive_video_fraction'], threshold)
        if best is None or key > best[0]:
            best = key, threshold, measured
    return best[1:]

def cross_scene(rows, data):
    predictions_out, folds = {}, []
    for scene in sorted({r['scene_id'] for r in rows}):
        calibration = [r for r in rows if r['scene_id'] != scene]
        held = [r for r in rows if r['scene_id'] == scene]
        threshold, measured = select_threshold(calibration, data)
        pred = {r['id']: predictions(*data[r['id']], threshold) for r in held}
        predictions_out.update(pred)
        folds.append({'held_scene':scene, 'calibration_ids':[r['id'] for r in calibration],
                      'held_ids':[r['id'] for r in held], 'threshold':threshold,
                      'calibration_metrics':measured, 'held_metrics':metrics(held,pred)})
    return {'metrics':metrics(rows,predictions_out), 'folds':folds, 'predictions':predictions_out,
            'confidence_intervals':bootstrap(rows,predictions_out, repetitions=500)}

def cached_features(store, rows):
    db = sqlite3.connect(f"file:{Path(store).resolve().as_posix()}?mode=ro", uri=True)
    result = {}
    try:
        for row in rows:
            namespace = db.execute('select namespace from active where video_id=?',(row['id'],)).fetchone()[0]
            index = db.execute('select config,status from indexes where namespace=?',(namespace,)).fetchone()
            config = json.loads(index[0])
            if (index[1] != 'ready' or config['model_id'] != IndexConfig().model_id
                    or config['revision'] != IndexConfig().revision or config['dimension'] != 768):
                raise ValueError('Cached model/configuration mismatch')
            windows = {int(i):json.loads(data) for i,data in db.execute('select id,data from windows where namespace=? order by id',(namespace,))}
            media, intervals = [], []
            for wi, modality, slot, dimension, blob in db.execute('select window_id,modality,slot,dimension,vector from embeddings where namespace=? order by window_id,modality,slot',(namespace,)):
                if modality not in ('image','video'):
                    continue
                w=windows[wi]
                media.append(np.frombuffer(blob,dtype=np.float32).copy())
                if modality == 'image':
                    stamp = w['timestamps'][w['image_slots'][slot]]
                    intervals.append((stamp,min(row['duration'],stamp+1)))
                else:
                    intervals.append((w['t_start'],w['t_end']))
            result[row['id']] = (np.stack(media),intervals)
    finally:
        db.close()
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--vpc-features')
    args=parser.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    manifest_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')); rows=manifest['videos']; audit(rows)
    references=[r for r in rows if r['split']=='development']
    validation=[r for r in rows if r['split']=='validation']
    if any(r['events'] for r in references): raise ValueError('Abnormal reference rejected')
    protocol={'version':'research-screening-v1','manifest_sha256':file_hash(manifest_path),
        'source_sha256':file_hash(Path(__file__)), 'normal_prompts':NORMAL,'anomaly_prompts':CATEGORIES,
        'test_opened':False,'selection':'leave-one-scene-out validation; recall >=.95 then recall then FP then selected duration',
        'smoothing_sigma_seconds':[0,1], 'time_resolution_seconds':1,
        'sources':{'prism':'27c4533a45670f3bceddcf875de39a23dd09bc65','vpc':'eefdc8d7a84f8eb06a561e3c42ab26a1b944f9b6'},
        'limitations':['Adaptations, not official benchmark reproductions','Same small synthetic validation previously inspected; not independent test',
            'Flashback uses a small fixed generic bank, not original million-caption memory',
            'PRISM uses Gemma, not author Qwen/PE encoders; sampling differs',
            'Cached Gemma media encoding is excluded from current scoring time','Selecting best method still uses pooled validation results']}
    protocol_path=root/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol mismatch: use fresh output')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    texts=NORMAL+[t for values in CATEGORIES.values() for t in values]
    cache=root/'text-embeddings.npy'
    telemetry=[]
    if cache.exists(): embeddings=np.load(cache,allow_pickle=False)
    else:
        client=WorkerClient(); pieces=[]
        for i in range(0,len(texts),4):
            pieces.extend(client.embed('query',IndexConfig(),texts=texts[i:i+4]))
            telemetry.append(client.last_telemetry)
            print('TEXT',min(i+4,len(texts)),'/',len(texts),flush=True)
        embeddings=np.stack(pieces); np.save(cache,embeddings)
        (root/'embedding-telemetry.json').write_text(json.dumps(telemetry,indent=2),encoding='utf-8')
    normal=embeddings[:len(NORMAL)]; groups=[]; index=len(NORMAL)
    for values in CATEGORIES.values(): groups.append(embeddings[index:index+len(values)]); index+=len(values)
    cached=cached_features('scratch/event-retrieval/open-anomaly-multiscene-v2/store/events.sqlite3',rows)
    all_data={}; started=time.perf_counter()
    for row in validation:
        media, intervals=cached[row['id']]
        scores=semantic_scores(media,normal,groups)
        bank=unit(np.concatenate([cached[r['id']][0] for r in references if r['scene_id']==row['scene_id']]))
        distances=1-unit(media)@bank.T
        scores['gemma_normal_knn']=np.sort(distances,axis=1)[:,:3].mean(1)
        for method, values in scores.items():
            edges, signal=timeline(values,intervals,row['duration'])
            for sigma in (0,1):
                key=method+f'_smooth{sigma}'
                all_data.setdefault(key,{})[row['id']]=(edges,gaussian_filter1d(signal,sigma) if sigma else signal)
    if args.vpc_features:
        from .research_vpc import score_vpc
        vpc=score_vpc(rows,Path(args.vpc_features))
        for method, data in vpc.items():
            for sigma in (0,1):
                all_data[method+f'_smooth{sigma}']={key:(edges,gaussian_filter1d(signal,sigma) if sigma else signal)
                    for key,(edges,signal) in data.items()}
    score_seconds=time.perf_counter()-started
    results={}
    for name,data in all_data.items():
        results[name]=cross_scene(validation,data)
        print('RESULT',name,json.dumps(results[name]['metrics']),flush=True)
    best=max(results,key=lambda name:(results[name]['metrics']['recall']>=.95,results[name]['metrics']['recall'],
        -results[name]['metrics']['fp'],-results[name]['metrics']['positive_video_fraction']))
    report={'status':'complete','protocol':protocol,'methods':results,'best_exploratory_method':best,
        'scoring_seconds_excludes_cached_encoding':score_seconds,'test_opened':False,'production_activation':False,
        'timeline_scores':{name:{key:{'edges':edges.tolist(),'scores':scores.tolist()} for key,(edges,scores) in data.items()} for name,data in all_data.items()}}
    (root/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('BEST',best,flush=True)

if __name__=='__main__':main()
