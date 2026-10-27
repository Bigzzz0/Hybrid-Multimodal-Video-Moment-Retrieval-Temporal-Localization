"""Actual fixed-grid contextual Qwen run; preserve unknowns and all raw replies.

Normality policy/references are reused, target intervals are explicit and 3s long.
All units are run, not selected from labels, score peaks or known prior errors.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
from app.gemma2.normality import sample_frames
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash
from .open_anomaly_eval import audit,metrics,bootstrap

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    source=Path('scratch/event-retrieval/open-anomaly-context-ablation-v1/protocol.json')
    prior=json.loads(source.read_text(encoding='utf-8'))
    manifest_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));audit(manifest['videos'])
    rows=[r for r in manifest['videos'] if r['split']=='validation']
    payload=[];lookup={};frames_hash={}
    protocol={'source_sha256':file_hash(Path(__file__)),'manifest_sha256':file_hash(manifest_path),
        'context_source_sha256':file_hash(source),'unit_seconds':3.,'stride_seconds':3.,'before_after_context_seconds':5.,
        'same_model':'caprl_qwen3vl_4b_q6','max_frames':12,'normal_references':'unchanged prior development references',
        'changed_factors':['12s units replaced by 3s grid','Explicit target bounds; before/after frames are context only'],
        'test_opened':False,'production_activation':False,'selection':'all validation units, label-blind fixed grid'}
    protocol_path=root/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Use fresh output')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for row in rows:
        template=next(p for p in prior['payload'] if p['id']==row['id']+'-0')
        for number,start in enumerate(np.arange(0,row['duration'],3.)):
            end=min(float(start+3),row['duration']);identifier=row['id']+f'-unit{number}'
            frames=sample_frames(row['path'],float(start),end,row['duration'],root/'frames'/identifier)
            event=copy.deepcopy(template)
            event.update(id=identifier,frame_paths=[f['path'] for f in frames],timestamps=[f['timestamp'] for f in frames])
            event['scene_context']+=(f' TARGET INTERVAL: {start:.3f} to {end:.3f} seconds. '
                'Assess only activity in the TARGET INTERVAL. Earlier and later frames are context, '
                'not additional target events. Do not classify the target as abnormal solely because '
                'an event occurs outside its interval. If no transition is established use uncertain.')
            payload.append(event);lookup[identifier]=(row['id'],float(start),end)
            for path in event['frame_paths']+[f['path'] for f in event['reference_frames']]:
                frames_hash[path]=file_hash(Path(path))
    (root/'requests.json').write_text(json.dumps({'payload':payload,'frame_hashes':frames_hash},indent=2),encoding='utf-8')
    partial_path=root/'partial.json'
    partial=json.loads(partial_path.read_text(encoding='utf-8')) if partial_path.exists() else {'replies':[],'telemetry':[],'elapsed_seconds':0}
    replies,telemetry=partial['replies'],partial['telemetry'];done={r['id'] for r in replies}
    todo=[e for e in payload if e['id'] not in done]
    client=WorkerClient();started=time.perf_counter()
    for offset in range(0,len(todo),8):
        response=client.post('/verify-batch',{'events':todo[offset:offset+8]},timeout=8*480+30)
        replies.extend(response['results']);telemetry.append(response.get('telemetry',{}))
        partial_path.write_text(json.dumps({'replies':replies,'telemetry':telemetry,
            'elapsed_seconds':partial['elapsed_seconds']+time.perf_counter()-started},indent=2),encoding='utf-8')
        print('QWEN',len(replies),'/',len(payload),flush=True)
    if len(replies)!=len(payload) or {r['id'] for r in replies}!=set(lookup):raise ValueError('Missing/duplicate replies')
    pred={r['id']:[] for r in rows}
    for reply in replies:
        video,start,end=lookup[reply['id']]
        assessment=(reply.get('verdict') or {}).get('assessment','uncertain')
        pred[video].append({'t_start':start,'t_end':end,'assessment':assessment})
    result={'status':'complete','protocol':protocol,'predictions':pred,'replies':replies,'telemetry':telemetry,
        'metrics':metrics(rows,pred),'confidence_intervals':bootstrap(rows,pred),
        'http_requests':sum(r.get('calls',0) for r in replies),
        'unknown_requests':sum(bool(r.get('calls_unknown')) for r in replies),
        'elapsed_seconds':partial['elapsed_seconds']+time.perf_counter()-started,
        'code_unchanged':file_hash(Path(__file__))==protocol['source_sha256'],
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in frames_hash.items()),
        'test_opened':False,'production_activation':False}
    (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('QWEN COMPLETE',json.dumps(result['metrics']),flush=True)

if __name__=='__main__':main()
