"""Matched 12s-grid comparison: real all-unit Qwen versus Gemma-gated Qwen.

Same frozen Qwen payloads (frames, timestamps, references and normality prompt).
Recompute Gemma gates from its cached media vectors with freshly encoded queries.
No tuning, test access, demo DB writes, changed Qwen prompts or clamped evidence.
"""
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import requests

from app.gemma2 import normality as n
from app.gemma2.config import IndexConfig
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash, vectors
from .open_anomaly_eval import audit, metrics, bootstrap

DATA=Path('scratch/event-retrieval')

def overlaps(a,b):
    return max(a['t_start'],b['t_start']) < min(a['t_end'],b['t_end'])

def filter_payload(payload,intervals,candidates):
    """Only remove whole units; never change their input or consult labels."""
    return [copy.deepcopy(p) for p in payload if any(overlaps(intervals[p['id']],c)
        for c in candidates[intervals[p['id']]['video_id']])]

def predictions(rows,replies,intervals):
    if len({r['id'] for r in replies})!=len(replies):raise ValueError('Duplicate Qwen reply')
    pred={r['id']:[] for r in rows}
    for reply in replies:
        unit=intervals[reply['id']]
        pred[unit['video_id']].append({'t_start':unit['t_start'],'t_end':unit['t_end'],
            'assessment':(reply.get('verdict') or {}).get('assessment','uncertain')})
    return pred

def recompute_gate(source,rows,root,client):
    frozen=json.loads((source/'results.json').read_text(encoding='utf-8'))
    config=IndexConfig(**frozen['protocol']['config'])
    if frozen['protocol']['prompts']!=n.PROMPTS:raise ValueError('Prompt bank changed')
    jobs={j['video_id']:j for j in frozen['verification_jobs']}
    connection=sqlite3.connect(f"file:{(source/'store/events.sqlite3').resolve().as_posix()}?mode=ro",uri=True)
    cache=root/'gate-query-embeddings.npz';telemetry=[];started=time.perf_counter()
    contexts=sorted({j['open_policy']['context'] for j in jobs.values()})
    texts=[t+' Scene context: '+context for context in contexts for role in ('suspicious','normal') for t in n.PROMPTS[role]]
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:
            if f['texts'].tolist()!=texts:raise ValueError('Cached text mismatch')
            encoded=vectors(f['embeddings'],config.dimension)
    else:
        batches=[]
        for start in range(0,len(texts),4):
            batches.extend(client.embed('query',config,texts=texts[start:start+4]))
            telemetry.append(client.last_telemetry)
        encoded=np.stack(batches);np.savez(cache,texts=texts,embeddings=encoded)
    lookup=dict(zip(texts,encoded));all_scores={};candidates={}
    try:
        for row in rows:
            job=jobs[row['id']];policy=job['open_policy'];namespace=job['namespace']
            record=connection.execute('select config,status from indexes where namespace=?',(namespace,)).fetchone()
            if record is None or record[1]!='ready' or json.loads(record[0])!=config.json():
                # JSON turns the tuple modalities into a list.
                if record is None or record[1]!='ready' or json.loads(record[0])!=json.loads(json.dumps(config.json())):
                    raise ValueError('Index configuration mismatch')
            bank=json.loads(connection.execute('select data from normality_banks where profile_id=? and version=?',
                (policy['profile_id'],policy['reference_version'])).fetchone()[0])
            for ref in bank['references']:
                if ref['video_id'] not in {r['id'] for r in frozen['lock']['source_ledger'] if r['split']=='development' and not r['events']}:
                    raise ValueError('Non-development reference')
            channel_rows={}
            for wi,modality,dimension,blob in connection.execute('select window_id,modality,dimension,vector from embeddings where namespace=? order by window_id,slot',(namespace,)):
                if dimension!=config.dimension:raise ValueError('Dimension mismatch')
                channel_rows.setdefault(modality,[]).append((wi,np.frombuffer(blob,dtype=np.float32).copy()))
            channels={key:(np.array([i for i,v in values]),vectors([v for i,v in values],config.dimension)) for key,values in channel_rows.items()}
            prompts=tuple(np.stack([lookup[t+' Scene context: '+policy['context']] for t in n.PROMPTS[role]]) for role in ('suspicious','normal'))
            scores=n.score_channels(channels,bank,prompts,frozen['lock']['thresholds']['combined'],config.dimension)
            timeline=[json.loads(data) for data, in connection.execute('select data from windows where namespace=? order by id',(namespace,))]
            all_scores[row['id']]=scores
            candidates[row['id']]=[{k:u[k] for k in ('t_start','t_end')} for u in n.segments(timeline,scores)]
    finally:connection.close()
    result={'candidates':candidates,'scores':all_scores,'thresholds':frozen['lock']['thresholds']['combined'],
        'seconds':time.perf_counter()-started,'embedding_telemetry':telemetry,'cached_media':True,
        'includes_original_video_indexing':False}
    (root/'gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'datasets/ubnormal-open-multiscene-v1/manifest.json'
    source=DATA/'open-anomaly-multiscene-v2'
    prompt_path=DATA/'open-anomaly-context-ablation-v1/protocol.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));audit(manifest['videos'])
    rows=[r for r in manifest['videos'] if r['split']=='validation']
    original=json.loads(prompt_path.read_text(encoding='utf-8'));payload=original['payload']
    if original['manifest_hash']!=n.digest(manifest):raise ValueError('Manifest mismatch')
    active=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
    if any(j['status'] in ('running','queued','cancelling') for j in active):raise ValueError('Demo jobs active')
    intervals={}
    for event in payload:
        video,index=event['id'].rsplit('-',1);row=next(r for r in rows if r['id']==video)
        intervals[event['id']]={'video_id':video,'t_start':int(index)*12.,'t_end':min(int(index)*12+12.,row['duration'])}
    protocol={'version':'matched-gemma-gate-v1','manifest_sha256':file_hash(manifest_path),
        'source_results_sha256':file_hash(source/'results.json'),'frozen_qwen_payload_sha256':file_hash(prompt_path),
        'code_hashes':{str(p):file_hash(p) for p in [Path(__file__),Path('backend/app/gemma2/normality.py'),Path('evaluation/embeddinggemma2/worker.py'),Path('backend/inference_worker/vlm_service.py')]},
        'only_difference':'skip complete frozen 12-second Qwen units with no strict overlap with frozen-threshold Gemma candidates',
        'frame_or_prompt_changes':False,'batch_size':8,'order':['all_qwen','gemma_then_qwen'],
        'threshold_tuning':False,'threshold_origin':'prior validation lock; not independent test',
        'test_opened':False,'production_activation':False,
        'timing_scope':'cached video index / existing Qwen JPEGs; Gemma query encoding and scoring measured separately; one sequential run, no file-cache order control'}
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text(encoding='utf-8'))!=protocol:raise ValueError('Use fresh output')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    hashes={p:file_hash(Path(p)) for e in payload for p in e['frame_paths']+[r['path'] for r in e['reference_frames']]}
    for p,sha in hashes.items():
        if original['frame_hashes'].get(p)!=sha:raise ValueError('Frozen frame changed')
    client=WorkerClient();gate=recompute_gate(source,rows,root,client)
    selected=filter_payload(payload,intervals,gate['candidates'])
    (root/'requests.json').write_text(json.dumps({'all_qwen':payload,'gemma_then_qwen':selected,'intervals':intervals,'frame_hashes':hashes},indent=2),encoding='utf-8')
    print('GEMMA GATE',len(selected),'/',len(payload),'units',flush=True)
    arms={}
    for name,events in [('all_qwen',payload),('gemma_then_qwen',selected)]:
        state=root/(name+'-partial.json')
        prior=json.loads(state.read_text(encoding='utf-8')) if state.exists() else {'replies':[],'telemetry':[],'seconds':0.}
        replies,telemetry=prior['replies'],prior['telemetry'];done={r['id'] for r in replies}
        todo=[e for e in events if e['id'] not in done];started=time.perf_counter()
        for offset in range(0,len(todo),8):
            response=client.post('/verify-batch',{'events':todo[offset:offset+8]},timeout=8*480+30)
            replies.extend(response['results']);telemetry.append(response.get('telemetry',{}))
            state.write_text(json.dumps({'replies':replies,'telemetry':telemetry,'seconds':prior['seconds']+time.perf_counter()-started},indent=2),encoding='utf-8')
            print(name,len(replies),'/',len(events),flush=True)
        if len(replies)!=len(events) or {r['id'] for r in replies}!={e['id'] for e in events}:raise ValueError('Missing/duplicate replies')
        pred=predictions(rows,replies,intervals)
        arms[name]={'predictions':pred,'metrics':metrics(rows,pred),'confidence_intervals':bootstrap(rows,pred),
            'replies':replies,'telemetry':telemetry,'qwen_seconds':prior['seconds']+time.perf_counter()-started,
            'http_requests':sum(r.get('calls',0) for r in replies),'units':len(events),
            'technical_errors':dict(Counter(r.get('error') for r in replies if r.get('error')))}
        (root/(name+'.json')).write_text(json.dumps(arms[name],indent=2),encoding='utf-8')
    all_replies={r['id']:r for r in arms['all_qwen']['replies']};shared_changes=[]
    for r in arms['gemma_then_qwen']['replies']:
        before=(all_replies[r['id']].get('verdict') or {}).get('assessment','uncertain')
        after=(r.get('verdict') or {}).get('assessment','uncertain')
        if before!=after:shared_changes.append({'id':r['id'],'all_qwen':before,'gemma_then_qwen':after})
    # Same-run deterministic subset isolates gate effects from any repeat variation.
    subset={e['id'] for e in selected}
    replay=predictions(rows,[r for r in arms['all_qwen']['replies'] if r['id'] in subset],intervals)
    result={'status':'complete','protocol':protocol,'gate':gate,'arms':arms,
        'gate_only_replay':{'metrics':metrics(rows,replay),'predictions':replay},'shared_assessment_changes':shared_changes,
        'actual_request_reduction':1-arms['gemma_then_qwen']['http_requests']/arms['all_qwen']['http_requests'],
        'skipped_ids':[e['id'] for e in payload if e['id'] not in subset],
        'code_unchanged':all(file_hash(Path(p))==sha for p,sha in protocol['code_hashes'].items()),
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in hashes.items()),'test_opened':False,'production_activation':False}
    (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({k:v['metrics'] for k,v in arms.items()}),flush=True)

if __name__=='__main__':main()
