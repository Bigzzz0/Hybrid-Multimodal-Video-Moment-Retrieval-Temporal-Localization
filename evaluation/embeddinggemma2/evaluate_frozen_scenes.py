"""Frozen current-method evaluation on newly selected official validation scenes.

No threshold search. Normal banks use training-normal recordings only.
Selected units run first; skipped units run separately as a diagnostic audit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import requests

from app.gemma2 import normality as n
from app.gemma2.config import IndexConfig
from app.gemma2.open_screening import reference_frames
from app.gemma2.service import EventService, WorkerClient
from app.gemma2.store import Store, ReindexRequired, file_hash
from . import worker
from .ablate_open_context import RULE
from .compare_matched_gemma_gate import filter_payload, predictions
from .open_anomaly_eval import audit, bootstrap, metrics, stage_diagnostics
from .research_qwen_residency import idle, release_policy, health

CONTEXT='A fixed surveillance camera observing a pedestrian area. Routine walking and waiting are normal; evaluate activity in context.'


def build_payload(service, row, timeline, evidence, folder):
    """Identical fixed-grid/reference ordering to the prior matched-frame study."""
    events=[];intervals={};lookup={w['id']:w for w in timeline}
    for i,start in enumerate(np.arange(0,row['duration'],12.)):
        end=min(float(start)+12,row['duration']);found={}
        for wi,channels in evidence.items():
            w=lookup[int(wi)]
            if w['t_start']>=end or w['t_end']<=start:continue
            for entries in channels.values():
                for score in entries:
                    for ref in score['references']:found.setdefault(ref['key'],ref)
        frames=n.sample_frames(row['path'],float(start),end,row['duration'],folder/row['id']/str(i))
        identifier=row['id']+f'-{i}'
        intervals[identifier]={'video_id':row['id'],'t_start':float(start),'t_end':end}
        events.append({'id':identifier,'kind':'open_anomaly','verification_mode':n.POLICY,'generate_caption':False,
            'scene_context':CONTEXT+' '+RULE,'frame_paths':[f['path'] for f in frames],
            'timestamps':[f['timestamp'] for f in frames],'reference_frames':reference_frames(service,list(found.values())[:2])})
    return events,intervals


def verify_phase(events,path):
    """Bounded offline retention experiment; no patch to running broker."""
    idle();worker.release_qwen();started=time.perf_counter()
    state=json.loads(path.read_text()) if path.exists() else {'replies':[],'telemetry':[],'seconds':0.}
    prior=state['seconds'];done={r['id'] for r in state['replies']}
    remaining=[e for e in events if e['id'] not in done]
    try:
        with release_policy(worker,True):
            for offset in range(0,len(remaining),8):
                idle();batch=remaining[offset:offset+8]
                response=worker.verify_batch(worker.VerifyBatch(events=batch))
                state['replies'].extend(response['results']);state['telemetry'].append(response['telemetry'])
                state['seconds']=prior+time.perf_counter()-started
                path.write_text(json.dumps(state,indent=2),encoding='utf-8')
                print(path.stem,len(state['replies']),'/',len(events),flush=True)
    finally:worker.release_qwen()
    state['seconds']=prior+time.perf_counter()-started;state['final_health']=health()
    if len(state['replies'])!=len(events) or {r['id'] for r in state['replies']}!={e['id'] for e in events}:
        raise ValueError('Missing/duplicate phase replies')
    path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--selection-lock',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'results.json').exists():raise ValueError('Completed results are immutable')
    manifest_path=Path(args.manifest);manifest=json.loads(manifest_path.read_text(encoding='utf-8'));rows=manifest['videos'];audit(rows)
    lock_path=Path(args.selection_lock);lock=json.loads(lock_path.read_text());config=IndexConfig()
    if sorted({int(r['scene_id'].split('-')[-1]) for r in rows})!=lock['scenes']:raise ValueError('Selection differs from pre-download lock')
    old_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    old=json.loads(old_path.read_text())['videos'];audit(old+rows)
    if {r['scene_id'] for r in rows}&{r['scene_id'] for r in old}:raise ValueError('Previously inspected scenes included')
    if {r['sha256'] for r in rows}&{r['sha256'] for r in old}:raise ValueError('Old video bytes reused')
    files=[Path(__file__),Path(worker.__file__),Path('backend/app/gemma2/normality.py'),Path('backend/inference_worker/vlm_service.py')]
    protocol={'version':'frozen-new-scenes-v1','manifest_sha256':file_hash(manifest_path),'selection_lock_sha256':file_hash(lock_path),
        'old_manifest_sha256':file_hash(old_path),'thresholds':lock['frozen_thresholds'],'config':config.json(),
        'context':CONTEXT,'qwen_rule':RULE,'prompts':n.PROMPTS,'code_hashes':{str(p):file_hash(p) for p in files},
        'threshold_tuning':False,'test_opened':False,'data_split':'new official validation scenes, not official test',
        'verification':'fixed12s grid, same uniform images/references and frozen gate; retain Qwen during selected phase; first-pass uncertain kept separate',
        'audit':'run skipped units separately after selected; all-unit result is diagnostic union of actual shared replies, not a timed second all-unit run',
        'production_activation':False}
    pp=root/'protocol.json';serial=json.loads(json.dumps(protocol))
    if pp.exists() and json.loads(pp.read_text())!=serial:raise ValueError('Protocol changed on resume')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    jobs=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
    if any(j['status'] in ('running','queued','cancelling') for j in jobs):raise ValueError('Demo jobs active')
    service=EventService(Store(root/'store'),WorkerClient(),config);index_telemetry=[];started=time.perf_counter()
    try:
        for row in rows:
            video=service.register(row['path'],row['id'],name=row['id']);video['dataset_split']=row['split'];service.store.put('videos',row['id'],video)
            try:service.store.require(row['id'],service.namespace(video))
            except ReindexRequired:
                job_id='index-'+row['id']
                try:job=service.store.get('jobs',job_id)
                except KeyError:job={'id':job_id,'video_id':row['id'],'cancel_requested':False}
                job.update(status='running',cancel_requested=False);service.store.put('jobs',job_id,job)
                service.index(job_id,row['id']);service.update(job_id,status='complete')
            index_telemetry.append(service.client.last_telemetry)
            print('INDEXED',row['id'],flush=True)
        index_seconds=time.perf_counter()-started
        profile_path=root/'profiles.json';profiles=json.loads(profile_path.read_text()) if profile_path.exists() else {}
        for scene in sorted({r['scene_id'] for r in rows}):
            if scene in profiles:continue
            refs=[r for r in rows if r['scene_id']==scene and r['split']=='development']
            if any(r['events'] or r['official_split']!='training' for r in refs):raise ValueError('Invalid reference source')
            profile=n.create_profile(service.store,scene,CONTEXT)
            bank=n.add_references(service,profile['id'],[{'video_id':r['id'],'t_start':0.,'t_end':r['duration'],'split':'development','verified_normal':True} for r in refs])
            profiles[scene]={'id':profile['id'],'version':bank['version']};profile_path.write_text(json.dumps(profiles,indent=2))
        texts=[t+' Scene context: '+CONTEXT for role in ('suspicious','normal') for t in n.PROMPTS[role]]
        encoded=[]
        for i in range(0,len(texts),4):encoded.extend(service.client.embed('query',config,texts=texts[i:i+4]))
        count=len(n.PROMPTS['suspicious']);prompts=(np.stack(encoded[:count]),np.stack(encoded[count:]))
        validation=[r for r in rows if r['split']=='validation'];scores={};candidates={};payload=[];intervals={};screened={}
        for row in validation:
            video=service.store.get('videos',row['id']);namespace=service.namespace(video);timeline=service.store.require(row['id'],namespace)
            bank=n.bank(service.store,profiles[row['scene_id']]['id'],profiles[row['scene_id']]['version'])
            evidence=n.score_channels(service.store.embeddings(namespace,config.dimension),bank,prompts,protocol['thresholds'],config.dimension)
            scores[row['id']]=evidence;candidates[row['id']]=n.segments(timeline,evidence)
            screened[row['id']]=[{'t_start':u['t_start'],'t_end':u['t_end'],'assessment':'suspicious'} for u in candidates[row['id']]]
            events,lookup=build_payload(service,row,timeline,evidence,root/'verification-frames');payload.extend(events);intervals.update(lookup)
        selected=filter_payload(payload,intervals,candidates);selected_ids={p['id'] for p in selected};skipped=[p for p in payload if p['id'] not in selected_ids]
        hashes={p:file_hash(Path(p)) for event in payload for p in event['frame_paths']+[r['path'] for r in event['reference_frames']]}
        (root/'requests.json').write_text(json.dumps({'all_units':payload,'selected':selected,'intervals':intervals,'frame_hashes':hashes},indent=2),encoding='utf-8')
        (root/'gate.json').write_text(json.dumps({'scores':scores,'candidates':candidates,'profiles':profiles},indent=2),encoding='utf-8')
        prep_seconds=time.perf_counter()-started-index_seconds
        service.client.post('/unload',{})
        print('GATED',len(selected),'/',len(payload),'Qwen units',flush=True)
        chosen=verify_phase(selected,root/'selected-partial.json')
        extra=verify_phase(skipped,root/'skipped-audit-partial.json') if skipped else {'replies':[],'telemetry':[],'seconds':0.}
        final=predictions(validation,chosen['replies'],intervals);all_pred=predictions(validation,chosen['replies']+extra['replies'],intervals)
        selected_grid=predictions(validation,[{'id':e['id'],'verdict':{'assessment':'suspicious'}} for e in selected],intervals)
        pred={'gemma_windows':screened,'selected_grid':selected_grid,'gemma_qwen':final,'all_qwen_diagnostic':all_pred}
        result={'status':'complete','protocol':protocol,'profiles':profiles,'index_seconds_this_process':index_seconds,
            'query_gate_frame_preparation_seconds':prep_seconds,'index_jobs':[service.store.get('jobs','index-'+r['id']) for r in rows],
            'index_telemetry':index_telemetry,'selected_phase':chosen,'skipped_audit':extra,'total_grid_units':len(payload),'selected_units':len(selected),
            'qwen_call_reduction':1-len(selected)/len(payload) if payload else None,'predictions':pred,
            'metrics':{key:metrics(validation,value) for key,value in pred.items()},
            'per_scene':{scene:metrics([r for r in validation if r['scene_id']==scene],final) for scene in profiles},
            'confidence_intervals':bootstrap(validation,final),'stage_diagnostics':stage_diagnostics(validation,selected_grid,final),
            'code_unchanged':all(file_hash(Path(p))==sha for p,sha in protocol['code_hashes'].items()),
            'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in hashes.items()),'test_opened':False,'production_activation':False}
        (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('COMPLETE',json.dumps(result['metrics']),flush=True)
    finally:
        service.client.post('/unload',{});worker.release_qwen();service.executor.shutdown(wait=True)


if __name__=='__main__':main()
