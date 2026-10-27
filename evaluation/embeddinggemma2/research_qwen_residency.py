"""Offline lifecycle-only experiment using the unchanged broker verify function.

Never changes the running broker or production defaults. Requires idle services.
The isolated process defers its own release hook within a verification phase.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time
from unittest.mock import patch

import psutil
import requests

from app.gemma2.store import file_hash
from . import worker
from .compare_matched_gemma_gate import predictions
from .open_anomaly_eval import audit, metrics


@contextmanager
def release_policy(module, retain):
    """Defer this isolated module's releases; always restore and release on exit."""
    counts={'deferred_releases':0}
    real_release=module.release_qwen
    def deferred():
        counts['deferred_releases']+=1
    if not retain:
        yield counts
        return
    try:
        with patch.object(module,'release_qwen',deferred):
            yield counts
    finally:
        real_release()


def idle():
    jobs=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
    if any(j['status'] in ('running','queued','cancelling') for j in jobs):
        raise ValueError('Demo jobs active: stop isolated benchmark')
    gemma=requests.get('http://127.0.0.1:8012/health',timeout=10).json()
    if gemma.get('resident'):
        raise ValueError('Gemma resident: stop isolated benchmark')
    if psutil.virtual_memory().available < 1024**3:
        raise MemoryError('Insufficient available RAM for next batch')


def health():
    r=requests.get(worker.QWEN_URL+'/health',timeout=10);r.raise_for_status();return r.json()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.repeats<=2:raise ValueError('Use one or two matched rounds')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    source=Path('scratch/event-retrieval/adaptive-frames-20261015-v2/requests.json')
    data=json.loads(source.read_text());events=data['uniform'];intervals=data['intervals']
    manifest_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    manifest=json.loads(manifest_path.read_text());audit(manifest['videos'])
    rows=[r for r in manifest['videos'] if r['split']=='validation']
    if len(events)!=16:raise ValueError('Expected frozen sixteen-unit benchmark')
    for p,sha in data['frame_hashes'].items():
        if file_hash(Path(p))!=sha:raise ValueError('Frozen frame changed')
    files=[Path(__file__),Path(worker.__file__),Path('backend/inference_worker/vlm_service.py')]
    protocol={'version':'qwen-residency-v1','source_sha256':file_hash(source),'manifest_sha256':file_hash(manifest_path),
        'code_hashes':{str(p):file_hash(p) for p in files},'batch_size':8,'repeats':args.repeats,
        'only_difference':'defer release until end of sixteen-unit phase instead of each eight-unit batch',
        'execution':'isolated in-process copy of unchanged worker.verify_batch; private Qwen worker shared with idle demo; running broker not patched',
        'order':[['release_each_batch','retain_phase'] if i%2==0 else ['retain_phase','release_each_batch'] for i in range(args.repeats)],
        'timing':'phase includes both batches, checks, telemetry and final release; initial forced unload outside each arm; cached JPEGs/Gemma gate, no media indexing',
        'test_opened':False,'production_activation':False,'no_verdict_cache':True}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    runs=[]
    try:
        for round_index,order in enumerate(protocol['order']):
            for name in order:
                idle();worker.release_qwen()
                run={'round':round_index+1,'arm':name,'replies':[],'telemetry':[],'batches':[],'model_loads':0}
                started=time.perf_counter()
                with release_policy(worker,name=='retain_phase') as lifecycle:
                    for offset in range(0,len(events),8):
                        idle();before=health()
                        if before['models']['states']['q6']=='unloaded':run['model_loads']+=1
                        tick=time.perf_counter();response=worker.verify_batch(worker.VerifyBatch(events=events[offset:offset+8]))
                        run['batches'].append({'offset':offset,'seconds':time.perf_counter()-tick,'before':before,'after':health()})
                        run['replies'].extend(response['results']);run['telemetry'].append(response['telemetry'])
                        (root/'partial.json').write_text(json.dumps({'completed_runs':runs,'current':run},indent=2),encoding='utf-8')
                        if any(r.get('error') for r in response['results']):raise ValueError('Technical error; retain partial results')
                        print('ROUND',round_index+1,name,len(run['replies']),'/',len(events),flush=True)
                run['seconds']=time.perf_counter()-started;run['lifecycle']=lifecycle
                if len(run['replies'])!=len(events) or {r['id'] for r in run['replies']}!={e['id'] for e in events}:raise ValueError('Missing or duplicate response')
                run['predictions']=predictions(rows,run['replies'],intervals);run['metrics']=metrics(rows,run['predictions'])
                run['calls']=sum(r.get('calls',0) for r in run['replies']);run['final_health']=health()
                if run['final_health']['models']['states']['q6']!='unloaded':raise ValueError('Qwen not released')
                runs.append(run)
                (root/'partial.json').write_text(json.dumps({'completed_runs':runs},indent=2),encoding='utf-8')
                print('DONE',name,round(run['seconds'],2),run['metrics'],flush=True)
    finally:
        worker.release_qwen()
    result={'protocol':protocol,'runs':runs,'test_opened':False,
        'code_unchanged':all(file_hash(Path(p))==sha for p,sha in protocol['code_hashes'].items()),
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in data['frame_hashes'].items())}
    comparisons=[]
    for i in range(args.repeats):
        base=next(r for r in runs if r['round']==i+1 and r['arm']=='release_each_batch')
        new=next(r for r in runs if r['round']==i+1 and r['arm']=='retain_phase')
        baseline={r['id']:r.get('verdict') or {} for r in base['replies']}
        changes=[{'id':r['id'],'before':baseline[r['id']].get('assessment'),'after':(r.get('verdict') or {}).get('assessment')}
                 for r in new['replies'] if baseline[r['id']].get('assessment')!=(r.get('verdict') or {}).get('assessment')]
        comparisons.append({'round':i+1,'seconds_saved':base['seconds']-new['seconds'],'fraction_saved':1-new['seconds']/base['seconds'],
                            'assessment_changes':changes,'raw_reply_changes':sum(baseline[r['id']].get('raw_text')!=(r.get('verdict') or {}).get('raw_text') for r in new['replies'])})
    result['comparisons']=comparisons
    (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(comparisons),flush=True)


if __name__=='__main__':main()
