"""Second factor: evidence-grounded instructions on both frozen frame samplers."""
import argparse
import copy
import json
from pathlib import Path
import time

import requests

from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash
from .compare_matched_gemma_gate import predictions
from .open_anomaly_eval import audit, metrics


EVIDENCE = ''' Evidence inspection protocol: Inspect the current frames chronologically before deciding. In observations, describe concrete visible changes across timestamps: actor or object state, movement, interaction, or environmental change. Distinguish observed actions from guesses about appearance or intent. In deviation_from_normal, identify the actual activity or environmental difference supported by current frames and camera routine; a shared background does not prove that the activity is normal, and an unfamiliar appearance does not by itself prove a behavioral anomaly. In alternative_explanation, state the strongest routine explanation and whether the images actually support it. Base the assessment on this comparison. Do not invent motion between sampled frames. If the relevant action or routine explanation cannot be resolved from the images, use uncertain and explain the missing evidence rather than choosing a confident normal or suspicious answer. Keep the existing JSON schema and zero-based evidence-frame index rules.'''


def evidence_payload(events):
    result=copy.deepcopy(events)
    for event in result:
        event['scene_context']+=EVIDENCE
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    source=Path(args.source);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    base=json.loads((source/'results.json').read_text()); frozen=json.loads((source/'requests.json').read_text())
    if not base['code_unchanged'] or not base['frames_unchanged']:raise ValueError('Unstable source')
    manifest_path=Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')
    if file_hash(manifest_path)!=base['protocol']['manifest_sha256']:raise ValueError('Manifest changed')
    manifest=json.loads(manifest_path.read_text());audit(manifest['videos'])
    rows=[r for r in manifest['videos'] if r['split']=='validation']
    jobs=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
    if any(j['status'] in ('running','queued','cancelling') for j in jobs):raise ValueError('Demo job active')
    for p,sha in frozen['frame_hashes'].items():
        if file_hash(Path(p))!=sha:raise ValueError('Frame changed')
    payloads={'evidence':evidence_payload(frozen['uniform']),'adaptive_evidence':evidence_payload(frozen['adaptive'])}
    protocol={'version':'evidence-frames-v1','evidence_instruction':EVIDENCE,'source_sha256':file_hash(source/'results.json'),
        'script_sha256':file_hash(Path(__file__)),'threshold_tuning':False,'test_opened':False,
        'order':'alternating first arm every four units','same_frames_as_source':True,
        'selection_uses_labels':False,'limitation':'Factorial extension reuses fresh baseline results from preceding run; no new held-out test'}
    pp=root/'protocol.json'
    if pp.exists() and json.loads(pp.read_text())!=protocol:raise ValueError('Use fresh output')
    pp.write_text(json.dumps(protocol,indent=2));(root/'requests.json').write_text(json.dumps(payloads,indent=2))
    state=root/'partial.json';arms=json.loads(state.read_text()) if state.exists() else {k:{'replies':[],'seconds':0.,'telemetry':[]} for k in payloads}
    client=WorkerClient();count=len(frozen['uniform'])
    for offset in range(0,count,4):
        order=list(payloads) if offset//4%2==0 else list(reversed(payloads))
        for name in order:
            done={r['id'] for r in arms[name]['replies']}
            batch=[e for e in payloads[name][offset:offset+4] if e['id'] not in done]
            if not batch:continue
            started=time.perf_counter();response=client.post('/verify-batch',{'events':batch},timeout=4*480+30)
            arms[name]['seconds']+=time.perf_counter()-started
            arms[name]['replies'].extend(response['results']);arms[name]['telemetry'].append(response.get('telemetry',{}))
            state.write_text(json.dumps(arms,indent=2));print(name,len(arms[name]['replies']),'/',count,flush=True)
    for name,arm in arms.items():
        if len(arm['replies'])!=count or {r['id'] for r in arm['replies']}!={e['id'] for e in payloads[name]}:raise ValueError('Missing/duplicate replies')
        arm['predictions']=predictions(rows,arm['replies'],frozen['intervals']);arm['metrics']=metrics(rows,arm['predictions'])
        arm['calls']=sum(r.get('calls',0) for r in arm['replies'])
    result={'protocol':protocol,'arms':arms,'test_opened':False,'source_arms':base['arms'],
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in frozen['frame_hashes'].items()),
        'code_unchanged':file_hash(Path(__file__))==protocol['script_sha256']}
    (root/'results.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps({k:v['metrics'] for k,v in arms.items()}),flush=True)


if __name__=='__main__':main()
