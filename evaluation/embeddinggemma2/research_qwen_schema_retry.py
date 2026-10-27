"""One label-blind schema retry; raw baseline stays immutable, evidence stays strict."""
import copy
import json
from pathlib import Path
import time
from app.gemma2.service import WorkerClient
from app.gemma2.store import file_hash
from .open_anomaly_eval import metrics,bootstrap

ROOT=Path('scratch/event-retrieval')

def main():
    source=ROOT/'research-qwen-units-20261015-v1'
    out=ROOT/'research-qwen-schema-retry-20261015-v1';out.mkdir(parents=True,exist_ok=True)
    original=json.loads((source/'results.json').read_text(encoding='utf-8'))
    requests=json.loads((source/'requests.json').read_text(encoding='utf-8'))
    failures={r['id'] for r in original['replies'] if r.get('status')=='unverified' and r.get('error')}
    protocol={'source_sha256':file_hash(Path(__file__)),'original_result_sha256':file_hash(source/'results.json'),
        'selection':'All technical unverified errors, no semantic status/label selection',
        'maximum_additional_attempts_per_unit':1,'frame_changes':False,'model_or_parser_changes':False,
        'test_opened':False,'production_activation':False}
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=protocol:raise ValueError('Fresh output required')
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    payload=[]
    for event in requests['payload']:
        if event['id'] not in failures:continue
        changed=copy.deepcopy(event);count=len(changed['frame_paths'])
        changed['scene_context']+=(f' OUTPUT FORMAT CHECK: There are exactly {count} current-video images. '
            f'Valid evidence indices are integers 0 through {count-1} inclusive, in the order of their Frame index labels. '
            'Timestamps are seconds, NOT evidence indices. Separate NORMAL reference images have NO current-video index. '
            'For normal or uncertain, use valid indices 0 and 0 when no specific evidence pair applies. '
            f'For suspicious, cite two distinct valid current-video indices satisfying 0 <= start < end <= {count-1}. '
            'Do not change your visual conclusion to satisfy the schema; use uncertain if evidence is insufficient.')
        payload.append(changed)
    (out/'requests.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    partial=out/'partial.json';stored=json.loads(partial.read_text(encoding='utf-8')) if partial.exists() else {'replies':[],'telemetry':[],'elapsed_seconds':0}
    replies,telemetry=stored['replies'],stored['telemetry'];done={r['id'] for r in replies};todo=[r for r in payload if r['id'] not in done]
    client=WorkerClient();started=time.perf_counter()
    for offset in range(0,len(todo),8):
        response=client.post('/verify-batch',{'events':todo[offset:offset+8]},timeout=8*480+30)
        replies.extend(response['results']);telemetry.append(response.get('telemetry',{}))
        partial.write_text(json.dumps({'replies':replies,'telemetry':telemetry,'elapsed_seconds':stored['elapsed_seconds']+time.perf_counter()-started},indent=2),encoding='utf-8')
        print('SCHEMA RETRY',len(replies),'/',len(payload),flush=True)
    if {r['id'] for r in replies}!=failures or len(replies)!=len(failures):raise ValueError('Missing/duplicate retry')
    lookup={r['id']:r for r in replies}
    combined=[lookup.get(r['id'],r) for r in original['replies']]
    # Keep a failed retry unknown; never adopt a clamped raw answer.
    rows=[r for r in json.loads((ROOT/'datasets/ubnormal-open-multiscene-v1/manifest.json').read_text(encoding='utf-8'))['videos'] if r['split']=='validation']
    pred={r['id']:[] for r in rows}
    for reply in combined:
        video,number=reply['id'].rsplit('-unit',1);row=next(r for r in rows if r['id']==video)
        start=int(number)*3.;end=min(start+3,row['duration'])
        pred[video].append({'t_start':start,'t_end':end,'assessment':(reply.get('verdict') or {}).get('assessment','uncertain')})
    result={'status':'complete','protocol':protocol,'predictions':pred,'replies':combined,'retry_replies':replies,'telemetry':telemetry,
        'metrics':metrics(rows,pred),'confidence_intervals':bootstrap(rows,pred),'original_metrics':original['metrics'],
        'additional_http_requests':sum(r.get('calls',0) for r in replies),
        'total_http_requests':original['http_requests']+sum(r.get('calls',0) for r in replies),
        'retry_seconds':stored['elapsed_seconds']+time.perf_counter()-started,
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in requests['frame_hashes'].items()),
        'code_unchanged':file_hash(Path(__file__))==protocol['source_sha256'],'test_opened':False,'production_activation':False}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('SCHEMA RETRY COMPLETE',json.dumps(result['metrics']),flush=True)

if __name__=='__main__':main()
