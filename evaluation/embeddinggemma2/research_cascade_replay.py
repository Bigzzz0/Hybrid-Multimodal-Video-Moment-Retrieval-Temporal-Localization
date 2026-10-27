"""Replay fixed-grid Qwen replies under label-blind candidate gates.

No new calls and no measured runtime savings claimed. Identical stored Qwen
frames/references/prompts; a unit passes when any candidate time overlaps it.
"""
import json
from pathlib import Path
from app.gemma2.store import file_hash
from .open_anomaly_eval import metrics,bootstrap

ROOT=Path('scratch/event-retrieval')
SOURCES=['research-screening-20261015-v2','research-fusion-20261015-v1','research-vpc-loro-20261015-v1']

def overlaps(unit,candidate):
    return max(unit['t_start'],candidate['t_start']) < min(unit['t_end'],candidate['t_end'])

def main():
    out=ROOT/'research-cascade-replay-20261015-v1';out.mkdir(parents=True,exist_ok=True)
    protocol={'code_sha256':file_hash(Path(__file__)),'gate':'any strict temporal overlap, boundary touching does not count',
        'test_opened':False,'production_activation':False,'kind':'offline_replay_of_actual_Qwen_replies',
        'not_measured_pipeline_latency_or_actual_saved_calls':True}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    qwen_path=ROOT/'research-qwen-units-20261015-v1/results.json'
    qwen=json.loads(qwen_path.read_text(encoding='utf-8'))
    rows=[r for r in json.loads((ROOT/'datasets/ubnormal-open-multiscene-v1/manifest.json').read_text(encoding='utf-8'))['videos'] if r['split']=='validation']
    results={};total=sum(len(p) for p in qwen['predictions'].values())
    for source in SOURCES:
        report=json.loads((ROOT/source/'results.json').read_text(encoding='utf-8'))
        for method,arm in report['methods'].items():
            pred={r['id']:[unit for unit in qwen['predictions'][r['id']] if any(overlaps(unit,c) for c in arm['predictions'][r['id']])] for r in rows}
            count=sum(len(p) for p in pred.values())
            results[method]={'metrics':metrics(rows,pred),'confidence_intervals':bootstrap(rows,pred,repetitions=500),
                'predictions':pred,'would_verify_units':count,'baseline_units':total,
                'would_reduce_units_fraction':1-count/total, 'not_actual_requests_or_time_saved':True,'screening_source':source}
    result={'status':'complete','protocol':protocol,'qwen_source_sha256':file_hash(qwen_path),'methods':results}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    best=max(results,key=lambda k:(results[k]['metrics']['recall']>=.95,results[k]['metrics']['recall'],
        -results[k]['metrics']['fp'],results[k]['would_reduce_units_fraction']))
    print('Replay best by recall then FP:',best,json.dumps(results[best]['metrics']),flush=True)

if __name__=='__main__':main()
