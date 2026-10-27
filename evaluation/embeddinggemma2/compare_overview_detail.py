"""Four-arm spatial/normal-motion ablation with frozen Gemma candidate units."""
import argparse
import copy
import json
from pathlib import Path
import time

from app.gemma2.config import IndexConfig
from app.gemma2.store import Store,file_hash
from .overview_detail import prepare_detail,normal_storyboard,DETAIL_RULE,NORMAL_RULE
from .person_region import SSDLitePersonDetector
from .evaluate_frozen_scenes import verify_phase
from .compare_matched_gemma_gate import predictions
from .open_anomaly_eval import audit,metrics,match_events,union
from .research_qwen_residency import idle


def matched_labels(rows,pred):
    return {(row['id'],i) for row in rows for i in match_events(
        union([(p['t_start'],p['t_end']) for p in pred[row['id']] if p['assessment']=='suspicious']),
        [(e['t_start'],e['t_end']) for e in row['events']])}


def qualifies(base,new,baseline_matches,new_matches):
    return (baseline_matches<=new_matches and new['uncertain_units']<=base['uncertain_units'] and
            ((new['tp']>base['tp'] and new['fp']<=base['fp']) or (new['tp']==base['tp'] and new['fp']<base['fp'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--payload-key',default='selected')
    parser.add_argument('--manifest',required=True);parser.add_argument('--reference-store',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--arms',nargs='+',choices=['control','detail','normal_motion','combined'],default=['control','detail','normal_motion','combined'])
    args=parser.parse_args()
    if 'control' not in args.arms or len(set(args.arms))!=len(args.arms):raise ValueError('Unique arms including control required')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);source=Path(args.source)
    frozen=json.loads((source/'requests.json').read_text());baseline=frozen[args.payload_key];intervals=frozen['intervals']
    for path,sha in frozen['frame_hashes'].items():
        if file_hash(Path(path))!=sha:raise ValueError('Frozen source frame changed')
    manifest_path=Path(args.manifest);rows=json.loads(manifest_path.read_text())['videos'];audit(rows)
    validation=[r for r in rows if r['split']=='validation'];development={r['id'] for r in rows if r['split']=='development' and not r['events']}
    for event in baseline:
        if any(ref['video_id'] not in development for ref in event['reference_frames']):raise ValueError('Reference split leak')
    files=[Path(__file__),Path('evaluation/embeddinggemma2/overview_detail.py'),Path('evaluation/embeddinggemma2/person_region.py'),Path('backend/inference_worker/vlm_service.py'),Path('evaluation/embeddinggemma2/worker.py')]
    protocol={'version':'overview-normal-motion-v1','source_sha256':file_hash(source/'requests.json'),'manifest_sha256':file_hash(manifest_path),
        'arms':args.arms,'code_hashes':{str(p):file_hash(p) for p in files},
        'detector':'CPU FP32 SSDLite320 COCO_V1, person >=.5, full frame + 2x2 tiles covering .6 width/height; NMS IoU .5',
        'regions':'fixed per unit; up to two median-x groups; .2 context on each side; min width .12 and height .25 of source; fallback without two detected frames',
        'image_budget':'same current-frame count/timestamps and original canvas; full overview plus one/two fixed detail panels; actual visual tokens not measured',
        'normal_motion':'four frames from the same verified-normal window as original reference, same canvas, 2x2 chronological storyboard; retain original single reference if fewer than four distinct frames; maximum two references unchanged',
        'detail_instruction':DETAIL_RULE,'normal_instruction':NORMAL_RULE,
        'promotion':'retain every control TP, increase TP without increasing FP OR same TP and fewer FP; no increase in uncertain units',
        'followup_holdout_scenes':[22,24],'threshold_tuning':False,'test_opened':False,
        'limitations':'Previously inspected validation becomes regression/development. No independent test claim. Person detail may miss nonhuman anomalies; overview retained.',
        'timing':'Sequential arm phases, no randomized timing comparison. Retain Qwen in each phase, unload between arms. CPU preparation separate.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    idle();store=Store(args.reference_store);config=IndexConfig();started=time.perf_counter()
    needed_detail=any(a in args.arms for a in ('detail','combined'));needed_story=any(a in args.arms for a in ('normal_motion','combined'))
    detector=SSDLitePersonDetector('scratch/event-retrieval/person-region-ssdlite-dev/model-cache') if needed_detail else None
    variants={a:copy.deepcopy(baseline) for a in args.arms};regions={};stories={};det_cache={}
    for index,event in enumerate(baseline):
        detail_paths=event['frame_paths']
        if needed_detail:detail_paths,regions[event['id']]=prepare_detail(event,root/'detail-frames'/event['id'],detector,det_cache)
        new_refs=[]
        if needed_story:
            for ref in event['reference_frames']:
                if ref['path'] not in stories:
                    revised,provenance=normal_storyboard(ref,store,config,development,root/'normal-storyboards')
                    stories[ref['path']]={'reference':revised,'provenance':provenance}
                new_refs.append(stories[ref['path']]['reference'])
        for name in args.arms:
            changed=variants[name][index]
            if name in ('detail','combined'):
                changed['frame_paths']=detail_paths;changed['scene_context']+=DETAIL_RULE
            if name in ('normal_motion','combined'):
                changed['reference_frames']=new_refs;changed['scene_context']+=NORMAL_RULE
            if changed['timestamps']!=event['timestamps'] or len(changed['frame_paths'])!=len(event['frame_paths']) or len(changed['reference_frames'])!=len(event['reference_frames']):raise ValueError('Budget or temporal grid changed')
        print('PREPARED',index+1,'/',len(baseline),flush=True)
    prep={'seconds':time.perf_counter()-started,'detector':detector.metadata if detector else None,'regions':regions,'normal_storyboards':stories}
    (root/'preparation.json').write_text(json.dumps(prep,indent=2),encoding='utf-8')
    hashes={p:file_hash(Path(p)) for events in variants.values() for e in events for p in e['frame_paths']+[r['path'] for r in e['reference_frames']]}
    (root/'requests.json').write_text(json.dumps({'arms':variants,'intervals':intervals,'frame_hashes':hashes},indent=2),encoding='utf-8')
    results={}
    for name in args.arms:
        phase=verify_phase(variants[name],root/(name+'-partial.json'))
        pred=predictions(validation,phase['replies'],intervals)
        results[name]={'phase':phase,'predictions':pred,'metrics':metrics(validation,pred),'matched_events':sorted(matched_labels(validation,pred))}
        (root/'arms-partial.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print('ARM COMPLETE',name,results[name]['metrics'],flush=True)
    control=results['control'];promotions={name:qualifies(control['metrics'],value['metrics'],set(map(tuple,control['matched_events'])),set(map(tuple,value['matched_events']))) for name,value in results.items() if name!='control'}
    result={'status':'complete','protocol':protocol,'arms':results,'preparation_seconds':prep['seconds'],'promotions':promotions,
        'code_unchanged':all(file_hash(Path(p))==sha for p,sha in protocol['code_hashes'].items()),
        'frames_unchanged':all(file_hash(Path(p))==sha for p,sha in hashes.items()),'test_opened':False,'production_activation':False}
    (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({'metrics':{k:v['metrics'] for k,v in results.items()},'promotions':promotions}),flush=True)


if __name__=='__main__':main()
