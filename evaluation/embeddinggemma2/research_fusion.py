"""Fixed OR fusions and matched cross-scene calibration of the old Gemma gate."""
import copy
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from app.gemma2 import normality as n
from app.gemma2.store import file_hash
from .open_anomaly_eval import metrics,bootstrap
from .research_screening import select_threshold,predictions

ROOT=Path('scratch/event-retrieval')
PAIRS=[('gemma_normal_knn_smooth1','vpc_temporal_smooth1'),
       ('gemma_normal_knn_smooth1','prism_whitened_axes_smooth1'),
       ('gemma_normal_knn_smooth1','semantic_margin_smooth0'),
       ('vpc_three_memory_smooth1','prism_whitened_axes_smooth1')]

def main():
    output=ROOT/'research-fusion-20261015-v1';output.mkdir(parents=True,exist_ok=True)
    source=ROOT/'research-screening-20261015-v2/results.json'
    report=json.loads(source.read_text(encoding='utf-8'))
    old_path=ROOT/'open-anomaly-multiscene-v2/results.json'
    old=json.loads(old_path.read_text(encoding='utf-8'))
    rows=[r for r in json.loads((ROOT/'datasets/ubnormal-open-multiscene-v1/manifest.json').read_text(encoding='utf-8'))['videos'] if r['split']=='validation']
    protocol={'source_sha256':file_hash(source),'code_sha256':file_hash(Path(__file__)),
        'old_scores_sha256':file_hash(old_path),'pairs':PAIRS,'test_opened':False,
        'selection':'fixed OR; each gate calibrated only on other scenes; old gate same threshold grid as prior'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    methods={};started=time.perf_counter()
    for pair in PAIRS:
        data=[{key:(np.asarray(v['edges']),np.asarray(v['scores'])) for key,v in report['timeline_scores'][name].items()} for name in pair]
        pred={};folds=[]
        for scene in sorted({r['scene_id'] for r in rows}):
            calibration=[r for r in rows if r['scene_id']!=scene];held=[r for r in rows if r['scene_id']==scene]
            thresholds=[select_threshold(calibration,d)[0] for d in data]
            for row in held:
                edges=data[0][row['id']][0]
                if not np.array_equal(edges,data[1][row['id']][0]):raise ValueError('Timeline mismatch')
                selected=(data[0][row['id']][1]>=thresholds[0])|(data[1][row['id']][1]>=thresholds[1])
                pred[row['id']]=predictions(edges,selected.astype(float),.5)
            folds.append({'held_scene':scene,'thresholds':thresholds,'calibration_ids':[r['id'] for r in calibration]})
        methods[' OR '.join(pair)]={'metrics':metrics(rows,pred),'predictions':pred,'folds':folds,'confidence_intervals':bootstrap(rows,pred,repetitions=500)}
    # Re-evaluate original channel-separated scoring without re-encoding or altering its values.
    jobs={j['video_id']:j for j in old['verification_jobs']}
    db=sqlite3.connect(f"file:{(ROOT/'open-anomaly-multiscene-v2/store/events.sqlite3').resolve().as_posix()}?mode=ro",uri=True)
    timelines={r['id']:[json.loads(data) for data, in db.execute('select data from windows where namespace=? order by id',(jobs[r['id']]['namespace'],))] for r in rows}
    db.close()
    def old_predictions(selected_rows,novelty,margin):
        pred={}
        for row in selected_rows:
            evidence=copy.deepcopy(jobs[row['id']]['screening_scores'])
            for entry in evidence.values():
                for channel in entry.values():
                    for score in channel:
                        score['selected']=score['novelty']>=novelty or (score['similarity']>=.35 and score['margin']>=margin)
            units=n.segments(timelines[row['id']],{int(key):value for key,value in evidence.items()})
            pred[row['id']]=[{**u,'assessment':'suspicious'} for u in units]
        return pred
    pred={};folds=[]
    for scene in sorted({r['scene_id'] for r in rows}):
        calibration=[r for r in rows if r['scene_id']!=scene];held=[r for r in rows if r['scene_id']==scene];options=[]
        for novelty in (.05,.1,.2,.3,.5):
            for margin in (0,.03,.1):
                measured=metrics(calibration,old_predictions(calibration,novelty,margin))
                key=(measured['recall']>=.95,measured['recall'],-measured['fp'],-measured['positive_video_fraction'],novelty,margin)
                options.append((key,novelty,margin))
        _,novelty,margin=max(options)
        pred.update(old_predictions(held,novelty,margin));folds.append({'held_scene':scene,'novelty':novelty,'margin':margin})
    methods['original_gemma_cross_scene']={'metrics':metrics(rows,pred),'predictions':pred,'folds':folds,'confidence_intervals':bootstrap(rows,pred,repetitions=500)}
    result={'status':'complete','protocol':protocol,'methods':methods,'seconds':time.perf_counter()-started,'production_activation':False}
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,d in methods.items():print(name,json.dumps(d['metrics']),flush=True)

if __name__=='__main__':main()
