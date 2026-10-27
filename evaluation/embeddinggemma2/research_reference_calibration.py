"""Correct self-reference scale collapse using leave-one-normal-recording-out.

No re-extraction or anomaly labels in memory/scaling. Hold scene out for threshold
selection exactly as in research_screening. This is a declared second round.
"""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from app.gemma2.store import file_hash
from .research_screening import timeline,cross_scene

ROOT=Path('scratch/event-retrieval')
KEYS=('spatial','temporal','global')

def robust_scale(values,normal):
    center=np.median(normal,axis=0)
    scale=np.quantile(normal,.9,axis=0)-np.quantile(normal,.1,axis=0)
    # Constant channels carry no calibrated scale and must not be amplified by 1e6.
    available=scale>1e-8
    return np.where(available,(values-center)/np.where(available,scale,1),0),available

def main():
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(Path('scratch/research-baselines/VideoPatchCore/Memorization').resolve()))
    from functions.memory_func import get_coreset
    source=ROOT/'research-vpc-20261015-v1'
    out=ROOT/'research-vpc-loro-20261015-v1';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'datasets/ubnormal-open-multiscene-v1/manifest.json').read_text(encoding='utf-8'))['videos']
    refs=[r for r in rows if r['split']=='development'];val=[r for r in rows if r['split']=='validation']
    protocol={'source_sha256':file_hash(Path(__file__)),'feature_protocol_sha256':file_hash(source/'protocol.json'),
        'normal_scale':'Leave one complete normal recording out; no bank member scores itself',
        'constant_channels':'disabled instead of division by 1e-6','test_opened':False,
        'selection':'same cross-scene threshold protocol as previous round','reason':'Full small temporal/global normal banks gave zero in-sample distances; self-scores cannot estimate normal variability'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    loaded={r['id']:dict(np.load(source/(r['id']+'.npz'),allow_pickle=False)) for r in rows}
    started=time.perf_counter();data={};normal_stats={}
    def memory(items):
        result={}
        for key in KEYS:
            array=np.concatenate([loaded[r['id']][key] for r in items])
            if not len(array):raise ValueError('Empty normal memory')
            ids=get_coreset(torch.from_numpy(array).float(),l=min(512,len(array)),device='cpu').numpy()
            result[key]=array[ids]
        return result
    def score(row,bank):
        f=loaded[row['id']];result=[]
        for i in range(len(f['intervals'])):
            scores=[]
            for j,key in enumerate(KEYS):
                a,b=f['offsets'][i,j].astype(int);patches=f[key][a:b]
                scores.append(float(cdist(patches,bank[key]).min(1).max())**2 if len(patches) else 0.)
            result.append(scores)
        return np.asarray(result)
    for scene in sorted({r['scene_id'] for r in rows}):
        normal=[r for r in refs if r['scene_id']==scene]
        calibration=np.concatenate([score(row,memory([r for r in normal if r['recording_id']!=row['recording_id']])) for row in normal])
        bank=memory(normal)
        normal_stats[scene]={'recordings':[r['id'] for r in normal], 'loro_scores':calibration.tolist()}
        for row in val:
            if row['scene_id']!=scene:continue
            values,available=robust_scale(score(row,bank),calibration)
            normal_stats[scene]['available_channels']=available.tolist()
            variants={'vpc_loro_spatial':values[:,0],'vpc_loro_temporal':values[:,1],'vpc_loro_global':values[:,2],
                'vpc_loro_three_memory':.25*values[:,0]+.25*values[:,1]+.5*values[:,2]}
            for key,values in variants.items():
                edges,signal=timeline(values,loaded[row['id']]['intervals'],row['duration'])
                for sigma in (0,1):data.setdefault(key+f'_smooth{sigma}',{})[row['id']]=(edges,gaussian_filter1d(signal,sigma) if sigma else signal)
    methods={name:cross_scene(val,signals) for name,signals in data.items()}
    result={'status':'complete','protocol':protocol,'normal_calibration':normal_stats,'methods':methods,
        'seconds':time.perf_counter()-started,'timeline_scores':{name:{key:{'edges':edges.tolist(),'scores':scores.tolist()} for key,(edges,scores) in signals.items()} for name,signals in data.items()},'production_activation':False}
    (out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name,result in methods.items():print(name,json.dumps(result['metrics']),flush=True)

if __name__=='__main__':main()
