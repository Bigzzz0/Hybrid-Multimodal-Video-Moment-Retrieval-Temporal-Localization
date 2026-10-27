"""Exploratory cached-vector prompt ablation; no production policy changes.

Outputs model decisions, not accuracy. No Qwen outputs are ground truth.
Run from the repository root with PYTHONPATH including backend.
"""
import copy,json,sqlite3,time
from pathlib import Path
import numpy as np
import requests
from app.gemma2.ranking import EVENTS, screen
from app.gemma2.store import vectors
root=Path('scratch/event-retrieval/gemma-precision'); root.mkdir(parents=True,exist_ok=True)
banks={'baseline':copy.deepcopy(EVENTS),'hard_negatives':copy.deepcopy(EVENTS)}
banks['hard_negatives']['pushing_punching']['negative'] += [
 'People standing or sitting close together without pushing or punching each other',
 'คนยืนหรือนั่งใกล้กันโดยไม่มีการผลักหรือชกกัน',
 'A person resting or lying on the floor while other people stand nearby without attacking',
 'A person waving or pointing an arm without hitting another person',
 'A person gently supporting another person to stand up without a forceful shove',
 'A person pushing or striking a gate or door, without hitting another person',
 'คนผลักประตูหรือกำแพง ไม่ได้ผลักหรือชกคนอื่น']
banks['hard_negatives']['falling']['negative'] += [
 'A person already lying on the floor, without a transition from standing to falling',
 'คนนอนอยู่บนพื้นอยู่แล้ว ไม่มีการเสียการทรงตัวล้มลง',
 'A person sitting or kneeling down slowly and deliberately',
 'A person bending over and then standing upright again without falling',
 'A person intentionally lying down to rest or exercise']
jobs=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
assert not any(j['status'] in ('running','queued','cancelling') for j in jobs),'Jobs active; experiment not started'
texts=list(dict.fromkeys(t for bank in banks.values() for kind in ('pushing_punching','falling') for role in ('positive','negative') for t in bank[kind][role]))
before=time.perf_counter()
r=requests.post('http://127.0.0.1:8012/embed',json={'modality':'query','backend':'gemma2','dimension':768,'texts':texts},timeout=240);r.raise_for_status();response=r.json()
assert response['revision']=='914f7f89142e33e77833254d9c9b90c3cef7303b'
emb=vectors(response['embeddings'],768); lookup=dict(zip(texts,emb));result={'purpose':'exploratory prompt ablation, not accuracy or validation','banks':banks,'model_revision':response['revision'],'encode_http_seconds':time.perf_counter()-before,'telemetry':response.get('telemetry'),'videos':[]}
conn=sqlite3.connect('file:backend/data/gemma2/events.sqlite3?mode=ro',uri=True)
for video in ['ut-punch-demo','ur-fall-demo','ur-normal-demo','ucf-long-fighting011','ucf-long-normal634','benchmark_classroom']:
 ns=conn.execute('SELECT namespace FROM active WHERE video_id=?',(video,)).fetchone()
 if ns is None:continue
 ns=ns[0]
 config,status=conn.execute('SELECT config,status FROM indexes WHERE namespace=?',(ns,)).fetchone(); config=json.loads(config)
 assert status=='ready' and config['revision']==response['revision'] and config['dimension']==768, 'Index/model mismatch'
 timeline={i:json.loads(d) for i,d in conn.execute('SELECT id,data FROM windows WHERE namespace=?',(ns,))}; channels={}
 for modality in ('image','video'):
  rows=list(conn.execute('SELECT window_id,vector FROM embeddings WHERE namespace=? AND modality=? ORDER BY window_id,slot',(ns,modality)))
  if rows: channels[modality]=(np.array([i for i,_ in rows]),vectors([np.frombuffer(b,dtype=np.float32) for _,b in rows],768))
 row={'video_id':video,'namespace':ns,'windows':len(timeline),'variants':{}}
 for name,bank in banks.items():
  prompts={kind:(np.stack([lookup[t] for t in bank[kind]['positive']]),np.stack([lookup[t] for t in bank[kind]['negative']])) for kind in ('pushing_punching','falling')}
  hits=screen(channels,prompts,['pushing_punching','falling'],.35,.02)
  row['variants'][name]={kind:[{**h,'t_start':timeline[h['window_id']]['t_start'],'t_end':timeline[h['window_id']]['t_end']} for h in hits if h['kind']==kind] for kind in ('pushing_punching','falling')}
 result['videos'].append(row)
 print(video, {name:{kind:len(hits) for kind,hits in kinds.items()} for name,kinds in row['variants'].items()},flush=True)
conn.close()
(root/f'prompt-ablation-{time.time_ns()}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
