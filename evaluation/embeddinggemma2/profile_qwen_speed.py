"""Small unchanged-input diagnostic of Qwen lifecycle and server timings."""
import base64
import json
from pathlib import Path
import time

import requests

from app.inference.contracts import VerifyRequest
from inference_worker.vlm_service import VariantVLMService
from app.gemma2.store import file_hash


def completion_payload(event):
    capture={}
    service=VariantVLMService()
    def frames(paths,prompt,max_new_tokens,frame_labels=None):
        capture.update(paths=paths,prompt=prompt,max_tokens=max_new_tokens,labels=frame_labels)
        return json.dumps({'assessment':'normal','event_present':False,'start_frame_index':0,'end_frame_index':1,
            'confidence':0,'reason':'probe','observations':'probe','deviation':'none','alternative_explanation':'none','uncertainty':'none'})
    service._generate_frames=frames
    fields={key:event[key] for key in ('frame_paths','timestamps','scene_context','reference_frames','verification_mode')}
    request={'query':'Any observable abnormal activity in this scene','semantic_requirements':[],
             'max_new_tokens':512,'release_after':False,'vlm_backend':'caprl_qwen3vl_4b_q6',**fields}
    service.verify(VerifyRequest(**request))
    started=time.perf_counter();content=[]
    for label,path in zip(capture['labels'],capture['paths']):
        content.append({'type':'text','text':label})
        content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(Path(path).read_bytes()).decode('ascii')}})
    content.append({'type':'text','text':capture['prompt']})
    return request,{'messages':[{'role':'user','content':content}],'max_tokens':512,'temperature':0},time.perf_counter()-started


def main():
    root=Path('scratch/event-retrieval/qwen-speed-profile-20261015-v1')
    root.mkdir(parents=True,exist_ok=False)
    source=Path('scratch/event-retrieval/adaptive-frames-20261015-v2/requests.json')
    frozen=json.loads(source.read_text());events={e['id']:e for e in frozen['uniform']}
    ids=['normal_scene_3_scenario_4-0','abnormal_scene_3_scenario_3-0','abnormal_scene_4_scenario_7-1']
    jobs=requests.get('http://127.0.0.1:8002/api/v1/events/jobs',timeout=10).json()
    if any(j['status'] in ('running','queued','cancelling') for j in jobs):raise ValueError('Demo job active')
    private='http://127.0.0.1:8013'
    result={'source_sha256':file_hash(source),'script_sha256':file_hash(Path(__file__)),
        'scope':'Small diagnostic; first request cold through worker then three warm server requests including identical repeat. Not an accuracy or representative throughput benchmark.',
        'changes_to_model_prompt_frames':False,'requests':[]}
    try:
        requests.post('http://127.0.0.1:8012/unload',timeout=30).raise_for_status()
        requests.post(private+'/v1/models/unload',timeout=30).raise_for_status()
        req,_,_=completion_payload(events[ids[0]])
        started=time.perf_counter();response=requests.post(private+'/v1/vlm/verify',json=req,timeout=240);response.raise_for_status()
        result['cold_private_wall_seconds']=time.perf_counter()-started;result['cold_private_reply']=response.json()
        result['cold_health']=requests.get(private+'/health',timeout=10).json()
        for identifier in ids:
            _,payload,prep=completion_payload(events[identifier])
            started=time.perf_counter();response=requests.post('http://127.0.0.1:18101/v1/chat/completions',json=payload,timeout=240);response.raise_for_status()
            data=response.json();result['requests'].append({'id':identifier,'same_input_repeat':identifier==ids[0],
                'wall_seconds':time.perf_counter()-started,'base64_preparation_seconds':prep,
                'timings':data.get('timings'),'usage':data.get('usage'),'choices':data.get('choices')})
            (root/'partial.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(identifier,json.dumps({k:result['requests'][-1][k] for k in ('wall_seconds','timings','usage')}),flush=True)
    finally:
        started=time.perf_counter();response=requests.post(private+'/v1/models/unload',timeout=30);response.raise_for_status()
        result['unload_seconds']=time.perf_counter()-started
        result['final_health']=requests.get(private+'/health',timeout=10).json()
        (root/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PROFILE COMPLETE',result['cold_private_wall_seconds'],result['unload_seconds'],flush=True)


if __name__=='__main__':main()
