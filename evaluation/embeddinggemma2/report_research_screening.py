"""Collect measured research trials without conflating calibration and old demos."""
import csv
from collections import Counter
import json
from pathlib import Path

ROOT=Path('scratch/event-retrieval')

def main():
    rows=[]
    sources=[('research-screening-20261015-v2','cross-scene'),('research-fusion-20261015-v1','cross-scene'),
             ('research-vpc-loro-20261015-v1','cross-scene')]
    for folder,protocol in sources:
        report=json.loads((ROOT/folder/'results.json').read_text(encoding='utf-8'))
        for name,result in report['methods'].items():rows.append({'method':name,'protocol':protocol,'source':folder,**result['metrics']})
    for folder,name,protocol in [('open-anomaly-context-ablation-v1','Qwen camera-normality 12s (historical)','historical validation'),
                                  ('research-qwen-units-20261015-v1','Qwen explicit-target 3s','new full validation')]:
        report=json.loads((ROOT/folder/'results.json').read_text(encoding='utf-8'))
        rows.append({'method':name,'protocol':protocol,'source':folder,**report['metrics']})
    retry_path=ROOT/'research-qwen-schema-retry-20261015-v1/results.json'
    retry=json.loads(retry_path.read_text(encoding='utf-8')) if retry_path.exists() else None
    if retry:
        rows.append({'method':'Qwen explicit-target 3s + one schema retry','protocol':'new full validation',
                     'source':retry_path.parent.name,**retry['metrics']})
    output=Path('C:/Users/User/Downloads/Video Event Retrieval/output/research-screening-20261015')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    vpc=json.loads((ROOT/'research-vpc-20261015-v1/resources.json').read_text(encoding='utf-8'))
    qwen=json.loads((ROOT/'research-qwen-units-20261015-v1/results.json').read_text(encoding='utf-8'))
    summary={'comparisons':rows,'vpc_resources':vpc,'qwen_resources':qwen['telemetry'],
             'qwen_requests':qwen['http_requests'],'qwen_seconds':qwen['elapsed_seconds'],
             'test_opened':False,'production_activation':False}
    summary['qwen_technical_errors']=dict(Counter(r.get('error') for r in qwen['replies'] if r.get('error')))
    summary['qwen_code_unchanged']=qwen['code_unchanged']
    summary['qwen_frames_unchanged']=qwen['frames_unchanged']
    summary['schema_retry']=retry
    def ordering(row):
        return (row['recall']>=.95,row['recall'],-row['fp'],-row['positive_video_fraction'])
    best_screen=max((r for r in rows if r['protocol']=='cross-scene'),key=ordering)
    best_qwen=max((r for r in rows if r['protocol']!='cross-scene'),key=ordering)
    summary['best_cross_scene_screening']=best_screen
    summary['best_observed_qwen_validation']=best_qwen
    replay_path=ROOT/'research-cascade-replay-20261015-v1/results.json'
    replay=json.loads(replay_path.read_text(encoding='utf-8'))
    summary['cascade_replay']=replay
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# ผลทดลองวิธีตรวจความผิดปกติจากงานวิจัย — 15 ตุลาคม 2026','',
        'รันจริงบน RTX 5070 ในบร้านช์ทดลอง ใช้ UBnormal ภาพจำลอง 16 คลิป 3 ฉาก รวม 210.93 วินาที',
        'development ปกติ 6 คลิป; validation 10 คลิป มีเหตุ 7 ช่วง และช่วงปกติรวม 60.13 วินาที ไม่มีเสียง',
        'ไม่เปิด official test ไม่ฝึกน้ำหนัก ไม่เปลี่ยน demo defaults และไม่ merge main','',
        '## ข้อสรุป','',
        f"คัดกรองเมื่อกันฉากออกจาก threshold: {best_screen['method']} จับได้ {best_screen['tp']}/7, FP {best_screen['fp']}; คัดเวลาวิดีโอ {best_screen['positive_video_fraction']:.1%}",
        f"ผล Qwen ที่สังเกตดีที่สุดระหว่างสองรอบ: {best_qwen['method']} จับได้ {best_qwen['tp']}/7, FP {best_qwen['fp']}, uncertain/unverified {best_qwen['uncertain_units']} หน่วย",
        'สองบรรทัดนี้เป็นคนละ protocol ห้ามนำมาอ้างว่าเป็นผล Gemma/VPC → Qwen แบบ end-to-end ที่รันแล้ว',
        'ไม่พบหลักฐานพอให้เปลี่ยน demo defaults หรืออ้างว่าผ่านเป้าหมายใช้งานจริง','',
        '## วิธีเปรียบเทียบ','',
        'วิธีคัดกรองใหม่กันหนึ่งฉากไว้ประเมิน หมุนครบ 3 ฉาก เลือก threshold จาก validation อีกสองฉากเท่านั้น',
        'คลังปกติของแต่ละฉากมาจาก development ของฉากนั้น ไม่ใช้ validation เติมคลัง',
        'เลือก threshold ให้ recall >=95% ก่อน จากนั้น recall สูงกว่า, FP ต่ำกว่า และระยะเวลาที่คัดน้อยกว่า',
        'ใช้ IoU >=0.3 จับคู่เหตุหนึ่งต่อหนึ่ง ไม่ใช้ frame AUC แทน event recall',
        'ยังเป็น exploratory validation: ชุดนี้เคยถูกวิเคราะห์มาก่อน และเลือกวิธีจากผล pooled validation; ไม่ใช่ independent test','',
        '## ผลทั้งหมด','',
        '| วิธี | Protocol | TP/7 | FP | Recall | Precision | รอตรวจ | เวลาวิดีโอที่คัด |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['protocol']} | {row['tp']} | {row['fp']} | {row['recall']:.1%} | {row['precision']:.1%} | {row['uncertain_units']} | {row['positive_video_fraction']:.1%} |")
    lines+=['','## Replay การคัดก่อน Qwen 3s ก่อน schema retry','',
        'ใช้คำตอบ Qwen ที่รันจริงบนกริดเดียวกัน เลือกหน่วยที่ทับกับช่วงคัดโดยไม่ใช้ label เปลี่ยนเฟรมหรือคำสั่ง',
        'ตาราง replay นี้ใช้ผลรอบก่อน schema retry ซึ่งมี uncertain/unverified 19 หน่วย ไม่ใช่ผล pipeline ที่แก้ schema แล้ว',
        'ยังไม่ได้วัดเวลา pipeline ที่คัดก่อนจริง จำนวนด้านล่างคือจำนวนหน่วยที่น่าจะต้องตรวจ ไม่ใช่ calls/เวลาที่ประหยัดแล้ว',
        '', '| ตัวคัดก่อน Qwen 3s | TP/7 | FP | uncertain | หน่วยที่ต้องตรวจโดยประมาณ /57 |',
        '|---|---:|---:|---:|---:|']
    for name,arm in replay['methods'].items():
        m=arm['metrics']
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['uncertain_units']} | {arm['would_verify_units']} |")
    lines+=['','## จุดที่ต่างจากงานต้นฉบับ','',
        '- PRISM: ใช้สูตร text covariance/whitened anomaly axes จาก source ที่ pin revision แต่ใช้ Gemma แทน Qwen/PE พร้อม generic text bank 24 ข้อ และ sampling ของเรา',
        '- Flashback-style: ใช้ top-10 retrieval, anomaly penalty .95 และ temperature .01 บน text bank เล็ก ไม่ใช่การทำซ้ำคลังหนึ่งล้านคำอธิบายของบทความ',
        '- VideoPatchCore: ใช้ RN101 และ feature/coreset functions จากต้นฉบับจริง แต่แทน YOLOv5l ด้วย SSDLite CPU, ใช้ 2FPS/4เฟรม/stride1s, memory cap512 และ normalize จาก development',
        '- รอบ VPC แรกพบ in-sample normal distances เป็นศูนย์ใน memory ที่เก็บครบ จึงเก็บผลเดิมและทำรอบ LORO แยก ไม่อ้างว่า scaling เดิมมีความน่าเชื่อถือ',
        '- LORO ใช้ normal recording หนึ่งตรวจเทียบอีก recording เพื่อประมาณความแปรปรวน; ช่องที่ไม่มีความแปรปรวนถูกปิด ไม่ขยายคะแนนด้วยการหาร 1e-6',
        '- Qwen 3s เปลี่ยนทั้งความยาวหน่วยและคำสั่งระบุ target interval ชัดเจน จึงไม่อ้างว่าพิสูจน์ผลของ sampling เพียงปัจจัยเดียว',
        '- ยังไม่ได้รัน official AnomalyRuler แบบ induction/deduction เต็ม, Flashback เต็ม หรือ pretrained VadCLIP; ห้ามใช้ผล adaptations ตัดสินว่างานต้นฉบับแพ้','',
        '## เวลาและทรัพยากร','',
        f"VPC feature extraction: {vpc['feature_seconds']:.2f}s, load {vpc['load_seconds']:.2f}s; อ่าน JPEG ที่เคยเตรียมไว้ ไม่รวม decode MP4/ดาวน์โหลดน้ำหนัก",
        f"VPC PyTorch allocated peak {vpc['resources']['torch_allocated_peak_bytes']/2**30:.3f} GiB, reserved {vpc['resources']['torch_reserved_peak_bytes']/2**30:.3f} GiB",
        f"VPC sampled process RSS {vpc['resources']['rss_peak_bytes']/2**30:.3f} GiB; sampled whole-card VRAM {vpc['resources']['whole_gpu_sampled_peak_bytes']/2**30:.3f} GiB",
        f"Qwen 3s: {qwen['http_requests']} private HTTP requests; elapsed {qwen['elapsed_seconds']:.2f}s หลังเตรียมเฟรม รวมโหลด/ปล่อยในแต่ละ batch",
        'Gemma media embeddings ใช้ cache จากรอบเดิม เวลา scoring ในผลใหม่จึงไม่ใช่ end-to-end indexing time',
        'VRAM ทั้งการ์ดรวมโปรแกรมอื่น วัดทุก .5s; Qwen ใช้ llama.cpp ไม่ใช่ PyTorch allocator ของ broker',
        'ไม่เปรียบเทียบ inference-only กับ cold indexing แล้วอ้าง speedup','',
        '## ข้อจำกัดและหลักฐาน','',
        ('Schema retry: รันซ้ำเฉพาะ technical unverified โดยเพิ่ม valid index range, ใช้ภาพและ parser เดิม; '
         f"เพิ่ม {retry['additional_http_requests']} calls, {retry['retry_seconds']:.2f}s; รวม {retry['total_http_requests']} calls") if retry else 'Schema retry ยังไม่มีผลครบ',
        'ข้อมูลยังมีเพียง 7 เหตุและ normal exposure ประมาณหนึ่งนาที ยังรับรอง <=2 false alerts/hour ไม่ได้',
        'คำตอบ Qwen ที่ผิด schema/เลขเฟรม หรือไม่ชัดเจน คง uncertain แยกจากผลปกติ ไม่ถือเป็นการยืนยันถูก',
        'รายการผิดพลาด Qwen: '+json.dumps(summary['qwen_technical_errors'],ensure_ascii=False),
        'ตรวจ hash ระหว่างรัน Qwen: code_unchanged='+str(qwen['code_unchanged'])+', frames_unchanged='+str(qwen['frames_unchanged']),
        'เก็บ predictions รายคลิป, fold thresholds, raw scores, source/frame/manifest hashes และคำตอบ Qwen ครบใน scratch/event-retrieval/',
        'ชุดทดสอบที่เกี่ยวข้องผ่าน 28 ข้อ (หลังเปลี่ยน pytest basetemp เป็น workspace เนื่องจาก Temp ของ Windows ไม่มีสิทธิ์)',
        '', '## ทำซ้ำ','', '```powershell',
        '$env:PYTHONPATH="$PWD/backend;$PWD"',
        'python -m evaluation.embeddinggemma2.research_vpc --output <fresh-vpc-dir>',
        'python -m evaluation.embeddinggemma2.research_screening --output <fresh-results-dir> --vpc-features <vpc-dir>',
        'python -m evaluation.embeddinggemma2.research_reference_calibration',
        'python -m evaluation.embeddinggemma2.research_fusion',
        'python -m evaluation.embeddinggemma2.research_qwen_units --output <fresh-qwen-dir>',
        '```','',
        'fusion/LORO/report scripts อ้างอิงผลของรอบนี้ตามชื่อ directory ที่ระบุใน source; ใช้ fresh output เพื่อเก็บประวัติ',
        '', '## แหล่งต้นฉบับ','',
        '- https://github.com/SkiddieAhn/Paper-VideoPatchCore (eefdc8d7a84f8eb06a561e3c42ab26a1b944f9b6)',
        '- https://github.com/ytC2026/ICML2026-PRISM (27c4533a45670f3bceddcf875de39a23dd09bc65)',
        '- https://arxiv.org/abs/2505.15205',
        '- https://github.com/openai/CLIP (d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)',
        '- https://github.com/lilygeorgescu/UBnormal']
    text='\n'.join(lines)+'\n'
    (output/'results-th.md').write_text(text,encoding='utf-8')
    Path('docs/research-screening-results-20261015.md').write_text(text,encoding='utf-8')
    print(output.resolve())

if __name__=='__main__':main()
