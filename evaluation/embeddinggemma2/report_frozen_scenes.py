"""Local research report/gallery from completed frozen-scene predictions."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil

from .open_anomaly_eval import metrics
from app.gemma2.store import file_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--selection-lock',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=json.loads((run/'results.json').read_text(encoding='utf-8'))
    if result['status']!='complete' or not result['code_unchanged'] or not result['frames_unchanged']:raise ValueError('Incomplete/changed experiment')
    if file_hash(Path(args.manifest))!=result['protocol']['manifest_sha256'] or file_hash(Path(args.selection_lock))!=result['protocol']['selection_lock_sha256']:
        raise ValueError('Report manifest/selection lock differs from experiment')
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'));all_rows=manifest['videos'];rows=[r for r in all_rows if r['split']=='validation']
    payload=json.loads((run/'requests.json').read_text(encoding='utf-8'));intervals=payload['intervals']
    selected={e['id'] for e in payload['selected']};replies={r['id']:r for phase in ('selected_phase','skipped_audit') for r in result[phase]['replies']}
    by_video={r['id']:[] for r in rows}
    for event in payload['all_units']:
        by_video[intervals[event['id']]['video_id']].append(event)
    labels={'gemma_windows':'Gemma: หน้าต่างต้นทาง','selected_grid':'Gemma: หน่วยตรวจ 12s','gemma_qwen':'Gemma → Qwen','all_qwen_diagnostic':'ตรวจช่วงที่ข้ามเพิ่ม (diagnostic)'}
    lines=['# ผลทดสอบฉากใหม่ — UBnormal','','คงวิธีและเกณฑ์เดิมก่อนเปิดผลชุดใหม่ ไม่ฝึกน้ำหนักและไม่เลือก threshold ใหม่','','## ข้อมูล','',
        f"- วิดีโอใหม่ {len(all_rows)} คลิป จากฉาก 6/8/14/16; training-normal {len(all_rows)-len(rows)} คลิป ({len({r['recording_id'] for r in all_rows if r['split']=='development'})} การบันทึกต้นฉบับ) สำหรับสร้างคลังปกติ",
        f"- ประเมิน official validation {len(rows)} คลิป มี {sum(len(r['events']) for r in rows)} ช่วงเหตุและ {sum(not r['events'] for r in rows)} คลิปปกติ รวม {sum(r['duration'] for r in rows):.2f}s; normal-only {sum(r['duration'] for r in rows if not r['events']):.2f}s",
        '- เป็นภาพจำลองของผู้จัดทำ UBnormal ไม่ใช่กล้องจริง ไม่มีการเปิด official test; แยกตาม recording โดย fog variant ไม่ถูกแยกข้ามชุด',
        '- ฉาก8 มี training original และ fog ของการบันทึกเดียวกัน จึงไม่ใช่ตัวอย่างปกติอิสระสองการบันทึก; context เป็น generic pedestrian-area เดิม คลังอ้างอิงแยกฉาก',
        '- UCSD แหล่งทางการเชื่อมต่อหมดเวลา ส่วน ShanghaiTech ได้หน้า OneDrive แต่ยังไม่มีไฟล์ที่ตรวจสอบได้ จึงไม่มีผลกล้องจริงที่อ้างจากสองชุดนั้นในรายงานนี้',
        '- แหล่งข้อมูลและสิทธิ์: [UBnormal official repository](https://github.com/lilygeorgescu/UBnormal), CC-BY-NC-ND-4.0; source revision และ SHA256 บันทึกใน manifest/selection และตรวจ CRC ขณะอ่านไฟล์จาก archive',
        '', '## ผลหลัก (temporal IoU ≥0.3, one-to-one matching)','',
        '| ขั้นตอน | TP | FP | FN | Recall | Precision | Uncertain units |','|---|---:|---:|---:|---:|---:|---:|']
    pct=lambda x:'—' if x is None else f'{x:.1%}'
    for arm,m in result['metrics'].items():
        lines.append(f"| {labels[arm]} | {m['tp']} | {m['fp']} | {m['fn']} | {pct(m['recall'])} | {pct(m['precision'])} | {m['uncertain_units']} |")
    lines += ['',f"ส่งเข้า Qwen {result['selected_units']}/{result['total_grid_units']} หน่วย ลดคำขอรอบหลัก {pct(result['qwen_call_reduction'])}; ตรวจหน่วยที่ถูกข้ามแยกภายหลังเพื่อวิเคราะห์ Gemma เท่านั้น ไม่ปะปนเวลา audit กับรอบหลัก",'',
        '## แยกตามฉาก (วิธีหลัก Gemma → Qwen)','','| ฉาก | TP | FP | FN | Recall | Precision |','|---|---:|---:|---:|---:|---:|']
    for scene,m in result['per_scene'].items():lines.append(f"| {scene} | {m['tp']} | {m['fp']} | {m['fn']} | {pct(m['recall'])} | {pct(m['precision'])} |")
    lines += ['', '## กรณีที่ผิดและข้อสรุป','',
        '- เหตุที่พลาดทั้งสามคลิป (abnormal_scene_6_scenario_6, abnormal_scene_8_scenario_9, abnormal_scene_16_scenario_1) ถูกส่งถึง Qwen แล้ว และ Qwen ตอบ normal ทุกหน่วยที่ตรวจ ไม่ใช่กรณีพบเหตุแล้วเสียจากการ union อย่างในการทดลองก่อน',
        '- normal_scene_16_scenario_2 แจ้งผิดโดยอ้างหมวก/หนวดและลักษณะคนงาน ส่วน normal_scene_16_scenario_3 อ้างชุดสีแดงคล้าย superhero แสดงว่าคำสั่งห้ามใช้รูปลักษณ์อย่างเดียวยังไม่ทำให้โมเดลปฏิบัติตามได้สม่ำเสมอ',
        '- การตรวจช่วงที่ Gemma ข้ามเพิ่มให้ผลรวมเท่าเดิม จึงยังไม่พบเหตุหายเพิ่มจาก gate ในชุดนี้ แต่ gate ยังลดงานได้น้อยมาก',
        '- ผล85.7%จากชุดเดิมจึงไม่ควรนำไปกล่าวเป็นความแม่นทั่วไป ชุดใหม่ได้ Recall50%/Precision60%; ยังต้องปรับการเห็นและตีความการกระทำของ Qwen พร้อมทดสอบข้อมูลที่กันไว้ใหม่ การเปลี่ยน threshold ให้เข้ากับชุดนี้ภายหลังจะไม่ถือเป็นผลทดสอบอิสระ']
    index_jobs=result['index_jobs'];resources=[t['qwen_resources'] for t in result['selected_phase']['telemetry'] if t.get('qwen_resources')]
    lines += ['', '## เวลาและทรัพยากร','',
        f"- สร้างดัชนีครั้งแรกใน process นี้ {result['index_seconds_this_process']:.2f}s; ผลรวม decode ที่ jobs บันทึก {sum(j.get('decode_seconds',0) for j in index_jobs):.2f}s และ encode {sum(j.get('encode_seconds',0) for j in index_jobs):.2f}s",
        f"- สร้างคลัง/encode query/คัดช่วง/เตรียมภาพตรวจ {result['query_gate_frame_preparation_seconds']:.2f}s",
        f"- Qwen รอบหลัก {result['selected_phase']['seconds']:.2f}s; audit ช่วงที่ข้ามเพิ่ม {result['skipped_audit']['seconds']:.2f}s (แยกออกจากวิธีหลัก)",
        '- เวลาไม่รวมดาวน์โหลดข้อมูล; ใช้ phase retention แบบ offline ที่ผ่านการทดลองก่อน ไม่ใช่การเปลี่ยนค่าเริ่มต้นใน demo']
    if resources:
        lines += [f"- Qwen phase peak VRAM ทั้งการ์ด {max(t['sampled_whole_gpu_peak_bytes'] or 0 for t in resources)/2**30:.2f}GiB; peak Qwen process-tree RSS {max(t['qwen_process_tree_peak_rss_bytes'] or 0 for t in resources)/2**30:.2f}GiB (sample ทุก0.5s, VRAM รวมโปรแกรมอื่น)"]
    errors=[r for r in result['selected_phase']['replies'] if r.get('error')]
    lines += [f"- Technical error ในรอบหลัก {len(errors)} หน่วย; uncertain แยกจาก normal ไม่แปลง OOM/ตรวจไม่สำเร็จเป็นปกติ",'',
        '## ข้อจำกัด','',
        '- ชุดใหม่ไม่ซ้ำฉากและไฟล์กับชุดเก่า แต่ยังเป็นภาพจำลองจาก dataset เดียวกัน และมีจำนวนน้อย ยังไม่ใช่หลักฐานว่าผ่านเป้าหมายใช้งานจริง',
        '- normal exposure ประมาณ1.5นาที ไม่เพียงพอรับรองเกณฑ์แจ้งผิด≤2เหตุต่อชั่วโมง; ค่า bootstrap ที่เก็บในผลดิบเป็นความไม่แน่นอนของชุดเล็ก ไม่ใช่การรับรอง generalization',
        '- ผลหลักใช้ first-pass Qwen ตาม controlled baseline เดิม เก็บ uncertain ไว้ ไม่เพิ่มวิธี retry/threshold ใหม่หลังเห็นผล',
        '- all_qwen_diagnostic รวมคำตอบจริงรอบหลักกับคำตอบของหน่วยที่ข้ามภายหลัง ไม่ใช่การรัน all-unit อีกครั้งที่ควบคุมลำดับและใช้เทียบเวลา',
        '- ประเมินช่วงเหตุจากขอบเขตหน่วยเดิมก่อน union; temporal matching และความต่อเนื่องของเหตุมีผลต่อ TP/FP ห้ามตีความเป็นการยืนยันชนิดการกระทำทุกกรณี',
        '- ไม่เปลี่ยน demo jobs/reviews, model weights, thresholds หรือ merge main', '', '## การทำซ้ำ','',
        '`python -m evaluation.embeddinggemma2.evaluate_frozen_scenes --manifest <manifest> --selection-lock <pre-download-lock> --output <fresh-run>`','',
        'ผลดิบ/manifest/protocol/คะแนน/คำตอบรายช่วงแนบใน output และใน scratch ของ worktree; วิดีโอ/ดัชนี/weights ไม่เข้า Git']
    report='\n'.join(lines)+'\n';(out/'results-th.md').write_text(report,encoding='utf-8')
    Path('docs/new-scenes-results-20261015.md').write_text(report,encoding='utf-8')
    for filename in ('results.json','protocol.json','gate.json'):shutil.copy2(run/filename,out/filename)
    shutil.copy2(args.manifest,out/'manifest.json')
    shutil.copy2(args.selection_lock,out/'selection-lock.json')
    with (out/'per-video.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['video_id','scene','seconds','gt_events','tp','fp','fn','selected_units','total_units']);w.writeheader()
        for row in rows:
            m=metrics([row],result['predictions']['gemma_qwen']);events=by_video[row['id']]
            w.writerow({'video_id':row['id'],'scene':row['scene_id'],'seconds':row['duration'],'gt_events':len(row['events']),
                        'tp':m['tp'],'fp':m['fp'],'fn':m['fn'],'selected_units':sum(e['id'] in selected for e in events),'total_units':len(events)})
    cards=[]
    for row in rows:
        events=by_video[row['id']];verdicts=[]
        for event in events:
            reply=replies[event['id']];v=reply.get('verdict') or {};span=intervals[event['id']]
            verdicts.append('<section><h4>'+escape(f"{span['t_start']:.2f}–{span['t_end']:.2f}s · "+('Gemma เลือก' if event['id'] in selected else 'audit ช่วงที่ข้าม'))+'</h4><pre>'+escape(json.dumps(v,ensure_ascii=False,indent=2))+'</pre></section>')
        frames=events[0]['frame_paths'];poster=out/(row['id']+'.jpg');shutil.copy2(frames[len(frames)//2],poster)
        gt='ปกติตาม label' if not row['events'] else 'Label: '+', '.join(f"{e['t_start']:.2f}–{e['t_end']:.2f}s" for e in row['events'])
        cards.append('<article><h2>'+escape(row['id'])+'</h2><p>'+escape(gt)+'</p><video controls preload="none" poster="'+escape(poster.name)+'" src="'+escape(Path(row['path']).as_uri())+'"></video><details><summary>คำตอบจริงทั้งหมดทุกช่วง</summary>'+''.join(verdicts)+'</details></article>')
    primary=result['metrics']['gemma_qwen']
    html='<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UBnormal — ผลทดสอบฉากใหม่</title><style>body{font:16px/1.6 system-ui,sans-serif;background:#f5f6fa;color:#202133;margin:0}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:28px}article{background:white;border:1px solid #dddde8;border-radius:12px;padding:20px;margin:20px 0}h2{font-size:20px;overflow-wrap:anywhere}video{width:100%;max-height:420px;background:#111;object-fit:contain}summary{cursor:pointer;padding:14px 0;color:#6152d8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f2f6;padding:16px;border-radius:8px;font-size:13px}header{background:white;padding:24px;border-radius:12px}</style><main><header><h1>ผลทดสอบฉากใหม่ — UBnormal</h1><p>ภาพจำลองจาก dataset ทางการ · ไม่มีเสียง · คงเกณฑ์เดิม · ไม่ใช่ official test</p><p>'+escape(f"Gemma → Qwen: TP {primary['tp']}, FP {primary['fp']}, FN {primary['fn']} · Recall {pct(primary['recall'])} · Precision {pct(primary['precision'])}")+'</p><p>ภาพเต็มตามสัดส่วนเดิม เปิดคำตอบจริงทั้งหมดได้ด้านล่างแต่ละคลิป</p><a href="https://github.com/lilygeorgescu/UBnormal">ที่มาและสิทธิ์ CC-BY-NC-ND-4.0</a></header>'+''.join(cards)+'</main></html>'
    (out/'review.html').write_text(html,encoding='utf-8')
    print('REPORT',out,flush=True)


if __name__=='__main__':main()
