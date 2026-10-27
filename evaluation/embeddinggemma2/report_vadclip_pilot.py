"""Publish measured VadCLIP pilot, including all prespecified variants."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil

import numpy as np

from .benchmark_vadclip import digest
from .open_anomaly_eval import metrics


def pct(value):
    return '—' if value is None else f'{value:.1%}'


def chart(row, raw, threshold):
    width, height = 900, 200
    scale = max(max(raw['binary']), threshold, .001) * 1.1
    x = lambda t: 20 + 860 * t / row['duration']
    y = lambda s: 175 - 150 * s / scale
    parts = [f'<svg role="img" aria-label="คะแนน VadCLIP ตามเวลา เส้นม่วงคือคะแนน แถบเขียวคือ label" viewBox="0 0 {width} {height}">']
    for event in row['events']:
        parts.append(f'<rect x="{x(event["t_start"]):.2f}" y="20" width="{x(event["t_end"])-x(event["t_start"]):.2f}" height="155" fill="#d6f4e4"/>')
    points = ' '.join(f'{x((b[2]+b[3])/2):.2f},{y(s):.2f}' for b, s in zip(raw['bounds'], raw['binary']))
    parts.append(f'<path d="M20 175H880" stroke="#999"/><line x1="20" y1="{y(threshold):.2f}" x2="880" y2="{y(threshold):.2f}" stroke="#e49b27" stroke-dasharray="6 5"/>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#6152d8" stroke-width="2"/><text x="20" y="195" font-size="12">0s</text><text x="810" y="195" font-size="12">{row["duration"]:.2f}s</text></svg>')
    return ''.join(parts), scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args(); run = Path(args.run); out = Path(args.output)
    result = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    if result['status'] != 'complete' or not result['code_unchanged'] or digest(args.manifest) != result['protocol']['manifest_sha256']:
        raise ValueError('Incomplete or mismatched experiment')
    rows = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['videos']
    validation = [r for r in rows if r['split'] == 'validation']
    out.mkdir(parents=True, exist_ok=True)
    lines = ['# VadCLIP เทียบระบบปัจจุบัน — ทดลองข้ามชุดข้อมูล', '',
        '## ข้อมูลและวิธี', '',
        '- ใช้ official UCF-Crime checkpoint จาก VadCLIP ไม่ฝึกน้ำหนักเพิ่ม; strict load ครบทุก parameter รวม CLIP ViT-B/16',
        f"- Official source revision `{result['protocol']['revision']}`; checkpoint SHA256 `{result['protocol']['checkpoint_sha256']}`",
        '- ตัวตรวจฝึกจาก UCF-Crime แต่ประเมินกับ UBnormal ภาพจำลอง จึงไม่ใช่การทำซ้ำคะแนนในบทความหรือหลักฐานกล้องจริง',
        f"- {len(validation)} validation videos, {sum(len(r['events']) for r in validation)} events, {sum(r['duration'] for r in validation):.2f}s; normal-only {sum(r['duration'] for r in validation if not r['events']):.2f}s",
        '- ใช้ training-normal 8 คลิป / 7 การบันทึกต้นฉบับเพื่อเกณฑ์ q95 เท่านั้น; fog และต้นฉบับอยู่ development เหมือนกัน ไม่ข้าม split',
        '- ล็อกสอง sampling ก่อนรัน: stride16=เฟรมแรกทุกกลุ่ม16เฟรม; mean16=เฉลี่ย features ทั้ง16เฟรม; เก็บ tail ตาม FPS จริง',
        '- ใช้ crop กลางตาม official image_crop(type=0), FP32, raw features ไม่เพิ่ม normalization; temporal context256 และ prompt labels UCF เดิม',
        '- Primary: stride16/binary/threshold0.5; อีก7แบบเป็น diagnostic ที่ประกาศก่อนเห็นผล ไม่เลือกแบบชนะจาก validation เพื่อใช้งานจริง',
        '- เกณฑ์อีกแบบคือ95th percentile (higher) ของคะแนน training-normal รวมทุกกล้อง ไม่ใช้ label validation และไม่เปลี่ยนตามผล validation',
        '- upstreamเผยแพร่ features สำเร็จรูป แต่มีความไม่ชัดเจนของ raw extraction จึงต้องระวัง feature-distribution mismatch แม้ใช้ checkpoint และ forward เดิม',
        '- Semantic score=1-softmax(normal), binary score=sigmoid(logit); เป็นคะแนนโมเดล ไม่รับรองว่าเป็น probability ที่ calibrate แล้ว',
        '', '## Event metrics — temporal IoU≥0.3, one-to-one matching', '',
        '| วิธี | TP | FP | FN | Recall | Precision | แจ้งผิดต่อชั่วโมงบนคลิปปกติ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    table = []
    for key in ('gemma_windows', 'gemma_qwen'):
        table.append((key, result['baseline_metrics'][key], None))
    for key, value in result['metrics'].items():
        if 'events' in value:
            table.append((key, value['events'], result['thresholds'][key]))
    for key, m, threshold in table:
        lines.append(f"| {key} | {m['tp']} | {m['fp']} | {m['fn']} | {pct(m['recall'])} | {pct(m['precision'])} | {m['false_alerts_per_hour']:.2f} |")
    lines += ['', 'Normal exposure เพียงประมาณ1.5นาที ค่าแจ้งผิดต่อชั่วโมงเป็นการคูณขยายจากชุดเล็ก ไม่เพียงพอรับรองเป้าหมาย≤2เหตุ/ชั่วโมง; precision ที่ไม่มี positive prediction แสดงเป็น —',
        '', '## Frame ranking — ไม่ใช่ event recall', '', '| Sampling/head | ROC-AUC | AP |', '|---|---:|---:|']
    for key, m in result['metrics'].items():
        if 'auc' in m:
            lines.append(f"| {key} | {m['auc']:.4f} | {m['ap']:.4f} |")
    lines += ['', '## เวลาและทรัพยากร', '',
        f"- GPU: {result['gpu']}; torch {result['torch_version']}; FP32",
        f"- โหลด checkpoint และสร้างโมเดล {result['load_seconds']:.2f}s; รวมรันสอง sampling, calibration, repeated scoring และสร้าง metrics {result['total_seconds']:.2f}s",
        f"- Peak PyTorch allocated {result['peak_allocated_bytes']/2**30:.3f}GiB / reserved {result['peak_reserved_bytes']/2**30:.3f}GiB",
        f"- Peak process RAM {result['peak_process_rss_bytes']/2**30:.3f}GiB; GPUทั้งการ์ดที่สุ่มวัด {result['whole_gpu_peak_mib']:.0f}MiB (รวมโปรแกรมอื่น, sampled ไม่ใช่ exact peak)"]
    for sampling, timings in result['timing'].items():
        subset = [timings[r['id']] for r in validation]
        seconds = sum(t['extraction_seconds']+t['first_score_seconds'] for t in subset)
        repeated = [v for t in subset for v in t['repeated_score_seconds']]
        lines.append(f"- {sampling}: decode+CLIP+first scorer ของ12validation clips {seconds:.2f}s; encode {sum(t['encoded_images'] for t in subset)}ภาพ; cached-feature scorer p50 {np.quantile(repeated,.5)*1000:.2f}ms / p95 {np.quantile(repeated,.95)*1000:.2f}ms")
    lines += ['- cached-feature scorer ไม่รวม decode/CLIP; ไม่อ้างเวลาเฉพาะส่วนนี้เป็นความเร็วครบระบบ',
        '- เวลาระบบเดิมมาจากรอบก่อนบนเครื่องเดียวกัน ไม่ได้รันสลับจับคู่ในรอบนี้ จึงไม่สรุป speedup แบบควบคุม',
        '', '## ข้อสรุปและข้อจำกัด', '',
        '- Primary VadCLIP พบ0/6เหตุ; เกณฑ์q95 semantic ทั้งสองsampling พบ1/6เหตุ แจ้งผิด1เหตุ เทียบ Gemma→Qwenเดิมพบ3/6เหตุ แจ้งผิด2เหตุ',
        '- semantic-q95 พบ abnormal_scene_8_scenario_8 แต่แจ้งผิด normal_scene_8_scenario_2; binary-q95 เลือกบางส่วนของเหตุฉาก8 แต่ช่วงแตกเป็นสองส่วนจนแต่ละช่วงไม่ผ่าน temporal IoU0.3',
        '- การใช้mean16เพิ่ม frame ranking แต่ไม่เพิ่มจำนวนเหตุที่จับคู่ได้ในเกณฑ์ที่ล็อกไว้ จึงยังไม่มีหลักฐานให้ใช้แทนระบบเดิม',
        '- ดูทุก variant และผลรายคลิป ไม่ใช้ผลที่ดีที่สุดย้อนหลังมาแทน primary ที่ล็อกไว้',
        '- คะแนนต่ำอาจเกี่ยวข้องทั้ง domain shift, extraction mismatch และความสามารถของตัวตรวจ ต้องทดสอบ features ทางการของ UCF ก่อนสรุปว่าโมเดลต้นฉบับทำงานไม่ได้',
        '- ไม่มีการเปิด official test, เปลี่ยน demo/default, ฝึกโมเดล, หรือ merge main',
        '- แหล่งโมเดล: https://github.com/nwpu-zxr/VadCLIP ; ข้อมูล: https://github.com/lilygeorgescu/UBnormal (CC-BY-NC-ND-4.0)',
        '', '## ทำซ้ำ', '',
        '`python -m evaluation.embeddinggemma2.benchmark_vadclip --manifest <manifest> --repo <pinned-official-repo> --checkpoint <official-model_ucf.pth> --baseline <frozen-baseline-results> --output <new-folder>`',
        '', 'ผลดิบมี protocol, checkpoint/source hashes, scores ทุกช่วง, features NPZ, timings และ bootstrap95CI; weights/video อยู่ scratch ไม่เข้า Git']
    report = '\n'.join(lines) + '\n'
    (out / 'results-th.md').write_text(report, encoding='utf-8')
    Path('docs/vadclip-pilot-results-20261015.md').write_text(report, encoding='utf-8')
    for name in ('results.json', 'protocol.json'):
        shutil.copy2(run / name, out / name)
    shutil.copy2(args.manifest, out / 'manifest.json')
    with (out / 'per-video.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['video_id', 'arm', 'threshold', 'tp', 'fp', 'fn', 'recall', 'precision'])
        writer.writeheader()
        for key, pred in result['predictions'].items():
            for row in validation:
                m = metrics([row], pred)
                writer.writerow(dict(video_id=row['id'], arm=key, threshold=result['thresholds'][key], **{k: m[k] for k in ('tp', 'fp', 'fn', 'recall', 'precision')}))
    cards = []
    for row in validation:
        raw = result['raw_scores']['stride16'][row['id']]
        threshold = result['thresholds']['stride16/binary/training_normal_q95_higher']
        plot, scale = chart(row, raw, threshold)
        details = []
        for arm, pred in result['predictions'].items():
            m = metrics([row], pred)
            details.append(dict(arm=arm, threshold=result['thresholds'][arm], tp=m['tp'], fp=m['fp'], fn=m['fn'], intervals=pred[row['id']]))
        cards.append('<article><h2>'+escape(row['id'])+'</h2><video controls preload="none" src="'+escape(Path(row['path']).as_uri())+'"></video><p>Label: '+escape(str(row['events']))+'</p>'+plot+f'<p>stride16 binary · แกนคะแนน 0–{scale:.4f} · เส้นส้ม=q95 {threshold:.5f} · primary threshold=0.5</p><details><summary>ช่วงที่พบและผลทุกแบบ</summary><pre>'+escape(json.dumps(details, ensure_ascii=False, indent=2))+'</pre></details></article>')
    primary = result['metrics'][result['protocol']['primary']]['events']
    html = '<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VadCLIP pilot</title><style>body{font:16px/1.6 system-ui;background:#f5f6fa;color:#24243a;margin:0}main{max-width:1100px;padding:24px;margin:auto}article,header{background:white;border:1px solid #dddde8;border-radius:12px;padding:24px;margin-bottom:24px}h2{font-size:19px;overflow-wrap:anywhere}video{width:100%;max-height:420px;object-fit:contain;background:#111}svg{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f2f7;padding:16px}summary{cursor:pointer;color:#6152d8}p{overflow-wrap:anywhere}</style><main><header><h1>VadCLIP เทียบระบบเดิม</h1><p>UBnormal ภาพจำลอง · 12 validation videos / 6 events · น้ำหนักทางการ UCF-Crime · ไม่ใช่ผลทำซ้ำบทความ</p><p>'+escape(f"Primary: TP{primary['tp']} FP{primary['fp']} FN{primary['fn']} · Recall {pct(primary['recall'])}")+'</p><a href="results-th.md">รายงานผลและทรัพยากรครบทุกแบบ</a></header>'+''.join(cards)+'</main></html>'
    (out / 'review.html').write_text(html, encoding='utf-8')
    print('REPORT', out)


if __name__ == '__main__':
    main()
