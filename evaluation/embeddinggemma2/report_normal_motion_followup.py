"""Report controlled normal-motion reference trials without hiding failed gates."""
import argparse
import csv
from html import escape
import json
from pathlib import Path
import shutil

from app.gemma2.store import file_hash
from .open_anomaly_eval import metrics, bootstrap
from .compare_matched_gemma_gate import predictions


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def percentage(value):
    return '—' if value is None else f'{value:.1%}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); root = Path(args.root); out = Path(args.output)
    lock = read(root / 'amendment-lock.json')
    prior = Path('scratch/event-retrieval/overview-detail-new-scenes-20261015-v1')
    datasets = [
        ('previous_regression', prior, Path('scratch/event-retrieval/datasets/ubnormal-new-scenes-20261015/manifest.json')),
        ('old_regression', root / 'regression', Path('scratch/event-retrieval/datasets/ubnormal-open-multiscene-v1/manifest.json')),
        ('fresh_followup', root / 'holdout-comparison', Path('scratch/event-retrieval/datasets/ubnormal-detail-holdout-20261015/manifest.json')),
    ]
    evaluated = []; cards = []; csv_rows = []; score_rows = []
    out.mkdir(parents=True, exist_ok=True)
    for name, run, manifest_path in datasets:
        result = read(run / 'results.json')
        manifest = read(manifest_path)
        if result['status'] != 'complete' or not result['code_unchanged'] or not result['frames_unchanged']:
            raise ValueError('Incomplete or changed experiment: ' + name)
        if file_hash(manifest_path) != result['protocol']['manifest_sha256']:
            raise ValueError('Manifest hash mismatch')
        rows = [r for r in manifest['videos'] if r['split'] == 'validation']
        requests = read(run / 'requests.json')
        summary = {'dataset': name, 'videos': len(rows), 'events': sum(len(r['events']) for r in rows),
            'normal_seconds': sum(r['duration'] for r in rows if not r['events']),
            'protocol_sha256': file_hash(run / 'protocol.json'),
            'metrics': {}, 'bootstrap_95ci': {}, 'matched_events': {}, 'seconds': {}, 'resources': {}, 'technical_failures': {}, 'model_uncertain_units': {}, 'model_calls': {},
            'preparation_seconds': result['preparation_seconds'], 'original_promotion': result['promotions']['normal_motion']}
        for arm in ('control', 'normal_motion'):
            value = result['arms'][arm]
            summary['metrics'][arm] = value['metrics']
            summary['matched_events'][arm] = value['matched_events']
            summary['bootstrap_95ci'][arm] = bootstrap(rows, value['predictions'])
            summary['seconds'][arm] = value['phase']['seconds']
            summary['technical_failures'][arm] = sum(bool(r.get('error')) or r.get('verdict') is None for r in value['phase']['replies'])
            summary['model_uncertain_units'][arm] = sum((r.get('verdict') or {}).get('assessment') == 'uncertain' for r in value['phase']['replies'])
            summary['model_calls'][arm] = sum(r.get('calls', 0) for r in value['phase']['replies'])
            samples = [t['qwen_resources'] for t in value['phase']['telemetry'] if t.get('qwen_resources')]
            summary['resources'][arm] = {
                'sampled_whole_gpu_peak_bytes': max((s['sampled_whole_gpu_peak_bytes'] for s in samples), default=None),
                'qwen_process_tree_peak_rss_bytes': max((s['qwen_process_tree_peak_rss_bytes'] for s in samples), default=None)}
            for row in rows:
                m = metrics([row], value['predictions'])
                csv_rows.append({'dataset': name, 'video_id': row['id'], 'arm': arm,
                                 **{k: m[k] for k in ('tp', 'fp', 'fn', 'uncertain_units')}})
            for reply in value['phase']['replies']:
                unit = requests['intervals'][reply['id']]
                score_rows.append({'dataset': name, 'arm': arm, 'id': reply['id'],
                                   **unit, 'reply': reply})
        if name == 'fresh_followup':
            source = root / 'holdout-source'
            selection = read(source / 'requests.json')
            comparison_lock = read(root / 'comparison-lock.json')
            if file_hash(source / 'requests.json') != comparison_lock['source_requests_sha256']:
                raise ValueError('Fresh comparison selection changed')
            selected_ids = {e['id'] for e in selection['selected']}
            summary['selected_units'] = len(selected_ids)
            summary['total_units'] = len(selection['all_units'])
            summary['gated_replay_metrics'] = {}
            summary['screening_metrics'] = read(source / 'results.json')['metrics']['selected_grid']
            for arm in ('control', 'normal_motion'):
                replies = [r for r in result['arms'][arm]['phase']['replies'] if r['id'] in selected_ids]
                summary['gated_replay_metrics'][arm] = metrics(rows, predictions(rows, replies, selection['intervals']))
        control = summary['metrics']['control']; candidate = summary['metrics']['normal_motion']
        baseline_matches = {tuple(v) for v in summary['matched_events']['control']}
        candidate_matches = {tuple(v) for v in summary['matched_events']['normal_motion']}
        summary['nonregression_guard'] = (baseline_matches <= candidate_matches and
            candidate['fp'] <= control['fp'] and candidate['uncertain_units'] <= control['uncertain_units'])
        evaluated.append(summary)
        target = out / name; target.mkdir(exist_ok=True)
        for filename in ('results.json', 'protocol.json', 'preparation.json'):
            shutil.copy2(run / filename, target / filename)
        shutil.copy2(manifest_path, target / 'manifest.json')
        for row in rows:
            sections = []
            for arm in ('control', 'normal_motion'):
                source = result['arms'][arm]
                replies = {r['id']: r for r in source['phase']['replies']}
                events = [e for e in requests['arms'][arm] if requests['intervals'][e['id']]['video_id'] == row['id']]
                images = []; answers = []
                for event in events:
                    unit = requests['intervals'][event['id']]
                    answers.append({'interval': unit, 'reply': replies[event['id']]})
                    for ref in event['reference_frames']:
                        path = Path(ref['path'])
                        image_name = f'{arm}-{file_hash(path)[:12]}-{path.name}'
                        dest = target / image_name
                        if not dest.exists():
                            shutil.copy2(path, dest)
                        image_url = name + '/' + image_name
                        if image_url not in [u for u, _ in images]:
                            images.append((image_url, ref['video_id']))
                m = metrics([row], source['predictions'])
                sections.append('<section><h3>' + escape(arm) + '</h3><p>' +
                    escape(f"TP{m['tp']} FP{m['fp']} FN{m['fn']} · ไม่แน่ใจ {m['uncertain_units']} ช่วง") +
                    '</p>' + ''.join('<figure><img loading="lazy" src="' + escape(url) +
                    '"><figcaption>คลิปปกติอ้างอิง: ' + escape(identifier) + '</figcaption></figure>' for url, identifier in images) +
                    '<details><summary>คำตอบจริงครบทุกช่วง</summary><pre>' +
                    escape(json.dumps(answers, ensure_ascii=False, indent=2)) + '</pre></details></section>')
            cards.append('<article><h2>' + escape(name + ' / ' + row['id']) + '</h2><video controls preload="none" src="' +
                escape(Path(row['path']).as_uri()) + '"></video><p>Label: ' + escape(str(row['events'])) +
                '</p><div class="comparison">' + ''.join(sections) + '</div></article>')
    if file_hash(prior / 'results.json') != lock['prior_result_sha256']:
        raise ValueError('Original failed-promotion result changed')
    final = {'status': 'complete', 'amendment': lock, 'datasets': evaluated,
             'all_studies_pass_original_promotion': all(r['original_promotion'] for r in evaluated),
             'all_studies_pass_nonregression_guard': all(r['nonregression_guard'] for r in evaluated),
             'test_opened': False, 'production_activation': False}
    lines = ['# Qwen: เปรียบเทียบกับภาพลำดับเวลาจากคลิปปกติ', '',
        '## วิธีที่ปรับ', '',
        '- เปลี่ยนภาพปกติอ้างอิงเดี่ยวให้เป็นภาพลำดับ4เวลาในกรอบ2×2 จากหน้าต่างปกติเดียวกัน พร้อมเวลาและป้าย NORMAL',
        '- คง Qwen CapRL-Qwen3VL-4B Q6, เฟรมเหตุปัจจุบัน, timestamps, จำนวนภาพอ้างอิงสูงสุด2ภาพ, ขนาดcanvas และช่วงตรวจเดิม',
        '- เพิ่มคำอธิบายว่า reference มาจากคนละคลิป และให้เปรียบเทียบลักษณะการกระทำตามเวลา; ไม่ใช้เฟรมอ้างอิงเป็นเหตุปัจจุบัน',
        '- ทุก reference มาจาก development-normal; ถ้ามีไม่ถึง4เฟรมต่างกันใช้ภาพเดิมและบันทึก fallback ไม่สร้างการเคลื่อนไหวขึ้นเอง',
        '- ไม่ฝึกน้ำหนัก ไม่ปรับ threshold จากชุด follow-up และไม่ได้วัดจำนวน visual tokens จริง',
        '', '## เหตุผลที่เป็นการทดลองต่อ', '',
        'ชุดฉาก6/8/14/16 ก่อนหน้านี้เพิ่มจาก3เป็น4เหตุ ที่FP2เท่าเดิม แต่เพิ่ม uncertain หนึ่งช่วง จึงไม่ผ่าน promotion เดิม การทดลองนี้ประกาศ amendment ก่อน inference ฉาก22/24 เพื่อศึกษาต่ออย่างเปิดเผย ไม่ถือว่าวิธีผ่านเกณฑ์เดิมแล้ว',
        '', '## ผล event matching: temporal IoU≥0.3', '',
        '| ชุด | วิดีโอ / เหตุ | วิธี | TP | FP | FN | Recall | Precision | รอตรวจทั้งหมด |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for value in evaluated:
        for arm, m in value['metrics'].items():
            lines.append(f"| {value['dataset']} | {value['videos']} / {value['events']} | {arm} | {m['tp']} | {m['fp']} | {m['fn']} | {percentage(m['recall'])} | {percentage(m['precision'])} | {m['uncertain_units']} |")
    lines += ['', 'รอตรวจทั้งหมดใช้ค่าuncertain_unitsจากmetricเดิม ซึ่งรวมทั้งคำตอบuncertainที่ถูกschemaและtechnical-unverifiedที่ยังไม่มีverdict; แยกสองชนิดในsummary.json ไม่ถือเป็นเหตุที่ยืนยันถูก',
        'fresh_followup เปรียบเทียบทุก12ช่วงเพื่อแยกผลของQwen; previous/oldใช้ payload ที่ล็อกในงานก่อนหน้า ผลแต่ละชุดจึงต้องอ่านแยกกัน',
        '', '## ผลต่างและข้อสรุป', '']
    for value in evaluated:
        before = {tuple(x) for x in value['matched_events']['control']}
        after = {tuple(x) for x in value['matched_events']['normal_motion']}
        lines.append(f"- {value['dataset']}: เหตุที่ได้เพิ่ม {sorted(after-before)}; เหตุเดิมที่หาย {sorted(before-after)}; รักษาผลเดิมโดยFP/uncertainไม่เพิ่ม={value['nonregression_guard']}; ผ่านเกณฑ์promotionเดิมที่ต้องมีimprovementด้วย={value['original_promotion']}")
        lines.append(f"  คำตอบuncertainจริง={value['model_uncertain_units']}; technical-unverified={value['technical_failures']}")
    lines += ['- จะพิจารณาใช้เป็นค่าเริ่มต้นเมื่อรักษาเหตุเดิมได้ครบ เพิ่มTPโดยFPไม่เพิ่ม หรือTPเท่าเดิมแต่FPลด และ uncertain ไม่เพิ่มในแต่ละชุด',
        f"- ผ่านเกณฑ์เดิมครบทุกชุด: {final['all_studies_pass_original_promotion']}; production activation=False",
        '- ผลดีบนชุดเล็กบางฉากไม่ใช่หลักฐานว่าดีขึ้นทั่วไป ต้องเก็บวิธีและผลที่ไม่ผ่านไว้ ไม่เลือกตัวเลขดีที่สุดจากทุกชุดมารวมอ้าง']
    fresh = next(v for v in evaluated if v['dataset'] == 'fresh_followup')
    lines += ['', '## ฉากสำรองเมื่อใช้ Gemma gate เดิม', '',
        f"Gemmaเลือก {fresh['selected_units']}/{fresh['total_units']} หน่วย; ค่าด้านล่างกรองจากคำตอบจริงของall-gridด้วยselected IDsเดิม เป็นreplayไม่ใช่การรันเพิ่มหรือเวลาที่วัดแยก", '',
        '| ขั้นตอน | TP | FP | FN | Recall | Precision |', '|---|---:|---:|---:|---:|---:|']
    for arm, m in [('Gemma selected grid', fresh['screening_metrics']), *fresh['gated_replay_metrics'].items()]:
        lines.append(f"| {arm} | {m['tp']} | {m['fp']} | {m['fn']} | {percentage(m['recall'])} | {percentage(m['precision'])} |")
    lines += ['', '## เวลา', '', '| ชุด | เตรียมreference CPU (s) | control Qwen (s) | normal-motion Qwen (s) |', '|---|---:|---:|---:|']
    for value in evaluated:
        lines.append(f"| {value['dataset']} | {value['preparation_seconds']:.2f} | {value['seconds']['control']:.2f} | {value['seconds']['normal_motion']:.2f} |")
    lines += ['', 'รันแต่ละวิธีเป็นphaseต่อกัน ไม่ใช่การทดสอบเวลาสลับลำดับหลายรอบ จึงไม่สรุป speedup; เวลานี้ไม่รวม Gemma indexing ซึ่งเก็บแยกในผลดิบของ fresh-source',
        'Qwen รันผ่าน llama.cpp จึงดู PyTorch allocated ของbrokerเป็นVRAMของQwenไม่ได้; เก็บ sampled whole-card GPU peak และ process-tree RAM แยกไว้ใน summary.json โดย GPU รวมโปรแกรมอื่นด้วย',
        '', '## ขอบเขตและการทำซ้ำ', '',
        '- UBnormal เป็นภาพจำลอง; previous/old เป็น regression ที่ตรวจมาแล้ว ส่วน fresh เป็น validation ฉากสำรองที่เลือกและดาวน์โหลดไว้ก่อน ไม่มี official test',
        '- แหล่งและสิทธิ์: https://github.com/lilygeorgescu/UBnormal — CC-BY-NC-ND-4.0; manifest มี recording IDs, source revision และ SHA256',
        '- การอนุมานความผิดปกติขึ้นกับบริบทพื้นที่; binary label ไม่ได้ยืนยันชื่อการกระทำหรือเจตนาของบุคคล',
        '- normal exposure สั้น จึงยังรับรอง≤2แจ้งผิด/ชั่วโมงไม่ได้; เก็บ95%video-bootstrap intervals ใน summary.json พร้อมจำนวนตัวอย่าง',
        '- Full JSON คำตอบจริง, ภาพอ้างอิง, hashes และ protocol เก็บในรายงาน ไม่แก้คำตอบย้อนหลัง ไม่เปลี่ยนหน้าdemoหรือmerge main',
        '', '`python -m evaluation.embeddinggemma2.compare_overview_detail --source <frozen-requests> --payload-key <selected|uniform> --manifest <manifest> --reference-store <store> --arms control normal_motion --output <fresh-run>`']
    report = '\n'.join(lines) + '\n'
    (out / 'results-th.md').write_text(report, encoding='utf-8')
    Path('docs/normal-motion-followup-results-20261015.md').write_text(report, encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    (out / 'unit-replies.json').write_text(json.dumps(score_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(root / 'amendment-lock.json', out / 'amendment-lock.json')
    shutil.copy2(root / 'comparison-lock.json', out / 'comparison-lock.json')
    extra = out / 'fresh-source'; extra.mkdir(exist_ok=True)
    for filename in ('results.json', 'protocol.json', 'gate.json'):
        shutil.copy2(root / 'holdout-source' / filename, extra / filename)
    with (out / 'per-video.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    html = '<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen normal-motion follow-up</title><style>body{margin:0;background:#f5f6fa;color:#25253b;font:16px/1.6 system-ui}main{max-width:1250px;margin:auto;padding:24px}header,article{background:white;border:1px solid #dddde8;border-radius:12px;padding:24px;margin-bottom:24px}h2{font-size:19px;overflow-wrap:anywhere}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{width:100%;height:auto;object-fit:contain}video{width:100%;max-height:420px;object-fit:contain;background:#111}figure{margin:16px 0}figcaption{font-size:13px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f3f7;padding:16px;font-size:13px}summary,a{color:#6152d8;cursor:pointer}@media(max-width:800px){.comparison{grid-template-columns:1fr}}</style><main><header><h1>Qwen เทียบกับคลิปปกติแบบลำดับเวลา</h1><p>ผลจริงทั้งชุดเดิมและฉากสำรอง · UBnormal ภาพจำลอง · ไม่มีการฝึกน้ำหนักเพิ่ม</p><a href="results-th.md">รายงานผลครบทุกชุดและข้อจำกัด</a></header>' + ''.join(cards) + '</main></html>'
    (out / 'review.html').write_text(html, encoding='utf-8')
    print('REPORT', out, json.dumps(final['datasets']), flush=True)


if __name__ == '__main__':
    main()
