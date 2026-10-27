# Open anomaly: ขยายเป็นสามฉาก — 8 ตุลาคม 2026

ผลชุดใหญ่ขึ้นยังไม่ผ่านเป้าหมายต้นแบบ: Gemma จับเหตุได้ 6/7 ตาม temporal IoU ≥.3
แต่หลัง Qwen เหลือ 4/7 แม้ตัดการแจ้งผิดบนคลิปปกติได้ทั้งหมด ตัวคัดกรองและตัวตรวจยืนยัน
จึงยังต้องพัฒนา ไม่ใช้ผล smoke test เดิม 1/1 เป็นหลักฐานว่าระบบแม่นทั่วไป

## ข้อมูลและ protocol

ใช้ไฟล์ต้นฉบับ [UBnormal ทางการ](https://github.com/lilygeorgescu/UBnormal)
scene 3, 4, 5: 16 คลิป รวม 210.93 s เป็นภาพจำลอง ไม่มีเสียง
เก็บแหล่งข้อมูล/สิทธิ์ CC-BY-NC-ND-4.0/hash และตรวจ CRC/ขนาดจาก ZIP ทางการ
ล็อก split lists ที่ revision `8c77642bb72615988ace0451b94ec42f8953a525`

- development: คลิปปกติ 2 คลิปต่อฉาก รวม 6 คลิป ใช้สร้างคลังปกติเท่านั้น
- validation: 10 คลิป มีเหตุจาก track annotations รวม 7 ช่วง และคลิปปกติ 4 คลิป
  รวม 60.13 s เลือกชื่อเรียงลำดับก่อนอ่าน annotations ไม่เลือกเฉพาะเหตุที่โมเดลทำได้
- ไม่อ่าน official test ไม่เปลี่ยน defaults ไม่ fine-tune และไม่ merge main
- ประเมินแบบจับคู่เหตุหนึ่งต่อหนึ่งที่ IoU ≥.3; รวม predicted units ที่ติดกันก่อนจับคู่
  จำนวนหน่วย Qwen ไม่ใช่จำนวนเหตุจริง

ใช้ดัชนี/เฟรมเดิมเท่าที่มีและสร้างส่วนที่ขาดใหม่ รอบนี้ไม่ใช่ cold indexing ทั้งชุด
แยก normal reference ตาม scene ใช้ภาพ 1 FPS/video 2 FPS หน้าต่าง 6 s เลื่อน 3 s
video budget 140 tokens/frame, 768 dimensions, BF16 GPU ตาม config เดิม
เลือก threshold จาก validation ก่อน Qwen: novelty .05, margin .1, similarity .35
ค่าทั้งหมดเป็นผลทดลองของชุดเล็ก ไม่ติดตั้งเป็น production lock

## ผลสี่วิธีและ baseline

| วิธี | เหตุที่จับได้ | Recall | Precision | เหตุแจ้งผิด | Qwen requests |
|---|---:|---:|---:|---:|---:|
| คลังปกติอย่างเดียว | 6/7 | 85.7% | 66.7% | 3 | 0 |
| ข้อความอย่างเดียว | 4/7 | 57.1% | 80.0% | 1 | 0 |
| รวมสองทาง | 6/7 | 85.7% | 66.7% | 3 | 0 |
| รวมสองทาง + Qwen | 4/7 | 57.1% | 100% | 0 | 15 |
| Qwen ทุกหน่วย 12 s | 4/7 | 57.1% | 100% | 0 | 19 |

Precision 100% หมายถึง 4 predictions ที่จับคู่ได้ใน validation นี้ ไม่ได้แปลว่าระบบแม่น 100%
ทั้งสองวิธี Qwen ไม่มี uncertain ในรอบนี้ การลด requests จริงคือ 21.1% ยังไม่ถึงเป้า ≥50%
Gemma เลือกเวลาวิดีโอ 81.1% จึงยังลดงานให้ Qwen ได้น้อย

## พลาดตรงไหน

ตัวประเมินใหม่เก็บรายคลิปและแยกสองขั้นไว้ใน `stage_diagnostics`:

- คงเหตุไว้ได้ 4 เหตุ
- Qwen ปฏิเสธเหตุที่ Gemma คัดได้ตามเกณฑ์ 2 เหตุ:
  `abnormal_scene_3_scenario_8` และเหตุแรกของ `abnormal_scene_4_scenario_7`
- อีกเหตุใน `abnormal_scene_4_scenario_7` ยาว 2 s ท้ายคลิปถูกครอบด้วยช่วงคัดที่ยาว
  แต่ไม่ได้เป็น prediction แยกที่จับคู่ IoU ≥.3 ได้ ไม่สรุปว่า Gemma ไม่เคยดูช่วงนั้น
  การรวมช่วงและการระบุขอบเขตเป็นปัญหาที่ต้องแยกจากความเข้าใจภาพ

Qwen อธิบายคลิปที่ปฏิเสธว่าเป็นการเต้น/เดินในย่านกลางคืน และการนั่ง/คุย/เดินตามปกติ
นี่คือคำอธิบายของโมเดล ไม่ใช่ label ประเภทการกระทำที่ผู้สร้างยืนยัน
label ที่ใช้มีช่วงเวลา abnormal จาก tracks เท่านั้น จึงยังแยกไม่ได้จากตัวเลขเพียงอย่างเดียวว่า
โมเดลมองการเคลื่อนไหวผิด หรือใช้ความหมายคำว่า “ปกติ” ต่างจาก benchmark

หลักฐานนี้ชี้ให้ทดสอบคำสั่งที่แยก “ผิดจากกิจวัตรของกล้อง” ออกจาก “อันตราย/มีการบาดเจ็บ”
โดยใช้เฟรมและตัวอย่างปกติชุดเดียวกัน รวมคลิปปกติด้วย ไม่แก้ prompt แล้วรายงานเฉพาะคลิปที่เคยพลาด

## เวลาและทรัพยากร

- รอบสี่วิธี: 377.15 s รวมดัชนีส่วนที่ขาดและ inference ส่วนที่ใช้ cache แล้ว
- เวลา verify batches ของแบบคัดก่อน: 169.50 s, 15 HTTP requests
- Qwen ทุกหน่วย: 146.13 s, 19 HTTP requests; งบเฟรม/โมเดล/prompt เดียวกัน
- แบบคัดก่อนทั้งการ์ด GPU peak ≈11.48 GiB; Qwen process-tree RAM peak ≈6.64 GiB
- baseline ทั้งการ์ด GPU peak ≈11.42 GiB

ลด requests ไม่ได้แปลว่าเร็วขึ้นในรอบนี้: baseline รวมงานข้ามคลิปเป็น batch สูงสุด 8
แต่ตัวรันแบบคัดก่อนตรวจทีละวิดีโอ มีการโหลด/ปล่อยโมเดลต่างกัน จึงไม่ใช่ latency comparison
ที่ควบคุม scheduler เท่ากัน ต้องวัดแบบจัดคิวเดียวกันก่อนอ้างความเร็วที่ดีขึ้น
GPU peaks สุ่มทุก .5 s รวมโปรแกรมอื่นบนการ์ด ไม่ใช่ PyTorch allocation ของ Qwen/llama.cpp

## ความน่าเชื่อถือและงานต่อ

### รอบที่สอง: เปลี่ยนเฉพาะเกณฑ์ความปกติของ Qwen

ทำ controlled ablation จาก all-unit baseline ทั้ง 19 หน่วย: ภาพ ลำดับเวลา normal references
โมเดลและงบเฟรมเหมือนเดิม เปลี่ยนเฉพาะ `scene_context` โดยเพิ่มหลักว่า
ให้เทียบกิจวัตรของกล้อง ไม่ใช้ความอันตราย/การบาดเจ็บเป็นเงื่อนไขของ anomaly
คำอธิบายที่ฟังดูเป็นไปได้ยังไม่พิสูจน์ว่าพฤติกรรมนั้นเป็นกิจวัตร และถ้าหลักฐานไม่พอให้ uncertain
ไม่ใช้เสื้อผ้า/รูปลักษณ์ใหม่เพียงอย่างเดียวเป็นเหตุผิดปกติ

| Qwen ทุกหน่วย ใช้เฟรมเดิม | TP / FN / FP | Recall | Precision |
|---|---:|---:|---:|
| คำสั่งเดิม | 4 / 3 / 0 | 57.1% | 100% |
| เพิ่ม camera normality policy | 6 / 1 / 1 | 85.7% | 85.7% |

จับเหตุเดิมที่พลาดเพิ่มได้ 2 เหตุ (`abnormal_scene_3_scenario_8-0`,
`abnormal_scene_4_scenario_7-0`) แต่ทำให้ `normal_scene_4_scenario_4-0` แจ้งผิด
คำอธิบายแจ้งผิดพูดถึงคนชุดแดงและการเดินที่ “ผิดธรรมชาติ” แม้คำสั่งห้ามใช้เสื้อผ้าเพียงอย่างเดียวแล้ว
จึงยังต้องแยกความแตกต่างด้านรูปลักษณ์ออกจากพฤติกรรม และตรวจการทำตามคำสั่งของโมเดล
ไม่แก้ตัวประเมินเพื่อลบ false positive นี้

นี่เป็น **Qwen all-unit ablation** ไม่ใช่ผล pipeline Gemma + Qwen ที่เปิดใช้อยู่
และไม่ใช่หลักฐาน test: ออกแบบ policy หลังตรวจข้อผิดพลาดใน validation จึงยังมีโอกาส overfit
เป็นการรันครั้งเดียวต่อ variant แม้ decoder ใช้ temperature 0 ยังไม่ได้วัดความคงที่จากการรันซ้ำ
ทั้งสอง variant ไม่มี uncertain ผลใหม่มี 1 false alert ใน normal exposure 60.13 s
(อัตราสังเกตประมาณ 59.9/hour บนข้อมูลสั้นนี้) จึงยังไม่ผ่านเป้า ≤2/hour และไม่เปิดเป็นค่าเริ่มต้น

ใช้เวลา 144.11 s, 19 HTTP requests, ไม่ทราบจำนวน requests 0 หน่วย
whole-card GPU peak ≈11.49 GiB และ Qwen process-tree RAM peak ≈12.90 GiB
ตัวเลข RAM ไม่ใช่ GPU allocation และเป็นอีก run หนึ่ง ไม่ควรอ้างว่า policy นี้ลด RAM
ตรวจ hash เฟรมก่อน/หลังแล้วเท่ากัน; `code_unchanged=true`, `frames_unchanged=true`
ผล/requests/transitions/protocol เก็บใน `open-anomaly-context-ablation-v1`

คำสั่งรอบสอง:

```powershell
python -m evaluation.embeddinggemma2.ablate_open_context --manifest <manifest.json> --baseline <baseline-dir> --experiment <run-dir> --output <fresh-ablation-dir>
```

มีเพียง 7 เหตุและ normal exposure ประมาณหนึ่งนาที Bootstrap ระดับ recording ให้ช่วง
recall สุดท้ายประมาณ 16.7–100% ซึ่งยังกว้างมาก ค่า precision/false alerts bootstrap ที่ไม่มี
false positives อาจเป็นช่วงเสื่อม 100–100%/0–0 ไม่ควรตีความว่ารู้ความเสี่ยงจริงแน่นอน
ยังไม่มีผล ShanghaiTech ภาพจริงหรือ official held-out test และยังพิสูจน์ ≤2 false alerts/hour ไม่ได้

ลำดับต่อคือทดสอบนิยาม normality ของ Qwen แบบเปลี่ยนปัจจัยเดียว แล้วแก้การรวมเหตุ/ขอบเขตเวลา
และขยายคลังปกติพร้อมวัดงานที่ส่งต่อ ใช้ validation สำหรับพัฒนา เก็บ test ไว้หลังล็อกค่าจริง

การทดสอบนิยาม normality รอบนี้ทำแล้วและพบ tradeoff ตามตารางข้างต้น
ลำดับพัฒนาถัดไปจึงควรแก้การรวมเหตุสั้น และลดการใช้รูปลักษณ์เป็นเหตุแจ้งผิด
ก่อนเลือก policy สุดท้าย ยังไม่เปลี่ยน defaults หรือ thresholds ของ demo

## การแก้เครื่องมือและหลักฐาน

พบว่าตัวรันเดิมใช้ SQLite backup ทั้งฐานเมื่อ reuse index ทำให้ events/jobs/profiles/reviews
ของรอบก่อนติดมาด้วย จึงหยุดรอบแรกก่อนใช้ผลและเก็บ directory เดิมไว้
แก้เป็นคัดลอกเฉพาะ videos/indexes/active/windows/embeddings ของคลิปใน manifest
กรองผลตาม job ID อีกชั้น เพิ่ม test ป้องกันผลเก่าปะปนและยืนยันฐานต้นทางไม่เปลี่ยน
baseline ผูก manifest hash และอ่าน profile/version จาก job snapshot โดยตรง
metrics บังคับให้ทุกคลิปมี predictions อย่างชัดเจน ไม่ถือผลที่หายไปว่าเป็นรายการว่าง

ผลดิบ ignored ใน `scratch/event-retrieval/`:

- `datasets/ubnormal-open-multiscene-v1`: selection ก่อน labels, official lists, videos, tracks, manifest
- `open-anomaly-multiscene-v1`: รอบที่หยุดเพื่อแก้ index-cache isolation ไม่ใช่ผลทดลองสำเร็จ
- `open-anomaly-multiscene-v2`: ผลครบสี่วิธี, validation lock, hashes, รายเหตุ; `code_unchanged=true`
- `open-anomaly-multiscene-baseline-v1`: all-unit Qwen ที่รันจริงพร้อม request payloads/replies

ไฟล์คลิป/model/environment/ผลดิบไม่เข้า Git การ refresh graph ทำไม่ได้เพราะเครื่องนี้ไม่มี graft CLI
การทดสอบที่เกี่ยวข้องหลังสองรอบผ่าน 130 ข้อ มี deprecation warnings เดิม 2 รายการ
ไม่ได้แก้ frontend ในรอบนี้ จึงไม่รัน frontend checks ซ้ำจากรอบส่งมอบก่อนหน้า

คำสั่งทำซ้ำใช้ inference environment ที่มีอยู่และ `PYTHONPATH=backend;.`:

```powershell
python -m evaluation.embeddinggemma2.prepare_ubnormal --scenes 3 4 5 --per-validation-class 2 --output <fresh-dataset-dir>
python -m evaluation.embeddinggemma2.run_open_anomaly --manifest <manifest.json> --output <fresh-run-dir> --reuse-index-store <prior-events.sqlite3>
python -m evaluation.embeddinggemma2.benchmark_open_baseline --manifest <manifest.json> --experiment <run-dir> --output <fresh-baseline-dir>
```
