# Gemma คัดก่อน Qwen ด้วยคำสั่งและเฟรมตรงกัน — 15 ตุลาคม 2026

**ผลจริง: ทั้งสองแบบพบ 6/7 เหตุ แจ้งผิด 1 เหตุ ไม่พบเหตุหายเพิ่มจากการใช้ Gemma ในรอบควบคุมนี้**
ส่งเข้า Qwen ลดจาก 19 เหลือ 16 หน่วย (15.8%); ไม่มี uncertain/unverified ในทั้งสองแบบ

## ข้อมูลและการควบคุม

- UBnormal synthetic scenes 3/4/5: development ปกติ 6 คลิป, validation 10 คลิป รวมเหตุที่มี label 7 ช่วง; ช่วงปกติ 60.13 วินาที ไม่มีเสียง
- ใช้โมเดล CapRL-Qwen3VL-4B Q6 revision เดิม พร้อม camera-normality policy ที่เคยได้ 6/7; ไม่มีการฝึกน้ำหนัก
- กริดไม่เกิน 12 วินาทีรวม 19 หน่วย ทุกช่วงที่ส่งให้ Qwen ใช้ payload ตรงกันทั้ง scene_context, timestamps, frame_paths และ normal references
- Gemma คัดจาก media embeddings ที่ทำดัชนีไว้แล้ว พร้อม encode ข้อความ query ใหม่และคำนวณ novelty/semantic gates ใหม่
- Threshold novelty=.05, margin=.1, similarity=.35 ใช้ validation lock เดิม ไม่ปรับค่าจากผลรอบนี้
- ส่งหน่วย Qwen เมื่อทับกับ Gemma candidate อย่างน้อยบางส่วน โดยไม่เปลี่ยนขอบเขตหรือสุ่มเฟรมใหม่
- เรียก Qwen ทั้งสองแบบจริง แบ่ง batch สูงสุด8 เหมือนกัน ไม่ใช้ cached Qwen answers แทนการรัน
- เทียบ assessment ของหน่วยที่ทั้งสองแบบได้ดู และทำ same-run subset replay เพื่อแยกความแปรปรวนของการรัน
- ไม่เปิด official test ไม่เขียนทับ demo jobs/reviews ไม่เปลี่ยน defaults และไม่ merge main

## ผลการทดลอง

| วิธี | TP | FP | FN | Recall | Precision | Unknown | คำขอ Qwen |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_qwen | 6 | 1 | 1 | 85.7% | 85.7% | 0 | 19 |
| gemma_then_qwen | 6 | 1 | 1 | 85.7% | 85.7% | 0 | 16 |

Shared assessment changes: 0; same-run gate-only replay metrics: {"tp": 6, "fp": 1, "fn": 1, "recall": 0.8571428571428571, "precision": 0.8571428571428571, "uncertain_units": 0, "normal_only_hours": 0.016703703703703703, "false_alerts_per_hour": 59.866962305986696, "positive_video_fraction": 0.6222996515679443}

สามช่วงที่ Gemma ข้ามมาจากคลิปปกติที่มี label อยู่แล้ว (labels ใช้ตรวจผลภายหลัง ไม่ใช้ตัดสิน gate):

- normal_scene_4_scenario_3-0
- normal_scene_4_scenario_3-1
- normal_scene_5_scenario_3-1

## เวลาและทรัพยากร

| วิธี | Qwen client wall time | Gemma gate | รวมส่วนที่วัด |
|---|---:|---:|---:|
| all_qwen | 230.10s | 0.00s | 230.10s |
| gemma_then_qwen | 195.73s | 6.45s | 202.18s |

เวลาไม่รวมสร้าง media index, ถอด MP4, เตรียม JPEG หรือดาวน์โหลดโมเดล ซึ่งมีอยู่ก่อนเริ่มรอบนี้
Gemma gate รวมโหลด/encode query และอ่านเวกเตอร์/คำนวณคะแนน; Qwen wall time รวม HTTP, โหลด/ปล่อยโมเดลแต่ละ batch และ client bookkeeping
รัน all_qwen ก่อน gemma_then_qwen ครั้งเดียว ไม่มีการควบคุม OS file cache หรือสลับลำดับ จึงไม่อ้าง speedup ทั่วไปจากเวลารอบเดียว
จำนวน private HTTP requests ไม่รวม JSON-repair generation ภายใน worker; ไม่เท่ากับจำนวน decoder generations เสมอไป

ทรัพยากร Qwen sampled ทุก .5s; whole-card VRAM รวมโปรแกรมอื่น ไม่ใช่ PyTorch allocated ของ Qwen:

- all_qwen: whole-card GPU peak 11.409 GiB; Qwen process-tree RSS peak 12.840 GiB
- gemma_then_qwen: whole-card GPU peak 11.180 GiB; Qwen process-tree RSS peak 12.822 GiB

## ข้อสรุปและขอบเขต

บนชุดนี้ Gemma คัดก่อนไม่ทำให้ผลแย่ลง และลดคำขอตรวจได้15.8% แต่ยังไม่ถึงเป้าหมายลดอย่างน้อย50%
อย่าสรุปว่า Gemma ไม่พลาดในทุกฉาก: threshold นี้เลือกจาก validation ชุดเดิมที่มีเพียง7เหตุ ยังไม่ใช่ independent test
ยังประเมินการแจ้งผิดไม่เกิน2ครั้งต่อชั่วโมงไม่ได้ เพราะ normal exposure มีประมาณหนึ่งนาที
ผลนี้เป็น controlled fixed-grid wrapper ไม่ใช่การเทียบ production pipelines ที่รวมช่วงใหม่คนละแบบ
ข้อชี้แจงผลเก่า: open-anomaly-live-preset-v1 เคยรัน Gemma คัดก่อนด้วย camera_normality_v2 ได้6/7+1FPแล้ว (15คำขอ แต่ขอบเขตบางหน่วยต่างจาก baseline); คำอธิบายก่อนหน้าที่แยกไว้เฉพาะ all-unit จึงไม่ครบ
ผลก่อนปรับ camera-normality policy ที่เหลือ4/7ไม่สามารถนำมาโทษการคัดก่อนโดยไม่ควบคุมคำสั่ง Qwen

## การตรวจและทำซ้ำ

Tests ผ่าน30ข้อ รวมการไม่เปลี่ยน payload, boundary-touch exclusion, skipped recording, missing/duplicate replies และข้อมูลอ้างอิงปกติ
code_unchanged=true, frames_unchanged=true; ตรวจ metrics จาก predictions ซ้ำแล้วตรงกัน
ผลดิบ: scratch/event-retrieval/matched-gemma-gate-20261015-v1 (protocol, requests/frame hashes, gate scores, partial replies และ results)

```powershell
$env:PYTHONPATH="$PWD/backend;$PWD"
python -m evaluation.embeddinggemma2.compare_matched_gemma_gate --output <fresh-output>
```
