# ทดลอง Action Expert ช่วย Gemma/Qwen — 8 October 2026

ทดลองจริงแล้ว: SlowFast ช่วยจัดลำดับช่วงที่ควรตรวจได้ดีใน validation ชุดเล็กนี้
แต่ยังไม่แก้การยืนยันของ Qwen จึงยังไม่เปิดเป็นค่าเริ่มต้นในเว็บหรือ merge main
และยังไม่อ้างว่าแทน Gemma/SigLIP2 หรือผ่านเกณฑ์ระบบสุดท้ายแล้ว

## วิธีทดลอง

ใช้ [SlowFast R50 AVA detection ของ PyTorchVideo](https://pytorchvideo.org/docs/tutorial_torchhub_detection_inference)
ที่ฝึกมาแล้ว ไม่ฝึกน้ำหนักใหม่ ใช้ full-frame RGB พร้อมกล่องคนจาก pretrained
SSDLite COCO ที่ตรวจจากเฟรมกลาง ไม่ใช้กล่องคนหรือเวลาจากเฉลยในการป้อนโมเดล
ใช้ 32 เฟรมจริงที่ไม่ซ้ำจากช่วง 2 วินาทีที่มี motion สูงสุดของแต่ละ window
(เงียบใช้ช่วงกลาง) มี slow pathway 8 เฟรม, short side 256, mean .45/std .225
ตาม input contract ของโมเดล ยกเว้นระยะเวลา 2 วินาทีเป็น configuration ทดลอง
ของเรา ตรวจชัดเจนว่าท้ายคลิปมีเฟรมพอ ไม่ทำซ้ำเฟรมแบบเงียบ ๆ

รัน CPU FP32, 4 threads ทั้ง detector/action model ไม่ใช้ GPU และไม่โหลด
โมเดลนี้แข่งกับ Gemma/Qwen บน GPU Dependencies เพิ่มเฉพาะ ignored
`scratch/action-expert-tools` ไม่ upgrade environment production
requirements เสริมอยู่ใน `evaluation/embeddinggemma2/requirements-action-expert.txt`.

[Official checkpoint definition](https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/models/hub/slowfast.py)
ระบุ `SLOWFAST_8x8_R50_DETECTION.pyth` บันทึก SHA256 ที่ดาวน์โหลดได้เป็น
`9bd6584a8212f91e4c7d96ec7d6bff31250e1c810fcee968471614a1b9658ac8`;
adapter ตรวจ hash ก่อนโหลด `weights_only=True` และ strict state dictionary.
นี่คือ hash ที่บันทึกในเครื่อง ไม่ใช่ checksum ที่ผู้เผยแพร่รับรองแยกต่างหาก

AVA ID เป็น one-based: fight/hit person=64, push person=76,
handshake=68, hug=70. ไม่ใช้ hit/push **object** (34/46) เป็นเหตุทำร้ายคน
คะแนนเป็น multi-label classifier score ไม่ใช่ probability ที่ calibrate แล้ว
fight/hit person เป็นหมวดกว้างกว่า punch โดยเฉพาะ จึงใช้เป็น evidence/triage
ยังไม่ประเมินล้ม เสียงตะโกน หรือคลิปยาวปกติในรอบนี้

## ล็อกก่อน validation

Development seq1/2: ทุก 41 windows, 4 เหตุเป้าหมายจาก labels เดิม
Validation seq13–16: 62 candidates เดิมที่ Gemma เก็บไว้, 8 เหตุเป้าหมาย
ไม่เพิ่ม candidate จากเฉลย ไม่เปิด test seq17–20
ข้อมูลเคยถูกใช้ในงานทดลองก่อนหน้า จึงไม่ใช่ blind test ใหม่

score = max(fight/hit person, push person), ต้องมากกว่า max(handshake,hug).
กติกาแรกเลือก threshold ที่ไม่มี out-of-target development window,
เพิ่ม coverage สูงสุด แล้วเลือกจำนวน windows น้อยที่สุดและ threshold สูงกว่า
ได้ **0.016354450955986977**. Development คัดได้ 2/4 เหตุจาก 5 windows

เมื่อวิเคราะห์ development พบว่าลด threshold เพื่อส่งต่อให้ Qwen ได้คุ้มกว่า
จึงเพิ่ม **exploratory** arm โดยเลือก coverage สูงสุดก่อน แล้วลด out-of-target
และจำนวน windows ได้ **0.007685719523578882**: 4/4 เหตุใน 11 windows,
out-of-target 1 window. Arm นี้คิดเพิ่มหลังเห็น development ไม่เรียกว่า
preregistered ตั้งแต่ต้น แต่ทั้งสอง threshold ถูกเขียน lock ก่อนเริ่ม validation
ไม่มีการปรับค่าตามผล validation

## ผลวัดจริง

Coverage เดิมหมายถึง window ใด window หนึ่งทับอย่างน้อย 50% ของเวลาเหตุ
แยก window 6 วินาทีสำหรับส่งตรวจ กับ span 2 วินาทีที่ Action Expert ดูจริง
เพื่อไม่ให้ช่วง context ที่กว้างทำให้ดูเหมือนระบุเวลาการกระทำได้แม่นแล้ว

| Arm | Dev: window coverage | Validation: window coverage | Validation: span coverage | Validation selected / 62 | Validation out-of-target windows |
|---|---:|---:|---:|---:|---:|
| Strict | 2/4 | 7/8 | 6/8 | 13 | 0 |
| Recall first (exploratory) | 4/4 | 8/8 | 7/8 | 16 | 0 |

Recall-first development span coverage จริงเป็น 3/4 ไม่ใช่ 4/4;
validation window temporal-IoU >=.3 เป็น 6/8 และ span temporal-IoU เป็น 7/8.
คัด 16/62 ลดจำนวน candidate windows 74.2% แต่ **ไม่ได้วัดว่า Qwen calls
หรือเวลารวมทั้งระบบลด 74.2%**. จำนวน out-of-target windows ไม่นับ kicks
หรือ unselected target annotations ตาม evaluator เดิม; ไม่ใช่ precision,
จำนวน false alerts ต่อชั่วโมง หรือหลักฐานว่าไม่มี false positives ในชีวิตจริง
ผล 8 เหตุมีความไม่แน่นอนสูง ต้องขยายข้อมูลก่อนตัดสิน production

Development wall time 58.73 s, validation 87.23 s รวมโหลดโมเดล,
motion/decode, person detection, action inference และบันทึกผล
peak process RSS ที่ sample ทุก .1 s: 1.429/1.430 GiB ตามลำดับ
เป็น RAM ของ process ไม่ใช่ CUDA allocated/reserved หรือ VRAM รวมทั้งการ์ด
มี unverified 3/41 และ 11/62 ตามลำดับ; ไม่เปลี่ยน unknown เป็น negative
ไม่มีการวัด cold disk-cache isolation หรือ query latency หลังทำดัชนีในรอบนี้

## Qwen ยังเป็นคอขวดหรือไม่

ตรวจ **offline** กับผล Qwen temporal-v1 control + motion retries ที่มีอยู่แล้ว
เฉพาะ 16 candidates ที่ action arm เลือก ตรวจให้ window/retry bounds ตรงกัน
ก่อน join. Qwen เดิมเห็น 8 ภาพ ไม่ใช่ 32 ภาพของ SlowFast และไม่มี action score
อยู่ใน prompt; นี่ไม่ใช่การรัน cascade ใหม่หรือการทดลอง action-guided prompt

Qwen ที่บันทึกไว้ยืนยันครอบคลุม **5/8 เหตุ** ทั้งเมื่อใช้ทุก candidate เดิม
และเมื่อกรองเหลือ 16 windows. ยังพลาด punch ใน seq13,14,16 แม้ candidate
window ถูกเก็บไว้ครบ. ดังนั้นการคัดช่วงที่ดีขึ้นอย่างเดียว **ยังไม่เพิ่มผลยืนยัน
สุดท้าย** ในชุดนี้ การเห็น window ครบยังไม่พิสูจน์ว่า Qwen เห็นจังหวะสัมผัส
หรือการเคลื่อนต่อเนื่องพอ โดยเฉพาะ seq16 ที่ motion span ยังไม่ครอบคลุมเหตุ
ไม่ควรสรุปจากข้อมูลนี้ว่า Qwen ผิดทั้งหมดหรือ SlowFast ยืนยัน punch ได้แล้ว

ขั้นถัดไปที่มีเหตุผลคือเก็บ action-high/Qwen-negative เป็นสถานะต้องตรวจต่อ
แล้วทดลองหลายช่วงสั้นที่เหลื่อมกัน/เฟรมก่อน-ระหว่าง-หลังสำหรับการยืนยัน
โดยล็อกกติกาจาก development และทดสอบ matched controls ไม่บังคับให้ Qwen
ตอบ positive ตามคะแนน action และไม่ใช้เฉลยเลือกเฟรมในระบบจริง
ยังไม่มี threshold ที่ผ่าน production gate, ไม่เปลี่ยนเว็บ/default/main

## ทำซ้ำและไฟล์หลักฐาน

ตั้ง PYTHONPATH เป็น backend และ repository root แล้วใช้ inference Python
ที่มี torch 2.10.0+cu128/torchvision 0.25.0+cu128 อยู่เดิม:

```powershell
python -m pip install --no-deps --target scratch/action-expert-tools -r evaluation/embeddinggemma2/requirements-action-expert.txt
python -m evaluation.embeddinggemma2.benchmark_action_expert
python -m evaluation.embeddinggemma2.analyse_action_expert freeze
python -m evaluation.embeddinggemma2.benchmark_action_expert --split validation
python -m evaluation.embeddinggemma2.analyse_action_expert report
python -m evaluation.embeddinggemma2.audit_action_cascade
```

คำสั่ง benchmark ต้องมี checkpoint และ control artifacts ตามพาธในโมดูล
ไม่เขียนทับ measured runs/locks เดิม การทำซ้ำต้องเก็บสำเนารอบเก่าแยกก่อน
Smoke inference แรกใช้ diagnostic clip seq2 14–16 s ที่รู้ตำแหน่งเหตุแล้ว
จึงไม่รวมเป็น benchmark sample และไม่ใช้เลือก threshold

Ignored artifacts ใต้ `scratch/event-retrieval/`:
`action-expert/{smoke.json,ava_action_list.pbtxt,SLOWFAST_8x8_R50_DETECTION.pyth}`,
`action-expert-dev/{protocol-lock.json,results.json}`,
`action-expert-recall-lock.json`, `action-expert-validation/{protocol-lock.json,results.json}`,
`action-expert-analysis.json`, `action-expert-cached-cascade.json`.
ไฟล์วิดีโอ/น้ำหนัก/output/environment ไม่เข้า Git

Tests ครอบคลุม AVA class indexing/person-vs-object, frame uniqueness,
RGB/normalization/ROI scaling, slow/fast alignment, finite outputs,
unknown propagation และ cache interval mismatch. Graph context อ่านจาก main
ก่อน scoping; โมดูล evaluation ใหม่ยัง unindexed และ graft executable ไม่พร้อม
จึงยัง refresh graph ไม่ได้

ตรวจ targeted suite แล้ว **68 tests ผ่าน**; มี dependency deprecation warnings
เดิมสองรายการจาก FastAPI/Starlette ไม่ใช่ failure ของรอบนี้
