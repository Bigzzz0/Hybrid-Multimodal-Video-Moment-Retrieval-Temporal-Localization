# ผลการพัฒนา Open Anomaly Screening — 8 ตุลาคม 2026

**มีผลรอบขยายแล้ว:** [สามฉาก / เจ็ดเหตุการณ์](open-anomaly-multiscene-results.md)
Gemma 6/7 → หลัง Qwen 4/7; ผลหนึ่งเหตุด้านล่างเป็นเพียง smoke test เดิม

เพิ่มโหมดความผิดปกติทั่วไปให้ Gemma คัดช่วงจากคลังปกติของพื้นที่และข้อความ
แล้วให้ Qwen ตรวจบริบทพร้อมตัวอย่างปกติ โดยไม่บังคับเลือกเฉพาะชก ผลัก หรือล้ม
เปิดทดลองในหน้า `/events` และยังไม่เปลี่ยนโหมดเริ่มต้นหรือ merge เข้า main

## ข้อมูลที่รันจริง

ดาวน์โหลดไฟล์ต้นฉบับจาก [UBnormal ทางการ](https://github.com/lilygeorgescu/UBnormal)
โดยอ่าน ZIP ผ่าน HTTP byte ranges ไม่ต้องดาวน์โหลดทั้ง 15 GB:

| การใช้งาน | คลิป | ความยาว |
|---|---|---:|
| คลังปกติ development | normal_scene_4_scenario_2 | 9.50 s |
| คลังปกติ development | normal_scene_4_scenario_6 | 15.03 s |
| validation ปกติ | normal_scene_4_scenario_3 | 15.03 s |
| validation ผิดปกติ | abnormal_scene_4_scenario_3 | 15.37 s |

เลือก scene 4 และเรียงชื่อคลิปก่อนอ่าน label ใช้ช่วงจาก tracks ตามสคริปต์ของผู้สร้าง
เหตุผิดปกติที่ label ครอบคลุมมีเพียงหนึ่งช่วง ทั้งชุดเป็นภาพจำลองและไม่มีเสียง
ไม่ได้เปิด official test หรือใช้ validation เป็นคลังปกติ

ShanghaiTech ยังดาวน์โหลดจากต้นทางไม่ได้: ลิงก์ OneDrive ทางการตอบ HTTP 401
เมื่อตรวจรายการไฟล์ จึงไม่มีผลชุดนี้ และยังไม่ครบการประเมินหลักตามแผน

## เปรียบเทียบสี่วิธี

| วิธี | จับเหตุที่มี label ได้ | เหตุแจ้งผิดบนคลิปปกติ | คำขอ Qwen |
|---|---:|---:|---:|
| คลังปกติอย่างเดียว | 1/1 | 0 | 0 |
| ข้อความอย่างเดียว | 0/1 | 0 | 0 |
| รวมคลังปกติและข้อความ | 1/1 | 0 | 0 |
| รวมสองทาง แล้วตรวจด้วย Qwen | 1/1 | 0 | 2 |

เกณฑ์เลือกจาก validation ก่อนเรียก Qwen: novelty .05, semantic margin .1,
minimum semantic similarity .35 สำหรับแต่ละช่องทาง ผลชี้ว่าคลังปกติช่วยคัดคลิปนี้
ที่ข้อความพลาด แต่ยังไม่พิสูจน์ว่าช่วยทุกฉากหรือทุกประเภทเหตุ

Qwen ตรวจหน่วย 0–12 และ 12–15.37 วินาทีเป็น suspicious ทั้งคู่ เมื่อรวมช่วงซ้อน/ติดกัน
ก่อนจับคู่แบบหนึ่งต่อหนึ่งที่ temporal IoU ≥.3 ได้หนึ่งเหตุจริง ไม่รายงานเป็นสองเหตุ
คำอธิบายของ Qwen เป็นผลโมเดลที่ผู้ใช้ต้องตรวจ ไม่ใช่ label ความหมายที่ยืนยันแล้ว

**ห้ามตีความ 1/1 เป็นความแม่นทั่วไป 100%**: มีเหตุเพียงหนึ่งเหตุและวิดีโอปกติ
15.03 วินาที การไม่พบแจ้งผิดในเวลาสั้นนี้ยังยืนยัน ≤2 ครั้งต่อชั่วโมงไม่ได้
ช่วงความไม่แน่นอนจาก bootstrap ที่มีเพียงสองคลิปก็ไม่ใช่หลักฐาน generalisation
ไม่ผ่านการรับรองเกณฑ์ production จึงเก็บสถานะทดลองไว้

## ทรัพยากรและ baseline ที่วัดจริง

ให้ Qwen ตรวจทุกหน่วย 12 วินาทีใน validation ทั้งสองคลิป ด้วย prompt บริบท
งบเฟรม และการเลือกตัวอย่างปกติแบบเดียวกัน:

- baseline เรียก Qwen จริง 4 คำขอ ใช้เวลา 55.49 s จับเหตุได้ 1/1
- คัดก่อนเหลือ 2 คำขอ ลด 50% ในชุดนี้ เวลาเฉพาะ Qwen รอบสุดท้าย 28.25 s
- รอบสุดท้ายทั้งหมด 44.48 s **ใช้ดัชนีที่มีแล้ว** รวมการโหลด Gemma สำหรับข้อความ
  จึงไม่ใช่เวลา cold indexing + end-to-end จากวิดีโอเปล่า
- Qwen รอบสุดท้าย: whole-card GPU peak ที่สุ่มทุก .5 s ≈11.51 GiB;
  RAM ของ Qwen และโปรเซสลูก ≈6.56 GiB
- ระหว่างสร้างดัชนีจริงก่อนหน้า: Gemma PyTorch allocated peak ≈1.82 GiB,
  reserved peak ≈2.08 GiB; รวมเวลา decode ของสี่คลิป ≈67.57 s และ encode ≈57.54 s
  เป็นผลรวมเวลาแต่ละขั้น ไม่ใช่ timer end-to-end เดียว

baseline นี้เทียบหน่วยตรวจ 12 วินาที ไม่ใช่หน้าต่างดัชนี 6 วินาทีที่เหลื่อมกัน
จำนวน baseline สมมติที่เว็บแสดงจากหน้าต่างดัชนีจึงเป็นอีกตัวเลขและระบุว่ายังไม่ได้รัน
VRAM รวมทั้งการ์ดรวมโปรแกรมอื่นด้วย ไม่ใช่ PyTorch allocation ของ Qwen
ไม่รวมการซ่อม JSON ภายใน private worker ในจำนวนคำขอ HTTP และไม่มี OOM ในรอบนี้

## หลักฐานและสถานะส่งมอบ

- `scratch/event-retrieval/datasets/ubnormal-open/manifest.json`: แหล่งข้อมูล/hash/split/label
- `scratch/event-retrieval/open-anomaly-ubnormal-smoke-v1`: ดัชนีจริงและ protocol รอบแรก
  ซึ่งหยุดที่ตัวประเมินเพราะเวลา tail ถูกปัดเกิน duration ต่ำกว่า 1 μs
- `scratch/event-retrieval/open-anomaly-ubnormal-smoke-v2`: ผลครบสี่วิธีและ code hashes
- `scratch/event-retrieval/open-anomaly-ubnormal-baseline-v1`: baseline ที่รันจริง
- `scratch/event-retrieval/open-anomaly-ubnormal-smoke-v3`: ตรวจซ้ำหลังเพิ่มการรวม telemetry
  และ source ledger; ผลเท่าเดิม, code_unchanged=true ตอนจบการรัน
- หลังรัน เพิ่ม guard ไม่ให้คลิปที่ทราบว่าเป็น validation/test ถูกประกาศใหม่เป็น development
  พร้อม test โดยไม่เปลี่ยน scoring/prompt/threshold ของผลที่วัดแล้ว
- นำผล v3 เข้าเว็บเป็น demo ที่ติดป้าย synthetic validation subset พร้อมคลังปกติ
  เฉพาะสองคลิป development; ไม่ติดตั้ง threshold ของชุดเล็กเป็นค่าเริ่มต้น

คลิป ดัชนี ผลดิบ และ environment อยู่ในพื้นที่ ignored ไม่เข้า Git
เพิ่มการทดสอบสำหรับ API, immutable versions, non-finite vectors, missing references,
uncertainty, errors/cancel/resume, timestamp sampling, leakage, telemetry aggregation
และการยืนยันตัวอย่างปกติในเว็บ รวมทั้งตรวจ TypeScript

งานที่ยังต้องทำก่อนสรุปประสิทธิภาพทั่วไป: เข้าถึง ShanghaiTech หรือข้อมูลกล้องจริง
ที่มี label เพิ่มคลิปปกติยาวและหลายฉาก แล้วรัน development/validation/test แยกกัน
รวมเหตุที่ไม่เคยใช้ปรับเกณฑ์และเสียง ยังไม่อ้างว่าแทนระบบเดิมสำเร็จจากชุดเล็กนี้
