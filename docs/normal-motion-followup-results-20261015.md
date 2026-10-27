# Qwen: เปรียบเทียบกับภาพลำดับเวลาจากคลิปปกติ

## วิธีที่ปรับ

- เปลี่ยนภาพปกติอ้างอิงเดี่ยวให้เป็นภาพลำดับ4เวลาในกรอบ2×2 จากหน้าต่างปกติเดียวกัน พร้อมเวลาและป้าย NORMAL
- คง Qwen CapRL-Qwen3VL-4B Q6, เฟรมเหตุปัจจุบัน, timestamps, จำนวนภาพอ้างอิงสูงสุด2ภาพ, ขนาดcanvas และช่วงตรวจเดิม
- เพิ่มคำอธิบายว่า reference มาจากคนละคลิป และให้เปรียบเทียบลักษณะการกระทำตามเวลา; ไม่ใช้เฟรมอ้างอิงเป็นเหตุปัจจุบัน
- ทุก reference มาจาก development-normal; ถ้ามีไม่ถึง4เฟรมต่างกันใช้ภาพเดิมและบันทึก fallback ไม่สร้างการเคลื่อนไหวขึ้นเอง
- ไม่ฝึกน้ำหนัก ไม่ปรับ threshold จากชุด follow-up และไม่ได้วัดจำนวน visual tokens จริง

## เหตุผลที่เป็นการทดลองต่อ

ชุดฉาก6/8/14/16 ก่อนหน้านี้เพิ่มจาก3เป็น4เหตุ ที่FP2เท่าเดิม แต่เพิ่ม uncertain หนึ่งช่วง จึงไม่ผ่าน promotion เดิม การทดลองนี้ประกาศ amendment ก่อน inference ฉาก22/24 เพื่อศึกษาต่ออย่างเปิดเผย ไม่ถือว่าวิธีผ่านเกณฑ์เดิมแล้ว

## ผล event matching: temporal IoU≥0.3

| ชุด | วิดีโอ / เหตุ | วิธี | TP | FP | FN | Recall | Precision | รอตรวจทั้งหมด |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| previous_regression | 12 / 6 | control | 3 | 2 | 3 | 50.0% | 60.0% | 0 |
| previous_regression | 12 / 6 | normal_motion | 4 | 2 | 2 | 66.7% | 66.7% | 1 |
| old_regression | 10 / 7 | control | 6 | 1 | 1 | 85.7% | 85.7% | 0 |
| old_regression | 10 / 7 | normal_motion | 6 | 1 | 1 | 85.7% | 85.7% | 0 |
| fresh_followup | 6 / 3 | control | 2 | 0 | 1 | 66.7% | 100.0% | 0 |
| fresh_followup | 6 / 3 | normal_motion | 2 | 1 | 1 | 66.7% | 66.7% | 1 |

รอตรวจทั้งหมดใช้ค่าuncertain_unitsจากmetricเดิม ซึ่งรวมทั้งคำตอบuncertainที่ถูกschemaและtechnical-unverifiedที่ยังไม่มีverdict; แยกสองชนิดในsummary.json ไม่ถือเป็นเหตุที่ยืนยันถูก
fresh_followup เปรียบเทียบทุก12ช่วงเพื่อแยกผลของQwen; previous/oldใช้ payload ที่ล็อกในงานก่อนหน้า ผลแต่ละชุดจึงต้องอ่านแยกกัน

## ผลต่างและข้อสรุป

- previous_regression: เหตุที่ได้เพิ่ม [('abnormal_scene_16_scenario_1', 0)]; เหตุเดิมที่หาย []; รักษาผลเดิมโดยFP/uncertainไม่เพิ่ม=False; ผ่านเกณฑ์promotionเดิมที่ต้องมีimprovementด้วย=False
  คำตอบuncertainจริง={'control': 0, 'normal_motion': 1}; technical-unverified={'control': 0, 'normal_motion': 0}
- old_regression: เหตุที่ได้เพิ่ม []; เหตุเดิมที่หาย []; รักษาผลเดิมโดยFP/uncertainไม่เพิ่ม=True; ผ่านเกณฑ์promotionเดิมที่ต้องมีimprovementด้วย=False
  คำตอบuncertainจริง={'control': 0, 'normal_motion': 0}; technical-unverified={'control': 0, 'normal_motion': 0}
- fresh_followup: เหตุที่ได้เพิ่ม []; เหตุเดิมที่หาย []; รักษาผลเดิมโดยFP/uncertainไม่เพิ่ม=False; ผ่านเกณฑ์promotionเดิมที่ต้องมีimprovementด้วย=False
  คำตอบuncertainจริง={'control': 0, 'normal_motion': 0}; technical-unverified={'control': 0, 'normal_motion': 1}
- จะพิจารณาใช้เป็นค่าเริ่มต้นเมื่อรักษาเหตุเดิมได้ครบ เพิ่มTPโดยFPไม่เพิ่ม หรือTPเท่าเดิมแต่FPลด และ uncertain ไม่เพิ่มในแต่ละชุด
- ผ่านเกณฑ์เดิมครบทุกชุด: False; production activation=False
- ผลดีบนชุดเล็กบางฉากไม่ใช่หลักฐานว่าดีขึ้นทั่วไป ต้องเก็บวิธีและผลที่ไม่ผ่านไว้ ไม่เลือกตัวเลขดีที่สุดจากทุกชุดมารวมอ้าง

## ฉากสำรองเมื่อใช้ Gemma gate เดิม

Gemmaเลือก 5/12 หน่วย; ค่าด้านล่างกรองจากคำตอบจริงของall-gridด้วยselected IDsเดิม เป็นreplayไม่ใช่การรันเพิ่มหรือเวลาที่วัดแยก

| ขั้นตอน | TP | FP | FN | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Gemma selected grid | 2 | 1 | 1 | 66.7% | 66.7% |
| control | 2 | 0 | 1 | 66.7% | 100.0% |
| normal_motion | 2 | 0 | 1 | 66.7% | 100.0% |

## เวลา

| ชุด | เตรียมreference CPU (s) | control Qwen (s) | normal-motion Qwen (s) |
|---|---:|---:|---:|
| previous_regression | 135.81 | 267.70 | 141.87 |
| old_regression | 1.07 | 126.73 | 237.22 |
| fresh_followup | 0.74 | 75.00 | 74.77 |

รันแต่ละวิธีเป็นphaseต่อกัน ไม่ใช่การทดสอบเวลาสลับลำดับหลายรอบ จึงไม่สรุป speedup; เวลานี้ไม่รวม Gemma indexing ซึ่งเก็บแยกในผลดิบของ fresh-source
Qwen รันผ่าน llama.cpp จึงดู PyTorch allocated ของbrokerเป็นVRAMของQwenไม่ได้; เก็บ sampled whole-card GPU peak และ process-tree RAM แยกไว้ใน summary.json โดย GPU รวมโปรแกรมอื่นด้วย

## ขอบเขตและการทำซ้ำ

- UBnormal เป็นภาพจำลอง; previous/old เป็น regression ที่ตรวจมาแล้ว ส่วน fresh เป็น validation ฉากสำรองที่เลือกและดาวน์โหลดไว้ก่อน ไม่มี official test
- แหล่งและสิทธิ์: https://github.com/lilygeorgescu/UBnormal — CC-BY-NC-ND-4.0; manifest มี recording IDs, source revision และ SHA256
- การอนุมานความผิดปกติขึ้นกับบริบทพื้นที่; binary label ไม่ได้ยืนยันชื่อการกระทำหรือเจตนาของบุคคล
- normal exposure สั้น จึงยังรับรอง≤2แจ้งผิด/ชั่วโมงไม่ได้; เก็บ95%video-bootstrap intervals ใน summary.json พร้อมจำนวนตัวอย่าง
- Full JSON คำตอบจริง, ภาพอ้างอิง, hashes และ protocol เก็บในรายงาน ไม่แก้คำตอบย้อนหลัง ไม่เปลี่ยนหน้าdemoหรือmerge main

`python -m evaluation.embeddinggemma2.compare_overview_detail --source <frozen-requests> --payload-key <selected|uniform> --manifest <manifest> --reference-store <store> --arms control normal_motion --output <fresh-run>`
