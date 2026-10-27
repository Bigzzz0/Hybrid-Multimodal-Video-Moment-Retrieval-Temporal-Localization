# ผลทดสอบคลิป Kamakura: EmbeddingGemma 2 แบบ Video + Audio

ทดสอบคลิปเต็ม Ghibli-Like Seaside Town / Kamakura One Day Train Trip near Tokyo / JAPAN TRAVEL VLOG จากไฟล์ของผู้ใช้

ความยาว 1183.42 วินาที · 1183 เฟรม · 198 ช่วง · 20 คำค้นจาก 10 เหตุการณ์ ไทย–อังกฤษ

| โหมด | CenterFrameHit@1 | @5 | ค้นหา p50 | ทำดัชนี | Peak allocated GPU | Peak RAM ขณะทำดัชนี |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 · เฉลี่ยเฟรม | 85.0% | 95.0% | 13.18 ms | 17.22 s | 771.2 MiB | 2554.5 MiB |
| Gemma 2 · เฉลี่ยเฟรม | 80.0% | 100.0% | 116.61 ms | 95.74 s | 1297.8 MiB | 2446.5 MiB |
| Gemma 2 · Native Video + Audio | 100.0% | 100.0% | 120.58 ms | 527.37 s | 4008.1 MiB | 3093.1 MiB |

## ทรัพยากรเพิ่มเติมขณะทำดัชนี

| โหมด | CPU core-seconds | GPU utilization เฉลี่ย | กำลังไฟ GPU เฉลี่ยทั้งเครื่อง | ช่วง/วินาที |
|---|---:|---:|---:|---:|
| SigLIP2 · เฉลี่ยเฟรม | 44.11 | 7.8% | 52.4 W | 11.497 |
| Gemma 2 · เฉลี่ยเฟรม | 168.16 | 59.4% | 142.7 W | 2.068 |
| Gemma 2 · Native Video + Audio | 861.75 | 88.0% | 208.8 W | 0.375 |

## วิธีทดสอบและขอบเขต

- เปิด text, vision และ audio encoder ครบ 744,371,488 parameters; ใช้ BF16 ตามคำแนะนำโมเดล
- Native AV ป้อนเฟรมเรียงตามเวลา สูงสุด 6 ภาพต่อช่วง พร้อมเสียงจริง 16 kHz mono ไม่ป้อนชื่อบท คำตอบอ้างอิง หรือ caption ที่สร้างขึ้น
- ใช้ vision budget สูงสุด 1,120 soft tokens ต่อเฟรมและ relative timestamps ปิด truncation มีการตรวจ input shapes ว่าภาพและเสียงเข้าโมเดลและลำดับไม่เกิน 8,192 tokens
- ทดลองเสียงจริงเทียบกับเสียงเงียบในภาพชุดเดิม: cosine 0.997493, L2 difference 0.070814; ยืนยันว่าเสียงมีผลต่อเวกเตอร์ ไม่ได้ยืนยันความแม่นในการฟัง
- โหมดภาพเดี่ยวทั้งสองใช้ค่าเฉลี่ยเวกเตอร์เฟรมภายในช่วงเดียวกัน จึงไม่ใช่ protocol ของ benchmark 28 คำค้นเดิม
- Warmup ทั้ง media และ query ก่อนวัด แยกเวลาโหลดออกจาก indexing และแต่ละ query วัด 5 ครั้ง ทุกโมเดลทำงานแยก process และไม่รันพร้อมกัน
- เวลาทำดัชนีรวมการอ่าน JPEG และ model encoding แต่ไม่รวมการถอดคลิปและเตรียม WAV ซึ่งเตรียมร่วมกันก่อนทุกโหมด
- Peak allocated GPU เป็นค่าจาก PyTorch; RAM เป็น process RSS สูงสุดที่สุ่มวัดทุก 1 วินาที ส่วน GPU utilization/กำลังไฟเป็น device-wide จาก nvidia-smi ไม่ใช่พลังงานเฉพาะงาน
- CenterFrameHit วัดภาพกลางช่วงที่ตรวจครบทั้ง 198 ช่วงก่อนดูผลโมเดล ไม่มีการวัด exact temporal IoU, audio recognition หรือ no-match queries
- ชุดนี้มีเพียงคลิปเดียวและ 10 แนวคิดที่เขียนสองภาษา จึงไม่อ้างว่าโมเดลชนะในทุกงาน และยังไม่ได้แยกผลของความละเอียด หลายเฟรม และเสียงออกจากกัน

## ตรวจซ้ำ

API integration tests 4 ข้อ, input-routing tests 2 ข้อ และ frontend production build ผ่าน

ผลดิบ เวกเตอร์ ภาพที่ตรวจ และ input-verification.json อยู่ใน scratch/embeddinggemma2/kamakura

[คู่มือ Google](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
