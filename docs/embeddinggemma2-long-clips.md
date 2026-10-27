# ผลทดสอบคลิปยาว — 8 ตุลาคม 2026

ใช้บร้านช์ `codex/embeddinggemma2-event-retrieval` configuration เดิม 768 มิติ, ช่วง 6 วินาที/เลื่อน 3 วินาที, image 1 FPS, video 2 FPS/12 เฟรม/140 tokens, BF16 บน RTX 5070

## คลิปและเวลาที่วัดจริง

| คลิป | ความยาว | ดัชนีครบ | เวลา index ผ่าน API | เวลา screen ภาพ ผ่าน API | Gemma allocated peak |
|---|---:|---:|---:|---:|---:|
| Fighting011_x264.mp4 | 274.67 s | 92/92 | 135.37 s | 215.63 s | 1.828 GiB |
| Normal_Videos_634_x264.mp4 | 448.66 s | 150/150 | 281.26 s | 5.05 s | 1.829 GiB |

API wall time รวมการจัดคิวและ polling ทุก 5 วินาที (จึงมี overhead ≤ประมาณ 5 วินาทีเมื่อทรัพยากรว่าง); job elapsed และ decode/encode แยกอยู่ใน test-results.json

## ผลที่พบ

- Fighting011: ภาพตัวอย่างที่ตรวจด้วยตายืนยันว่ามีการปะทะ แต่ยังไม่มี exact onset/offset ground truth คลิปไม่มีเสียง
- ค้นภาษาไทย “คนผลักหรือชกต่อยกัน”: Top 1 = 75–81 s; Top 2 = 72–78 s ภาพ sampled ที่ประมาณ 73–92 s มีการปะทะ ผลเป็นการตรวจเชิงคุณภาพ ไม่ใช่ Recall/Precision
- คัดกรองภาพเสนอ 86 window hits แล้วรวมเป็น 2 รายการ: 0–12 s และหลัง refine 21–274.67 s
- Qwen ตรวจ 0–12 s เสร็จและให้ event_present=false จึงเป็น candidate ที่โมเดลไม่ยืนยันว่ามีเหตุ
- รายการ 21–274.67 s ยัง unverified เนื่องจาก Incomplete Qwen caption JSON; ไม่ตีความเป็นไม่พบเหตุ ไม่ลบข้อมูล
- Normal634: ไม่เสนอ candidate ผลัก/ชกหรือล้มจากภาพตลอด 150 ช่วง
- ตรวจ shouting แยกบนเสียงของ Normal634 ก็ไม่เสนอ candidate; audio stream มีจริง แต่ยังไม่ได้ยืนยันว่า soundtrack ตรงกับภาพ/สถานการณ์โรงพยาบาล
- เวลาตรวจเสียง normal ผ่าน API 5.02 s; job elapsed 1.25 s

## ทรัพยากรและคำค้น

Fighting011 ใช้ Qwen จริง 3 calls (หนึ่งรายการ caption+verifier และอีกหนึ่งรายการ caption ไม่ครบ) ตัวเลข baseline 368 ในงานนี้เป็นจำนวนที่ระบบคำนวณจากทุก window × 2 visual types × 2 calls ไม่ใช่เวลาหรือ calls ของ control ที่รันจริง ห้ามใช้เป็นหลักฐานประหยัดเท่าใดบนคลิปยาว

### Fighting011_x264.mp4
- `คนผลักหรือชกต่อยกัน`: HTTP 3.753 s; Top 1 75.0–81.0 s
- `คนล้มลงบนพื้น`: HTTP 0.154 s; Top 1 96.0–102.0 s
- `people pushing or punching each other`: HTTP 0.124 s; Top 1 108.0–114.0 s

### Normal_Videos_634_x264.mp4
- `คนผลักหรือชกต่อยกัน`: HTTP 0.072 s; Top 1 213.0–219.0 s
- `คนล้มลงบนพื้น`: HTTP 0.069 s; Top 1 42.0–48.0 s
- `people pushing or punching each other`: HTTP 0.047 s; Top 1 195.0–201.0 s

ช่วง Qwen: whole-card sampled VRAM peak 11.38 GiB รวมโปรแกรมอื่น; Qwen process-tree sampled RAM peak 4.70 GiB (sampling 0.5 s) ตัวชี้วัดนี้ต่างจาก PyTorch allocated ของ Gemma

## สิ่งที่ต้องปรับจากการทดสอบนี้

1. อย่าให้ candidate ที่ต่อกันลากรวมช่วงเป็นเหตุยาวหลายร้อยวินาที ต้องแยกตามหลักฐานการกระทำและตรวจเหตุย่อย เพื่อไม่สูญเสีย onset/offset
2. การตรวจเพียง 8 เฟรมบนเหตุยาวทำให้เฟรมห่างเกินไป ควรแบ่งตรวจช่วงย่อยและจัดงบงานให้ชัดเจน
3. ปรับความกระชับของ Caption/prompt และขีดจำกัดคำตอบ พร้อม retry เฉพาะเหตุที่ JSON ไม่ครบ เก็บผลก่อน/หลังเพื่อเทียบ ไม่เปลี่ยนระหว่างรอบนี้
4. ทดสอบ normal ที่มีคนเดิน/เคลื่อนไหวใกล้เคียงกับโรงพยาบาลเพิ่ม คลิปถนนระยะไกลตัวเดียวไม่พิสูจน์ความแม่นในโรงพยาบาล

ไม่มี ground truth ที่ตรวจครบทั้งคลิป จึงไม่รายงาน Precision/Recall/false alarms per hour หรือถือว่าผ่าน acceptance ชุดนี้เป็น exploratory demo และไม่ใช้เลือก threshold แล้วอ้างว่าเป็น held-out test

## แหล่งข้อมูล

- [UCF-Crime ต้นทาง](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
- [ไฟล์เผยแพร่ผ่าน MTEB](https://huggingface.co/datasets/mteb/ucf-crime)
- URL รายคลิป/SHA/streams: acquisition.json; configuration/protocol: protocol.json; ผลดิบ: test-results.json และ normal-audio-screen.json
- ใช้คลิปเต็มตามต้นทาง ไม่วนซ้ำหรือเอาคลิปสั้นมาต่อให้ยาว ไม่ใส่คลิปเข้า Git สิทธิของต้นทางยังคงใช้บังคับ
