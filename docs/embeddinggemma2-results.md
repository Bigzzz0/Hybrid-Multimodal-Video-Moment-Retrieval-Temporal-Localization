# ผลทดลองบร้านช์ EmbeddingGemma 2

ทดสอบบน RTX 5070 12 GB วันที่ 7–8 ตุลาคม 2026 สถานะ: **ต้นแบบทดลอง ยังไม่ผ่าน acceptance และยังสรุปไม่ได้ว่าแทน SigLIP2 สำเร็จ** คู่มือและคำสั่งรันอยู่ใน [embeddinggemma2-events.md](embeddinggemma2-events.md)

## Retrieval แยกปัจจัย

ใช้วิดีโอ regression เดิม 2 คลิป, 7 event annotations, 28 คำค้นที่มีคำตอบ (ไทย 14/อังกฤษ 14) และ 4 คำค้นไม่มีคำตอบ ทุก arm ใช้เฟรมเดียวกันตาม configuration 6 วินาที/stride 3 วินาที, image 1 FPS, video 2 FPS, 12 เฟรม, video 140/image 280 tokens และ 768 มิติ วัด hit ที่ temporal IoU≥0.3

| ช่องทาง | TH R@1 | TH R@5 | EN R@1 | EN R@5 | PyTorch allocated peak (GiB) | ทำดัชนีรวม (s) |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 image | 50.0% | 85.7% | 28.6% | 85.7% | 0.754 | 93.77 |
| Gemma image | 42.9% | 85.7% | 35.7% | 71.4% | 1.497 | 105.17 |
| Gemma video | 42.9% | 64.3% | 21.4% | 85.7% | 1.829 | 94.86 |
| Gemma audio | 0% | 20.0% | 0% | 20.0% | 1.436 | 10.76 |
| Gemma video + audio RRF | 28.6% | 57.1% | 7.1% | 71.4% | 1.830 | 95.82 |
| Gemma ทุกช่องทางที่มี | 28.6% | 64.3% | 42.9% | 71.4% | 1.830 | 127.51 |

Caption ของสองคลิป regression ไม่มี จึงไม่เติม Caption ปลอมหรือรายงานคะแนนเป็นศูนย์ของ arm ที่ไม่ได้ทดสอบ Audio-only วัดเพียงคลิปที่มีเสียง: ไทย 10/อังกฤษ 10 answerable queries (5 visual events) ตัวเลขจึงมี denominator ต่างจาก arm ภาพ ทั้งคลิปมีคำค้นที่อิงภาพเป็นหลัก เสียงไม่ได้มีข้อมูลตอบทุกคำค้น

Gemma image และ SigLIP2 image มี R@1 รวมเท่ากัน 39.3%; SigLIP2 R@5 รวม 85.7% และ Gemma image 78.6% การเพิ่มช่องเสียงด้วย RRF ในชุดนี้ลดผลของคำค้นภาพ จึงยังไม่ควรใช้ equal-weight fusion เป็น configuration สุดท้ายโดยไม่มี validation แยกงาน

ทุก arm ที่มีข้อมูลยังคืน top-K ให้คำค้นไม่มีคำตอบทั้ง 4 ข้อ: no-match false-positive rate = 100% ระบบยังไม่มี rejection threshold ที่ผ่าน validation เป็นข้อบกพร่องที่ต้องปรับก่อนอ้างว่าใช้งาน no-match ได้ การไม่แสดง similarity เป็น probability ไม่ได้แก้ปัญหา no-match

ข้อจำกัดของ protocol: คำถามเป็น paraphrases ของเพียง 7 เหตุจาก 2 วิดีโอ ไม่ใช่ 28 เหตุอิสระ, annotation เดิมอาจไม่ครอบคลุมทุกช่วงที่ตอบคำค้นได้ และเหตุ students-sitting มี GT ยาว 32.8 วินาที ซึ่งช่วงผลลัพธ์ 6 วินาทีมี IoU สูงสุดเพียง 0.183 จึงไม่ผ่านเกณฑ์ 0.3 แม้ภาพถูกชนิดแล้ว ห้ามใช้ตัวเลขชุดนี้สรุปความเหนือกว่าทั่วไปหรือแทนผล held-out test

raw reports เก็บที่ `scratch/event-retrieval/ablations/*.json` มี configuration, รายคำค้น/อันดับ, Wilson intervals ตามจำนวนคำค้น, telemetry และ joint AV runs การตีความ CI ต้องระวังความสัมพันธ์ของ paraphrases จำนวนวิดีโอมีเพียง 2 คลิป

## Caption semantic search

ทดสอบแยกด้วย UT clip ที่เป็น action-trimmed punch/push ตามต้นทาง และใช้ Caption ที่ Qwen สร้างจริงเป็น input คำตอบ ground truth ใช้ action label ของต้นทาง ไม่ได้ใช้ Caption ที่โมเดลสร้างมาเป็นเฉลย

ไทย/อังกฤษอย่างละ 1 คำค้นที่มีคำตอบได้ R@1/R@5 = 100%; คำค้นไม่มีคำตอบอย่างละ 1 ยังถูกคืน top-K ทั้งหมด First-query p95 = 242 ms นี่เป็น smoke test ของความเชื่อมต่อ Caption → text embeddings เท่านั้น มีเพียงเหตุเดียวจึงไม่ใช่ผลยืนยันคุณภาพ Caption search

artifact: `scratch/event-retrieval/caption-smoke/gemma2_caption.json`

## Kamakura และเวลาใช้งานจริง

- ทำดัชนีครบ **395/395 ช่วง** ของวิดีโอ 19:43.4 นำ Caption เดิม **176 รายการ** เข้าแบบเต็มและสร้าง vectors ใหม่
- เวลาทำดัชนีสะสม: decode 594.92 s และ encode 902.49 s มีการ cancel/restart/resume จริง และเปิด active index หลังครบเท่านั้น
- วัดผ่าน HTTP API 12 คำค้น × 5 รอบซ้ำ: โหลดน้ำหนักกลับหลัง unload + คำค้นแรก **5.84 s**, คำค้นใหม่หลังจากนั้น p50 **374 ms**, p95 **504 ms**, คำค้นซ้ำที่มี query cache p95 **353 ms**
- 5.84 s เป็นการโหลดน้ำหนักกลับใน worker ที่ libraries ถูก initialize แล้ว ไม่ใช่การเริ่ม process ทั้งหมดใหม่ ในการเริ่มต้น worker ก่อนหน้านี้มี model-load telemetry ประมาณ **16.3 s** แยกสองสถานการณ์นี้เพื่อไม่อ้างว่าการเปิดเครื่องครั้งแรกใช้เพียง 5.84 s
- p95 แบบเรียก service โดยตรงบนคลิปสั้นและ query cache ประมาณ 7–24 ms ไม่เทียบตรงกับ HTTP API บน Kamakura 395 ช่วง

artifact: `scratch/event-retrieval/kamakura-runtime.json` ผลนี้วัดเวลา ไม่ได้อนุมานว่า 12 คำค้นนี้มีความแม่นเท่าใด

## คัดกรองและ Qwen

เดโมจริงตรวจพบ candidate ใน UT punch/push และ UR fall และ Qwen ให้ `event_present=true` คลิป UR กิจวัตรปกติตัวอย่างไม่ถูกเสนอเป็นเหตุล้ม AudioSet Shout ตัวอย่างมี candidate เสียง แต่ Qwen ให้เพียง visual context และ `audio_verified=false` เหตุทั้งหมดคง `review=pending` จนกว่าผู้ใช้จะตรวจเอง

การมี `event_present=true` ไม่ใช่หลักฐานว่าขอบเขตแม่นแล้ว: UR ตัวอย่างถูก refine เป็น 3.0–5.33 s แต่ป้ายกำกับ temporary-falling ทางการอยู่ประมาณ 2.73–3.73 s และยังต้องตรวจ alignment ของ preview ขอบเขตปลายยังรวมช่วงนอนบนพื้น จึงไม่อ้างผ่าน final recall/temporal IoU จากเดโมนี้

เปรียบเทียบ Qwen แบบหน่วยเดียวกันจริงบน UT clip 2 windows × 2 visual event types ใช้ prompt/เฟรมต่อหน่วยเหมือนกันทั้งสอง arm:

| วิธี | Calls (caption + verifier) | เวลา batch รวม |
|---|---:|---:|
| ตรวจทุกหน่วย | 8 | 23.75 s |
| ตรวจเฉพาะหน่วยที่คัดกรอง | 2 | 9.70 s |

ลด calls 75% และเวลาประมาณ 59.1% ในคลิปเดียว ผลทั้งสอง arm ตอบ JSON ได้ครบ (`verified` ซึ่งหมายถึงการตรวจเสร็จ ไม่ใช่ทุกหน่วยเป็นเหตุจริง) ไม่คาดเดาเวลาจากจำนวน calls และยังไม่ใช้ตัวเลขนี้อ้างว่าจะลด ≥50% ได้กับชุด held-out ทั้งหมด

Qwen ใช้ llama.cpp จึงไม่รวมอยู่ใน PyTorch allocated ของ broker ระหว่างเดโมสุ่มวัด VRAM **รวมทั้งการ์ด** ประมาณ 11.21 GiB และ process-tree RAM ของ Qwen ประมาณ 4.61 GiB ที่สูงสุด ตัวเลข GPU รวมโปรแกรมอื่นและเป็น sampled peak ไม่ใช่ VRAM เฉพาะ Qwen โดยตรง Gemma peak 1.83 GiB กับ whole-card peak ระหว่าง Qwen เป็นคนละตัวชี้วัด

artifact: `scratch/event-retrieval/qwen-comparison.json`, database job telemetry ใน `backend/data/gemma2/events.sqlite3`

## ข้อมูลและ acceptance ที่ยังไม่ครบ

ดาวน์โหลดวิดีโอที่อ่านได้จริง **210 คลิป**: UT 120, UR 60, AudioSet AV 30 พร้อม FSD audio-only previews อีก 60 ไฟล์ เก็บ URL/SHA/split ตาม recording และไม่มีไฟล์คลิปเข้า Git

dataset audit ยังเป็น `ready=false`: AudioSet Shout ต้องฟัง/ดูและระบุช่วงจริง, UR preview ต้องตรวจ frame/time alignment, ยังไม่มี 30 shouting hard negatives ที่ยืนยันแล้ว และวิดีโอ normal ต่อเนื่อง 90 นาที ยังไม่ปรับ threshold ด้วย test หรือเปิด held-out test เพื่อเลือกค่า

VIRAT เป็นแหล่ง surveillance ที่น่าพิจารณาเพิ่มเติม แต่ ground data ต้องใช้ตาม [Dataset Protection Agreement](https://viratdata.org/) จึงยังไม่นับคลิปที่ยังไม่ได้รับสิทธิ/ตรวจ annotation เป็น normal ground truth

ยังไม่รัน acceptance grid 140/280/560/1120 หรือเลือก MRL dimension เป็นค่าชนะ เพราะยังไม่มี validation set ที่ผ่าน audit เครื่องมือรันและล็อก configuration เตรียมไว้แล้ว ไม่มีการอ้างว่าทุกเป้าหมายผ่านหรือ merge เข้า main

## การตรวจซอฟต์แวร์

Backend 70 tests, runtime/worker 9 tests, experimental API/input-routing 6 tests และ frontend 7 tests ผ่าน รวม 92 tests; frontend production build และ TypeScript ผ่าน ทดสอบ namespace mismatch, activation atomic, restart/resume/cancel, OOM, no-audio/silent audio, dropped media/context overflow, malformed Qwen JSON, review preservation, overlapping-frame deduplication และ HTTP Range

ในเครื่องนี้ไม่มีคำสั่ง `graft` หรือ graft MCP จึงอ่าน graph cards ของ checkout เดิมก่อนเปิด source ตาม AGENTS แต่ยัง refresh graph ด้วย `graft build` ไม่ได้ ไม่รายงานว่า graph ถูกสร้างใหม่แล้ว
