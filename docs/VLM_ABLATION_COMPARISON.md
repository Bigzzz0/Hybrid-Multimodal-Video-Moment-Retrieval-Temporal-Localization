# VLM A–E Ablation Comparison

สถานะ: research/demo บน branch codex/vlm-model-ablation

วันที่บันทึกผล: 2026-09-21 ถึง 2026-09-22

เอกสารนี้เปรียบเทียบ VLM ทั้ง 6 runtime variants สำหรับระบบค้นหาช่วงเวลาในวิดีโอ โดยเน้นกรณีใช้งานจริงของโปรเจกต์คือการสร้าง caption ล่วงหน้าหลังอัปโหลดวิดีโอ แล้วนำ caption artifacts ไปช่วย Fast/Accurate search

## 1. รุ่นที่ทดสอบ

| ID | รุ่น | Runtime | Input | บทบาทที่เหมาะสม |
|---|---|---|---|---|
| A | Qwen3-VL-2B-Instruct | Transformers NF4 | frames | baseline และ fallback |
| B | Qwen3-VL-2B + VISE | Transformers NF4 + LoRA | frames | งานทดลอง adapter |
| C | CapRL-Qwen3VL-2B | Transformers NF4 | frames | caption ที่ละเอียดขึ้น |
| D-Q4 | CapRL-Qwen3VL-4B-Q4_K_M | llama.cpp CUDA | frames | คุณภาพ 4B แบบประหยัดกว่า Q6 |
| D-Q6 | CapRL-Qwen3VL-4B-q6_k | llama.cpp CUDA | frames | caption offline คุณภาพสูง |
| E | CapRL-Video-4B | Transformers NF4 | video chunk / sampled frames | เข้าใจลำดับเหตุการณ์ |

ทุกรุ่นใช้ local inference เท่านั้น ไม่มีการส่ง frame หรือวิดีโอขึ้น cloud และไม่มี face recognition, Identity หรือ Re-ID

## 2. วิธีทดสอบ smoke test

- วิดีโอ: คลิป Ghibli-like seaside town จาก local keyframe store
- ใช้ keyframe ชุดเดียวกันสำหรับทุกโมเดล
- ใช้ 10 เฟรมแรก แบ่งเป็น 4 + 4 + 2 เฟรม
- ใช้ prompt เดียวกันและ schema เดียวกัน: summary, objects, attributes, actions, relations, temporal_events, uncertainty
- ห้ามใช้ OCR, subtitle, audio หรือ identity เป็นหลักฐาน
- เรียก worker แบบ sequential และ release หลังจบแต่ละรุ่น
- ผลนี้เป็น smoke/regression test ไม่ใช่ held-out accuracy benchmark

### ฉากที่อยู่ใน 10 เฟรม

1. ชายหาด นักโต้คลื่น ทะเล และรูปปั้น
2. รูปปั้น ป่าไผ่ ทางเดิน โคมไฟ และรถไฟ
3. คนกำลังเทมัทฉะ/เครื่องดื่มลงแก้ว

## 3. เวลาในการประมวลผล

เวลาที่วัดได้จาก 10 เฟรม:

| รุ่น | Chunk 1 | Chunk 2 | Chunk 3 | รวมโดยประมาณ |
|---|---:|---:|---:|---:|
| A | 21.5s | 8.6s | 7.6s | **37.7s** |
| B | 34.0s | 30.1s | 21.9s | **86.0s** |
| C | 29.3s | 18.6s | 17.4s | **65.3s** |
| D-Q4 | 9.1s | 4.7s | 5.4s | **19.2s** |
| D-Q6 | 7.2s | 3.9s | 3.5s | **14.6s** |
| E | 51.7s | 31.4s | 31.9s | **115.0s** |

### ข้อควรระวังเรื่องเวลา

- D-Q4/Q6 เป็นเวลาหลัง llama.cpp, GGUF และ OS file cache อุ่นแล้ว จึงไม่ใช่ cold-start comparison ที่ยุติธรรมกับ A–C/E
- D-Q4 cold-start ก่อนเติม CUDA runtime ใช้ประมาณ 181 วินาทีต่อ 4 เฟรม
- หลังเติม CUDA runtime D-Q4 ลดเหลือประมาณ 76 วินาทีต่อ 4 เฟรมในรอบแรก
- การ backfill จริงควรโหลดโมเดลหนึ่งครั้ง ค้างไว้ตลอดวิดีโอ แล้ว unload เมื่อ variant นั้นเสร็จ ไม่ควรโหลด/unload ทุก scene
- E ใช้ video_chunk ในระดับ artifact แต่บน Windows ค่าเริ่มต้น VLM_VIDEO_USE_CONTAINER=false ทำให้ inference ใช้ sampled frame paths เพื่อป้องกันการ decode วิดีโอทั้งไฟล์ซ้ำ

## 4. ผล caption ที่ได้

### A — Qwen3-VL-2B

Chunk 1:
- Summary: ฉากชายฝั่งใกล้ Kamakura มีนักโต้คลื่นเดินตามชายหาดและรูปปั้น
- Objects: surfer, sea, waves, beach, statues, text
- Actions: walking, surfing, standing

Chunk 2:
- Summary: การเดินทางใน Kamakura มีรูปปั้น หุบไผ่ ทางเดิน และรถไฟ
- Objects: stone statues, bamboo, path, stone lantern, rock, train
- Actions: walking, station, train trip

Chunk 3:
- Summary: คนกำลังเทชาเขียวลงแก้วพลาสติกที่มีน้ำแข็ง โดยใช้เหยือกสีขาว
- Objects: cup, pitcher, ice cubes, straw, person, counter, glass, matcha whisk, laptop
- Actions: pouring, filling, placing

ประเมิน: ให้ผลสม่ำเสมอและ JSON ผ่านทุก chunk เหมาะเป็น default และ fallback

### B — Qwen3-VL-2B + VISE

Chunk 1:
- Summary: การเดินทางริมชายฝั่ง Kamakura มีคลื่น ชายหาด นักโต้คลื่น และรูปปั้น
- Objects: waves, beach, surfer, statues, text
- Actions: walking, surfing, viewing

Chunk 2:
- Summary: รูปปั้น ทางเดิน ป่าไผ่ โคมไฟ และรถไฟในฉากป่า
- Objects: statues, path, bamboo, stone lantern, rock, train, text
- Actions: showing, walking, station, train, passing

Chunk 3:
- Summary: คนกำลังเทของเหลวสีเขียวลงแก้วพลาสติกด้วยเหยือกสีขาว
- Objects: person, cup, pitcher, ice, milk, straw, counter
- Actions: pouring, filling

ประเมิน: ใช้งานได้และ JSON ผ่านทุก chunk แต่ smoke test ยังไม่แสดงคุณภาพที่ดีกว่า A อย่างชัดเจน ขณะที่ช้ากว่า A มาก จึงควรเก็บเป็น experimental variant

### C — CapRL-Qwen3VL-2B

Chunk 1:
- Summary: การเดินทางริมทะเลใน Kamakura มีนักโต้คลื่นเดินตามชายฝั่งและข้อความบนภาพ
- Objects: surfer, waves, beach, statues, text overlay
- Actions: walking, surfing, running, standing

Chunk 2:
- Summary: รูปปั้นใน Kamakura ป่าไผ่ ทางเดิน และรถไฟ พร้อมข้อความเกี่ยวกับเมืองชายทะเล
- Objects: statues, path, stone lantern, large rock, bamboo, train, text
- Actions: display, passing, walking

Chunk 3:
- Summary: คนกำลังเทมัทฉะลงแก้วพลาสติกที่มีน้ำแข็งและนม
- Objects: pitcher, cup, ice, milk, straw, person's hand, glass, matcha whisk, laptop
- Actions: pouring, filling

ประเมิน: caption มีรายละเอียดมากกว่า A ในบางฉาก แต่ใช้เวลาประมาณ 1.7 เท่าของ A เหมาะเป็น quality option หรือ backfill หลัง A มากกว่าเป็น default

### D-Q4 — CapRL-Qwen3VL-4B Q4_K_M

Chunk 1:
- Summary: ฉากชายหาดมีนักโต้คลื่น ตามด้วยข้อความ และรูปปั้นในสถานที่ชายทะเลใกล้ Tokyo
- Objects: surfer, waves, beach, stone statues, text overlay
- Actions: walking with surfboard, waving, standing, displaying text

Chunk 2:
- Summary: รูปปั้นและป่าไผ่พร้อมทางรถไฟและทางเดินเกี่ยวกับการเดินทางไป Kamakura
- Objects: stone statues, bamboo stalks, stone lantern, train track, stone path, large rock
- Actions: standing still, lining up, traveling through, lining a path

Chunk 3:
- Summary: คนกำลังเทมัทฉะลงแก้วใสที่มีน้ำแข็งและของเหลวสีขาว
- Objects ที่กู้คืนได้: white pitcher, clear plastic cup, green matcha tea, ice cubes, white liquid, person's hand, laptop, matcha whisk, glass, straw
- Actions: ไม่ครบ เพราะ output ถูกตัดกลาง JSON

ประเมิน: รายละเอียดดีและเร็วเมื่อ warm แต่ Q4 มีปัญหา output ถูกตัดใน chunk ที่มีวัตถุจำนวนมาก ระบบจึงใช้ partial-caption recovery เพื่อไม่ให้ artifact หาย แต่ json_valid ของ chunk นี้เป็น false ควรใช้เป็น research option หรือเพิ่มข้อจำกัดจำนวนรายการใน prompt

### D-Q6 — CapRL-Qwen3VL-4B Q6_K

Chunk 1:
- Summary: นักโต้คลื่นเดินบนชายหาดใกล้ Kamakura พร้อมข้อความและรูปปั้น
- Objects: surfer, ocean, sand, stone statues, text overlay
- Actions: walking, carrying surfboard, waving, displaying text

Chunk 2:
- Summary: รูปปั้นและป่าไผ่พร้อมทางรถไฟในฉากการเดินทางใกล้ Tokyo
- Objects: stone statues, bamboo trees, stone lantern, train track, stone path, rock
- Actions: standing still, train moving, people riding train

Chunk 3:
- Summary: มัทฉะถูกเทลงแก้วพลาสติกที่มีน้ำแข็งและนม โดยมีมืออยู่ใกล้แก้ว
- Objects: pitcher, cup, straw, hand, matcha whisk, glass, laptop
- Actions: pouring, resting hand

ประเมิน: JSON ผ่านครบทุก chunk รายละเอียดดี และเร็วที่สุดใน warm smoke test เหมาะที่สุดสำหรับ caption ล่วงหน้าบนเครื่องที่มี VRAM เพียงพอ

### E — CapRL-Video-4B

Chunk 1:
- Summary: ฉากชายหาดมีนักโต้คลื่น ตามด้วยข้อความ Kamakura และรูปปั้น
- Objects: ocean, sand, surfer, stone statues, text overlay
- Actions: surfer running on beach, waves rolling in, statues displayed, text appears

Chunk 2:
- Summary: รูปปั้นและป่าไผ่ใน Kamakura พร้อมข้อความเกี่ยวกับการเดินทางโดยรถไฟ
- Objects: stone statues, bamboo trees, stone lantern, gravel path, rock, train
- Actions: standing in rows, lined up, traveling on track, pathway through bamboo

Chunk 3:
- Summary: คนกำลังเทมัทฉะลงแก้วที่มีน้ำแข็งและนม
- Objects: pitcher, plastic cup, straw, hand, matcha, ice cubes, counter, whisk, glass
- Actions: pouring matcha, resting hand, containing drink

ประเมิน: เข้าใจการเปลี่ยนฉากและลำดับเหตุการณ์ได้ดี แต่ช้าที่สุดประมาณ 115 วินาทีต่อ 10 เฟรม จึงเหมาะกับ deep/background caption มากกว่าการสร้าง artifact หลักที่ต้องเสร็จเร็ว

## 5. สรุปข้อดีข้อเสีย

| รุ่น | คุณภาพ caption | ความเร็ว warm | ความเสถียร JSON | VRAM/ความเสี่ยง | เหมาะกับ |
|---|---|---:|---|---|---|
| A | ดีและสม่ำเสมอ | ดี | สูง | ต่ำสุดในกลุ่ม Transformers | default/fallback |
| B | ใกล้ A จาก smoke นี้ | ช้า | สูง | เพิ่ม dependency/LoRA | adapter research |
| C | ละเอียดกว่า A บางฉาก | ปานกลาง | สูง | สูงกว่า 2B | quality backfill |
| D-Q4 | รายละเอียดดี | เร็วเมื่อ warm | มี truncation บางครั้ง | GPU หนัก | research |
| D-Q6 | รายละเอียดดีและสม่ำเสมอ | **เร็วสุดเมื่อ warm** | สูง | GPU หนักสุดในกลุ่ม GGUF | primary offline caption |
| E | เข้าใจ temporal sequence ดี | ช้าที่สุด | สูง | โหลดหนัก/ใช้เวลานาน | optional deep caption |

## 6. คำแนะนำสำหรับระบบจริงของโปรเจกต์

### Profile ที่แนะนำ

    Upload
      ├─ Phase 1: SigLIP2 indexing    → Fast search ใช้ได้ทันที
      └─ Background: D-Q6 captioning  → scene/chunk artifacts
                                    → unload D-Q6 เมื่อวิดีโอเสร็จ

ลำดับ fallback:

1. D-Q6 เป็น captioner หลัก ถ้า VRAM และ llama.cpp พร้อม
2. A เป็น fallback ที่ปลอดภัยที่สุดเมื่อ D-Q6 load ไม่ได้หรือ timeout
3. C เป็นตัวเลือก quality backfill ถ้าต้องการ caption ที่ละเอียดกว่า A
4. E เปิดเป็น deep temporal caption เฉพาะวิดีโอที่ต้องการเข้าใจลำดับเหตุการณ์
5. B เก็บไว้สำหรับการทดลอง VISE จนกว่าจะมี benchmark ที่พิสูจน์ว่าดีกว่า A

ไม่ควรสร้าง caption A–E ครบทุกตัวอัตโนมัติใน upload เดียว เพราะจะเพิ่มเวลาและใช้ GPU โดยไม่เพิ่มประโยชน์ต่อการค้นหาในทันที ควรสร้าง artifact หลักหนึ่งรุ่นก่อน แล้วค่อยสั่ง backfill รุ่นอื่นเป็นงานวิจัยภายหลัง

### กติกา lifecycle

- โหลด VLM หนึ่งรุ่นค้างไว้ตลอดการประมวลผลของวิดีโอหนึ่งไฟล์
- ไม่ให้ SigLIP2, SAM และ VLM หนักอยู่ใน VRAM พร้อมกัน
- บันทึก model ID, revision, quantization, input mode และ prompt version ใน artifact
- เมื่อโมเดลล้มเหลว ให้เก็บ Fast/SigLIP2 result และ warning แทนการทำ upload ล้ม
- ไม่ใช้ caption รุ่นหนึ่งปนกับ calibration หรือ cache ของอีกรุ่น

## 7. ข้อจำกัดของผลนี้

- เป็น smoke test จากวิดีโอเดียวและ 10 เฟรม ไม่ใช่ผล AIRC held-out
- ยังไม่มี ground-truth เพื่อสรุปว่าโมเดลใดแม่นที่สุดเชิง retrieval
- เวลา D-Q4/Q6 เป็น warm/cache-heavy จึงควรวัด cold/warm แยกในการ benchmark จริง
- ผล E รอบนี้ใช้ sampled frame paths บน Windows ไม่ใช่ full native video tensor path
- D-Q4 มี partial JSON หนึ่ง chunk แม้ระบบจะกู้ structured fields ได้
- ต้องทำ full-video backfill และ benchmark ก่อนเปลี่ยน default อย่างถาวร

## 8. แหล่งอ้างอิง runtime

- [VISE](https://github.com/mbzuai-oryx/VISE)
- [CapRL](https://github.com/InternLM/CapRL)
- [CapRL-Qwen3VL-4B-GGUF](https://huggingface.co/internlm/CapRL-Qwen3VL-4B-GGUF)
- [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
- [llama.cpp multimodal interface](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)

