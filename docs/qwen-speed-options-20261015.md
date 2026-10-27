# เพิ่มความเร็วโดยรักษาหลักฐานเดิม — 15 ตุลาคม 2026

## วัดอะไรจริง

ใช้คำขอ uniform sampling เดิมจาก adaptive-frames-20261015-v2 ไม่เปลี่ยนโมเดล Q6, prompt, ภาพ, timestamps, normal references หรือ max_tokens ทดสอบ cold private-worker request หนึ่งครั้ง จากนั้นเรียก llama-server ที่โหลดอยู่ด้วย payload ซึ่งสร้างจากเมธอด prompt/frame labeling เดิมอีกสามครั้ง แรกสุดเป็นคำขอเหมือน cold request ทุกอย่าง อีกสองครั้งเป็นช่วงใหม่ ไม่ใช่ benchmark ความแม่นหรือ throughput ที่เป็นตัวแทนข้อมูลทั้งหมด

| การวัด | เวลา / จำนวน |
|---|---:|
| Cold request ผ่าน private worker รวมโหลด | 9.65s |
| โหลด Qwen จาก health telemetry | 2.83s |
| Inference ของ cold request ไม่รวมโหลด จาก health | 6.57s |
| ปล่อยโมเดลผ่าน private worker | 0.84s |
| Warm request ช่วงใหม่ A: wall / prompt / generation | 5.81s / 3.25s / 1.46s |
| Warm request ช่วงใหม่ B: wall / prompt / generation | 6.22s / 3.98s / 1.23s |
| Prompt tokens ช่วงใหม่ A / B | 9,701 / 11,587 |
| Output tokens ช่วงใหม่ A / B | 148 / 120 |
| อ่าน JPEG + base64 ใน profiler | 16–29ms ต่อคำขอ |

เวลา prompt_ms เป็นค่าที่ server รายงาน ไม่ใช่เวลารวม vision pipeline ทั้งหมด ส่วนต่างระหว่าง wall กับ prompt+generation อาจมี image encoding, HTTP และงาน server อื่น ไม่แยกอ้างว่าเป็นค่าใดค่าหนึ่งโดยยังไม่ instrument เพิ่ม

คำขอเหมือนเดิมบนโมเดลที่ยังโหลดอยู่ใช้ wall 1.95s, cache_n=11,581 และ prompt_n=1 เทียบกับสองช่วงใหม่ที่ cache_n=0 ผล assessment ของ cold กับ identical repeat เป็น normal ตรงกัน ค่านี้แสดงว่า prompt cache ทำงานอยู่แล้ว **ไม่ใช่ speedup ของวิดีโอใหม่** และไม่ได้พิสูจน์คุณภาพทุกตัวอย่าง

ผลดิบ: `scratch/event-retrieval/qwen-speed-profile-20261015-v1/results.json` โมเดลถูกปล่อยเรียบร้อยหลังวัด ไม่มีการแก้ production defaults หรืองาน review

## ลำดับที่แนะนำ

### 1. ลด lifecycle ที่ซ้ำ โดยคง batch limit เดิม

ปัจจุบัน broker `evaluation/embeddinggemma2/worker.py:verify_batch` ทำ release_qwen ใน finally ของทุก batch และ `backend/app/gemma2/open_screening.py:screen_open` ส่ง batch ละสูงสุด 8 รวมถึงการ retry แบบชุดเล็ก จึงมีการโหลดใหม่เมื่อมี batch ถัดไป

ควรเพิ่ม job-scoped verification lease ใน broker: encode/score Gemma ให้เสร็จก่อน ปล่อย Gemma แล้วโหลด Qwen เพียงครั้งเดียวสำหรับช่วง verify ของงาน คง batch limit และการตอบ progress/cancel เดิม ปล่อย Qwen เมื่อจบงาน ยกเลิก ผิดพลาด หมด lease หรือก่อนงาน Gemma ได้ GPU ต้องให้ broker เป็นผู้ควบคุมทั้งหมด ไม่เปิด flag keep_loaded ที่ค้าง GPU ได้ไม่จำกัด

นี่ลดงานโหลดซ้ำโดยไม่ลดเฟรมหรือเปลี่ยนคำสั่ง แต่ยังต้องเทียบผลทุกหน่วย เพราะ KV/cache และ runtime อาจทำให้การคำนวณต่างเล็กน้อย ในการทดลองก่อนที่ 16 หน่วยใช้ batch 4 มีสี่ lifecycle หากลดเหลือหนึ่งจะตัด overhead ได้คร่าว ๆ 3×(2.83+.84)≈11s เทียบ Qwen wall ราว130s ตัวเลขนี้เป็นเพียงประมาณการ ไม่ใช่ speedup ที่ทดสอบแล้ว ส่วน production ใช้ batch8อยู่แล้วจึงประหยัดน้อยกว่าตัวอย่าง batch4 และคำขอ <=8 ที่ไม่มี retries แทบไม่ได้ประโยชน์ส่วนนี้

### 2. Cache ผล Qwen เมื่อ input ตรงกันจริง ข้าม job ได้

ระบบปัจจุบันข้าม event ที่ verification_complete ในการ resume ได้อยู่แล้ว แต่ event key มี job_id จึงไม่ใช่ cache inference ที่ใช้ข้ามงานใหม่โดยทั่วไป ควรเก็บผลดิบที่ตรวจ schema ผ่านแยกจาก event/review โดย key ประกอบด้วย model revision/quantization/mmproj, runtime/template generation config, verification mode, prompt/context ทั้งหมด, ภาพและ normal references ตามลำดับพร้อม content hashes/timestamps และรุ่นตัวตรวจ schema

เมื่อ reference bank/context/ภาพ/รุ่นโมเดลเปลี่ยนให้ miss cache ไม่ใช้ชื่อไฟล์หรือช่วงเวลาอย่างเดียว ไม่ถือ error/OOM/invalid JSON ว่าเป็น normal และไม่คัดลอกผลยืนยันของคนจากงานเก่า ส่วน uncertain ต้องเก็บสถานะจริงและคงนโยบาย retry เดิม หากมีคำสั่ง force rerun ต้องข้าม cache ได้พร้อมบันทึกที่มา

ช่วยมากกับ demo/การรันซ้ำที่ inputs เดิม ไม่ลดต้นทุน first pass ของวิดีโอใหม่ อัตรา hit และเวลา saved ต้องวัดจากงานจริง ไม่อ้างอัตราเร็วหลายเท่าจาก identical-repeat test

### 3. Decode และเตรียมภาพล่วงหน้าบน CPU

อ่านวิดีโอต่อเนื่อง แทน seek ใหม่ทุกเฟรม แล้วเตรียม batch ถัดไประหว่าง Qwen ทำ batch ปัจจุบันได้โดยไม่ให้ Gemma แย่ง GPU จำกัดจำนวน batch ในคิวและตรวจ source-frame indices/pixels ให้ตรงเดิม เก็บ timestamps, JPEG encoding และลำดับภาพเดิมเพื่อแยกคุณภาพออกจากการเปลี่ยน input

Base64 ที่วัดใช้เพียงหลักสิบมิลลิวินาที จึงไม่ใช่เป้าหมายหลักและไม่ควรคาดว่าจะลดเวลารวมได้มากจากจุดนี้อย่างเดียว เวลาที่ได้คืนจากการ overlap ต้องวัดใหม่ ไม่บวกตัวเลข preparation ของ sampler คนละวิธีเข้ากับ baseline โดยตรง

### 4. ปรับ runtime หลังเก็บ timing ครบ แล้ววัดคุณภาพเทียบ

llama.cpp รองรับ Flash Attention, batch/ubatch และรายงาน timings แต่ binary ที่เราใช้แสดงค่า Flash Attention เริ่มต้น auto จึงไม่ควรสรุปว่าเพียงเปิดแล้วจะเร็วขึ้น ควรบันทึกสถานะจริง ปรับทีละค่า และวัดบนคำขอที่มีทั้ง normal/suspicious/uncertain โดยคง Q6 และภาพเดิมก่อน ไม่เร่งด้วยการเพิ่ม parallel requests จน VRAM เต็ม

หากต้องลด vision tokens/ความละเอียดเพื่อให้ first pass เร็วขึ้น จะกระทบหลักฐานที่เห็น ต้องเป็นการทดลองคุณภาพต่างหาก การย้าย normal references มาไว้ต้น prompt เพื่อหวัง KV prefix reuse ก็เปลี่ยนลำดับ input จึงต้องวัดใหม่ ไม่ถือว่าเป็นการแก้ที่รักษาคุณภาพโดยอัตโนมัติ

## สิ่งที่ยังไม่ควรใช้เป็นทางลัด

- ลด max_tokens 512→256 ไม่ทำให้คำตอบที่จบเองที่120–148tokensเร็วขึ้นโดยอัตโนมัติ และอาจตัดคำตอบอื่นที่ยาวกว่านั้น
- เพิ่มความเข้ม Gemma gate เพื่อลด Qwen calls โดยไม่วัด screening recall อาจทำเหตุหาย
- ลดเฟรม/ตัดบริบท/เปลี่ยน Q6 เป็น Q4 โดยอ้างคุณภาพเท่าเดิม ยังไม่มีหลักฐานรองรับ
- เพิ่ม prompt บังคับให้ไวต่อเหตุ: รอบก่อนเพิ่ม FP1→3 ไม่มีเหตุผลใช้เป็นค่าเริ่มต้น

## เกณฑ์รับการปรับความเร็ว

เทียบ baseline กับแต่ละการเปลี่ยนแปลงบน payload ชุดเดียวกัน ล็อก model/config และเก็บ hashes; เทียบ assessment/reason/evidence/schema errors/uncertain ทุกหน่วย รวม event recall/precision ตามกติกาเดิม สุ่มลำดับรันและทำซ้ำเพื่อวัด cold/warm p50/p95, เวลา end-to-end, RAM/VRAM, จำนวน model loads, cache hit และ cancellation/release failures

ใช้ชุดที่เคยตรวจ7เหตุเป็น regression เท่านั้น เพิ่มวิดีโอที่กันไว้ใหม่ก่อนอ้างว่ายังคงคุณภาพทั่วไป ยังไม่ประกาศเปอร์เซ็นต์ speedup จนวัด pipeline ที่แก้แล้วจริง

## เอกสารหลัก

- llama.cpp server documentation (timings, prompt cache, runtime options): https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Qwen3-VL official repository (pixel budgets เป็นตัวปรับ input ซึ่งต้องประเมินคุณภาพ): https://github.com/QwenLM/Qwen3-VL
