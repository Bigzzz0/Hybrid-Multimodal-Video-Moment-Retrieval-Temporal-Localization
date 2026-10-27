# EmbeddingGemma 2: ค้นหาและคัดกรองเหตุการณ์จากวิดีโอที่บันทึกไว้

บร้านช์ `codex/embeddinggemma2-event-retrieval` แยกจาก `main` ที่ commit `1c734de` อยู่ใน managed worktree อีก checkout หนึ่ง งานเดิมที่ยังไม่ commit และบริการเดิมไม่ได้ถูกย้ายหรือเขียนทับ ไม่ merge เข้า main อัตโนมัติ

## เริ่มเดโม

จาก worktree นี้:

```powershell
./scripts/start-gemma2-events.ps1 -SourceCheckout 'C:\Users\User\Downloads\Video Event Retrieval'
```

เปิด `http://127.0.0.1:3002/events` เลือกคลิป → สร้างดัชนี → เลือกพฤติกรรม → เริ่มคัดกรอง → ตรวจคลิปก่อน/หลังเหตุและกดยืนยันหรือปฏิเสธ ผู้ใช้ค้นข้อความไทยหรืออังกฤษได้ใน Fast/Accurate และเปิด Caption ฉบับเต็มได้

พอร์ต localhost ของบร้านช์: frontend 3002, API 8002, GPU broker 8012, Qwen ส่วนตัว 8013 การเริ่มบริการไม่สร้างดัชนีหรือโหลดน้ำหนักโดยอัตโนมัติ สคริปต์ใช้ environment และน้ำหนักที่มีอยู่ใน SourceCheckout และตรวจรุ่นก่อนเริ่ม ไม่ติดตั้งหรือ upgrade production environment

environment Gemma แยก overrides ของ Transformers/SentenceTransformers แต่ปัจจุบันอาศัย CUDA PyTorch และ dependency พื้นฐานจาก environment เดิมผ่าน `.pth` แบบอ่านอย่างเดียว จึงตรวจ exact package pins ทุกครั้ง ไม่ใช่ environment ที่แยกไฟล์ dependency ทั้งหมด หากย้ายเครื่องให้สร้าง environment จาก pins และ Torch CUDA wheel ที่ตรงรุ่นก่อนเริ่ม

ตัวตรวจ runtime: `evaluation/embeddinggemma2/runtime-lock.json` น้ำหนัก Gemma revision `914f7f89142e33e77833254d9c9b90c3cef7303b`; SigLIP2 revision `b53b807d3a2d5e2b3911292f2d69e5341cdc064c` ห้ามโหลด checkpoint คนละ revision แล้วใช้ดัชนีเดิม

## พฤติกรรมที่ทำแล้ว

- CUDA ใช้ BF16 สำหรับ Gemma; CPU ใช้ FP32 ไม่มี FP16 สำหรับ Gemma เริ่มที่ 768 มิติ และรองรับ 512/256 ด้วยการ truncate แล้ว normalize
- คำค้นใช้ `SearchQuery`, Caption ใช้ `Document`, media ใช้ prefix ว่าง ตรวจจำนวนเฟรม เสียง และ context ก่อน forward; ข้อมูลเกินงบไม่ถูกตัดเงียบ
- ช่วง 6 วินาที เลื่อน 3 วินาที รวมช่วงท้ายที่สั้น ภาพ 1 FPS และวิดีโอ 2 FPS สูงสุด 12 เฟรม; vision budget วิดีโอ 140 ภาพ 280
- เสียง mono PCM16 16 kHz ตรงช่วงภาพ แยก `available`, `silent`, `no_audio`; ไม่มีเสียงยังค้นภาพได้
- RRF k=60 รวม image/video/audio/caption พร้อมคะแนนรายช่องทาง Fast ใช้ vectors ที่เก็บไว้และ encode เฉพาะคำค้นใหม่ Accurate ตรวจ joint AV สำหรับ top 10 และ cache ภายในงบ 60 วินาที หากไม่ครบแสดงสถานะและคืนอันดับ Fast ที่มี
- เกณฑ์พฤติกรรมใช้ prompts ไทย/อังกฤษและพฤติกรรมปกติเปรียบเทียบ ตะโกนต้องมีหลักฐานเสียง ล้มและผลัก/ชกใช้ภาพหรือวิดีโอได้ ไม่บังคับทุกช่องทางเห็นตรงกัน
- รวมเหตุประเภทเดียวกันที่ทับหรือห่าง ≤3 วินาที ปรับขอบเขตด้วยช่วง 3 วินาทีเลื่อน 1.5 วินาที เก็บช่วงคัดกรองเดิมไว้สำหรับวัด recall ก่อน refinement/Qwen
- บันทึก candidates ก่อน refine/verify งานล้มเหลวหรือ Qwen ตอบไม่ครบเป็น `unverified` งานค้างหลัง restart เป็น `interrupted` และผู้ใช้เลือกทำต่อได้
- Qwen ตรวจสูงสุด 8 เหตุต่อ batch ใช้เฟรมเรียงเวลาสูงสุด 8 เฟรมทั่วเหตุ แสดงเฟรมที่ใช้จริงและข้อความเต็ม ตรวจ JSON ดิบซ้ำ เพราะ legacy worker อาจรายงาน valid ทั้งที่ fallback จากข้อความถูกตัด
- broker ปล่อย Gemma ก่อน Qwen และปล่อย Qwen ก่อน Gemma ไม่มีการใช้ Caption ภาพมายืนยันเสียงตะโกนหรือความหมายคำพูด เก็บผลตรวจโมเดลแยกจากการยืนยันของผู้ใช้
- ดัชนีใหม่แยกตาม revision/modality/dimension/sampling/video SHA/Caption SHA ใช้ SQLite vectors สำหรับต้นแบบ มี transaction ต่อช่วง เปิด active เฉพาะเมื่อครบ Resume ใช้ช่วงและ vectors เดิมที่ทำสำเร็จแล้ว
- ดัชนี SigLIP2 และ Gemma ไม่ปะปน หากรุ่นไม่ตรงคืน `409 reindex_required` การเลือก SigLIP2 ใน API ต้องระบุชัด ไม่เป็น fallback

SQLite ใช้การสแกน vectors ทั้ง namespace จึงต้องวัดการขยายตัวเมื่อมีวิดีโอจำนวนมาก ก่อนเปลี่ยนไปใช้ ANN index นี่เป็นข้อจำกัดของต้นแบบ ไม่ได้อ้างว่าเวลา Fast จะคงที่กับฐานข้อมูลทุกขนาด

## API

ทั้งหมดอยู่ใต้ `/api/v1`:

| API | หน้าที่ |
|---|---|
| POST `/events/videos/register` | ลงทะเบียนไฟล์ที่บันทึกไว้และ Caption เดิม; ชื่อแสดงผลเลือกได้ |
| POST `/events/videos/upload` | อัปโหลดไฟล์วิดีโอ |
| GET `/events/videos` | รายชื่อและความพร้อมดัชนี |
| POST `/events/jobs` | เริ่ม index/screen หรือ `resume_id` ของงานเดิม |
| GET `/events/jobs/{id}` | progress และ telemetry |
| POST `/events/jobs/{id}/cancel` | ยกเลิกที่ขอบเขตหน่วยงาน; ไม่ทำลายผลที่ทำสำเร็จ |
| GET `/events/list/{video_id}` | เหตุใน namespace ปัจจุบัน; `include_history` ดูผลเก่า |
| POST `/events/{id}/review` | confirmed/rejected/pending พร้อมหมายเหตุ |
| GET `/events/{id}/frames/{slot}` | ภาพที่ Qwen ใช้ตรวจจริง |
| GET `/events/{id}/audio` | เสียงตรงเหตุและบริบท ±5 วินาที |
| GET `/events/videos/{id}/stream` | วิดีโอพร้อม HTTP Range |
| POST `/search/moment` | `backend: gemma2`, `profile: fast/accurate` |
| GET `/events/health` | สถานะบริการและทรัพยากร |

broker รองรับ `/embed` text/image/video/audio/joint, `/health`, `/unload`, `/verify-batch` ส่วน private Qwen เพิ่ม unload ทุกโมเดลผ่านคิว inference เดียวกัน ไม่ unload ขณะ forward

## การย้ายงาน Caption เดิม

ตรวจและย้ายเพียง `backend/app/retrieval/search_engine.py`, `frontend/src/components/search/MomentCards.tsx`, `backend/tests/test_caption_evidence.py` จากงานที่ยังไม่ commit ใน checkout เดิม เพื่อรักษา Caption evidence/fallback ที่เกี่ยวข้อง ไม่ย้าย output/environment/น้ำหนักโมเดล

นำ encoder, input verification, benchmark และ UI ทดลองเดิมจาก `codex/embeddinggemma2-experiment` มาต่อยอด ไฟล์ README และผลเก่าใต้ `evaluation/embeddinggemma2` เป็นประวัติการทดลอง ไม่ใช่ผลรับรองบร้านช์นี้ Runtime ใหม่คือ `runtime_encoder.py` และ `worker.py` บน 8012

`scripts/import-gemma2-demo.py` อ่านข้อมูลวิดีโอและ Caption JSON จริงจากฐานเดิมแบบ read-only ไม่คัดลอก SigLIP vectors มาเป็น Gemma vectors Caption Kamakura ที่นำเข้า 176 รายการคงข้อความเดิมทั้งหมด ไม่สร้างคำตอบ benchmark จาก Caption ของโมเดล

## ข้อมูลทดลองที่หาให้

คำสั่งดาวน์โหลดเก็บไฟล์ใต้ `scratch/event-retrieval/datasets` ซึ่งไม่เข้า Git:

```powershell
python -m pip install -r evaluation/event_retrieval/requirements-data.txt
python -m evaluation.event_retrieval.acquire_public --output scratch/event-retrieval/datasets --ur-preview
python -m evaluation.event_retrieval.acquire_audioset --output scratch/event-retrieval/datasets/audioset --limit 30
```

ใช้ Python ใน Gemma environment สำหรับคำสั่งด้านบนเท่านั้น

| ชุดข้อมูล | จำนวนที่ดาวน์โหลด | การใช้/ข้อจำกัด |
|---|---:|---|
| UT-Interaction | 120 action segments; 40 ผลัก/ชก, 60 handshake/hug/point สำหรับ hard negatives, 20 kick | ไม่มีเสียง; เก็บ AVI เดิมและ MP4 เต็มเฟรมสำหรับเว็บ แยก split ตาม 20 original recordings; actor/background อาจซ้ำ ต้องระบุข้อจำกัด |
| UR Fall Detection | 30 ล้ม + 30 กิจวัตรปกติ | รอบนี้ใช้ RGB panel 320×240 ของ preview ทางการ ป้ายกำกับช่วง falling จาก frame label 0 รอตรวจ alignment; ไม่ใช้เปรียบเทียบความละเอียดต้นฉบับ |
| AudioSet Shout | 30 AV segments ที่อ่านได้จริง | เสียงมากับวิดีโอต้นทาง ป้ายกำกับ weak labels; ยังไม่เท่ากับ 30 เหตุยืนยันแล้ว ต้องฟัง/ดูและทำ onset/offset ก่อนวัด accuracy |
| FSD50K | Shout 30 + Speech 30 previews | audio-only diagnostics; ไม่จับคู่ภาพคนละเหตุและไม่นับเป็น AV ground truth |

หากต้องการ UR RGB ต้นฉบับ ให้รัน acquire_public โดยเอา `--ur-preview` ออก ตัวดาวน์โหลดรองรับ HTTP ranges และไฟล์ `.part` เพื่อ resume; เวลาจริงจาก synchronization CSV ไม่สมมติทุกเฟรมเป็น 30 FPS

แหล่งข้อมูลและเงื่อนไข:

- [UT-Interaction](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html): อ้างอิง Ryoo และ Aggarwal (2010), ไม่มีการสมมติว่าเป็น CC license
- [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html): CC BY-NC-SA 4.0 ใช้เชิงวิชาการไม่ใช่เชิงพาณิชย์ อ้างอิง Kwolek และ Kepski (2014)
- [AudioSet](https://research.google.com/audioset/download.html): metadata ไม่ใช่ license สำหรับเผยแพร่คลิปต่อ ป้ายกำกับ Shout ต้องตรวจซ้ำ
- [FSD50K](https://zenodo.org/records/4060432), [official browser](https://fsannotator.upf.edu/fsd/release/FSD50K/): audio previews คงลิงก์ Freesound และต้องเก็บ license/attribution รายคลิปก่อนแจกจ่าย

`manifest.combined.json` เป็น catalog ที่มี path/SHA/provenance/split ไม่ใช่ชุด acceptance ที่สมบูรณ์ ยังขาด shouting positives/hard negatives ที่ตรวจแล้วและวิดีโอปกติต่อเนื่อง 90 นาที (validation 30/test 60) `dataset.validate` ปฏิเสธการเปิด acceptance หากจำนวนไม่ครบหรือ annotation ยัง pending

## วัดผลและเลือกค่า

```powershell
python -m evaluation.event_retrieval.dataset scratch/event-retrieval/datasets/manifest.combined.json
python -m evaluation.event_retrieval.benchmark scratch/event-retrieval/regression-manifest.json --output scratch/event-retrieval/ablations
python -m evaluation.event_retrieval.evaluate_events MANIFEST --split validation --output scratch/event-retrieval/validation
python -m evaluation.event_retrieval.select_validation REPORTS_JSON --lock backend/data/gemma2/validation-lock.json
python -m evaluation.event_retrieval.evaluate_events MANIFEST --split test --validation-lock backend/data/gemma2/validation-lock.json --output scratch/event-retrieval/test
python -m evaluation.event_retrieval.qwen_comparison ut-punch-demo --limit 2 --output scratch/event-retrieval/qwen-comparison.json
```

รัน benchmark/comparison เมื่อคิวเดโมว่าง ทุก arm ผ่าน GPU broker; ไม่เปิด worker GPU เพิ่ม คำสั่ง select_validation ปฏิเสธ test reports และไม่เขียนทับ lock เดิม

benchmark แยก SigLIP2 image, Gemma image/video/audio/caption/video+audio/all จัด index ใหม่ตาม configuration วัด Thai/English Recall@1/@5, temporal IoU, no-match FP, warm query p50/p95, index/decode/encode และ telemetry Audio หรือ Caption ที่ไม่มีคงเป็น missing ไม่เติมผลปลอม

`grid --phase vision` เปรียบเทียบ budgets 140/280/560/1120 โดยทุก arm ใช้เฟรมเดียวกัน 6 เฟรมที่ 1 FPS เพราะ 12×1120 เกิน context หากจะใช้ 12 เฟรมให้เปรียบเทียบ 140/280/560 แยกเป็นอีกชุด `grid --phase dimensions --validation-lock ...` ทดลอง 768/512/256 หลังเลือก configuration ที่ผ่านความแม่นจาก validation เท่านั้น

`qwen_comparison` เรียกทั้ง all-window และ screened-window จริง โดย prompt และเฟรมต่อหน่วยเหมือนกัน รายงานเวลาทั้งสอง arm ไม่คาดเดาเวลาจากจำนวน calls การเปรียบเทียบนี้ยังต้องมี ground truth อิสระเพื่อวัดความแม่น

event evaluator จับคู่ one-to-one ที่ tIoU≥0.3 เหตุซ้ำเป็น FP รายงาน Wilson 95% CI และ false alerts/hour เฉพาะเมื่อมีชั่วโมง normal ที่ระบุจริง หากไม่มีข้อมูลให้ `None` ไม่ใช่ศูนย์ เลือกค่าเฉพาะ validation ที่ผ่าน recall ก่อน Qwen ≥95%, final recall ≥85%, precision ≥80%, FA/hour≤2, allocated≤6 GiB, warm Fast p95≤2s และ call reduction≥50%; จากนั้นเลือกเวลาต่ำสุดและ VRAM ต่ำกว่าหากเท่ากัน

threshold เริ่ม .35 และ margin .02 เป็นค่าทดลอง ยังไม่ได้ calibrate ไม่อ้าง similarity หรือ confidence จาก Qwen ว่าเป็น probability ระบบ retrieval ปัจจุบันยังคืน top-K แม้คำค้นไม่มีคำตอบ จึงต้องรายงาน no-match FP และทำ validation-based rejection ก่อนอ้างว่าสามารถแยก no-match ได้

## สถานะและขอบเขต

การทำงานของ demo และการทดสอบซอฟต์แวร์แยกจากการผ่านเกณฑ์ accuracy ไม่รับกล้องสด ไม่แจ้งเตือนภายนอก ไม่ถอดคำพูดไทย ไม่ fine-tune และไม่สรุปเจตนา/ความมึนเมา/ความอันตรายจากรูปลักษณ์

เก็บ index/database/คลิป/รายงาน raw/screenshot/environment ไว้ใต้ data หรือ scratch ที่ ignore ไว้ รายงานผลจริงอยู่ใน `docs/embeddinggemma2-results.md` และ artifact ใน scratch ไม่มีการ merge/main หรือเผยแพร่ข้อมูลโดยอัตโนมัติ

คู่มือโมเดล: [Google model card](https://huggingface.co/google/embeddinggemma-2), [Google multimodal guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
