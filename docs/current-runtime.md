# Current Production Runtime

เอกสารนี้เป็นแหล่งอ้างอิงของ implementation ปัจจุบันบน `main` โดยยึด
production implementation commit `6de9468` เป็นฐาน ไม่ใช่ผล benchmark ใหม่

## Model stack

| งาน | Backend | Model / revision | Runtime |
| --- | --- | --- | --- |
| Retrieval | SigLIP2 NaFlex | `google/siglip2-base-patch16-naflex` | Main API/worker |
| Scene caption | `caprl_qwen3vl_4b_q6` | `internlm/CapRL-Qwen3VL-4B-GGUF` / `922d08bb6257875336aa138616c74902f736099c` | llama.cpp CUDA |
| Accurate verifier | `caprl_qwen3vl_4b_q6` | CapRL Q6 ตัวเดียวกับ caption | llama.cpp CUDA |
| Object grounding | SAM 3.1 | `facebook/sam3.1` | ปิดด้วย `ENABLE_SAM_GROUNDING=false` |

ไฟล์โมเดล Q6 ต้องอยู่นอก Git:

- `CapRL-Qwen3VL-4B-q6_k.gguf`
- `CapRL-Qwen3VL-4B-mmproj-Q8_0.gguf`
- `llama-server.exe`

Qwen3-VL-2B ยังมีชื่อใน compatibility code และข้อมูลเก่าเท่านั้น ไม่ถูก
ลงทะเบียนหรือโหลดใน production caption/verifier flow ปัจจุบัน

## Process architecture

```text
Browser :3000
    │
    ▼
FastAPI :8000  ── local HTTP ──>  inference worker :8011
                                      │
                                      └─ llama-server / CapRL Q6
```

ทุก process ทำงาน local เท่านั้น worker bind ที่ `127.0.0.1` และรับ local file
paths ไม่ส่งวิดีโอหรือเฟรมไป cloud

## Search flow

### Fast

1. ใช้ SigLIP2 ค้น frame candidates
2. รวม stored scene captions ด้วย caption retrieval/RRF เมื่อ artifact พร้อม
3. สร้าง temporal proposals และจัดกลุ่ม occurrence
4. ถ้า caption ยังสร้างไม่เสร็จ ใช้ SigLIP2 อย่างเดียวพร้อม `caption_backfill_pending`

### Accurate

1. เริ่มจาก Fast candidates
2. ใช้ stored CapRL Q6 caption เป็นหลักฐานเบื้องต้น
3. โหลด CapRL Q6 verifier เมื่อ query ต้องตรวจ action, relation, count, spatial,
   temporal, ambiguity หรือ caption ยังไม่ชัด
4. ตรวจ candidate ตาม budget 60 วินาที และ unload worker model ใน `finally`
5. รวมคะแนนจาก Fast และ verifier; timeout/OOM/unavailable จะคืน Fast result พร้อม warning

SAM ไม่ถูกเรียกใน production flow เมื่อ `ENABLE_SAM_GROUNDING=false`

## Caption lifecycle

หลังอัปโหลด ระบบทำ Phase 1 ก่อนเพื่อสร้าง scene/keyframes และ SigLIP2 index
ทำให้ Fast Search ใช้งานได้ก่อน จากนั้นจึงสร้าง CapRL Q6 captions แบบ background:

1. เลือก keyframes สูงสุด 4 เฟรมต่อ scene
2. เปิด Q6 worker และเขียน artifact หลังจบแต่ละ scene
3. เก็บ metadata เป็น `pending`/`running` จน artifact ครบ
4. เปลี่ยน active caption version เมื่อทุก scene valid
5. งานหยุดกลางทางสามารถ resume ได้โดยไม่ลบ active version เดิม

คำสั่ง backfill:

```powershell
cd backend
python -m scripts.backfill_primary_captions --all-videos --resume
python -m scripts.backfill_primary_captions --video-id <ID> --resume
```

Production ไม่เปิด Qwen3-VL-2B เป็น fallback โดยอัตโนมัติ หาก Q6 ล้มเหลวทั้ง
scene ระบบเก็บ error และยังให้ค้นด้วย Fast ได้

## Configuration

ค่าหลักใน `backend/.env`:

```env
SIGLIP2_MODEL_ID=google/siglip2-base-patch16-naflex
QWEN_VL_MODEL_ID=internlm/CapRL-Qwen3VL-4B-GGUF
QWEN_VL_MODEL_REVISION=922d08bb6257875336aa138616c74902f736099c
CAPTION_PRIMARY_BACKEND=caprl_qwen3vl_4b_q6
CAPTION_FALLBACK_BACKEND=
VLM_VERIFIER_BACKEND=caprl_qwen3vl_4b_q6
CAPTION_ARTIFACT_VERSION=q6-scene-caption-v1
ENABLE_SAM_GROUNDING=false
INFERENCE_WORKER_URL=http://127.0.0.1:8011
MODEL_VRAM_BUDGET_MB=11800
ACCURATE_MAX_SECONDS=60
```

ตั้ง `LLAMA_CPP_PATH` ให้ชี้ไปยัง `llama-server.exe` และ `VLM_GGUF_DIR` ให้ชี้
ไปยังโฟลเดอร์ที่มี GGUF/mmproj ภายนอก repository ห้าม commit token, checkpoint,
cache, LanceDB หรือวิดีโอ

## Startup on Windows

เปิด 3 terminals ตามลำดับ:

```powershell
# Terminal 1: inference worker
cd "C:\Users\User\Downloads\Video Event Retrieval\backend"
python -m inference_worker.main

# Terminal 2: API
cd "C:\Users\User\Downloads\Video Event Retrieval\backend"
python main.py

# Terminal 3: frontend
cd "C:\Users\User\Downloads\Video Event Retrieval\frontend"
npm.cmd run dev
```

เปิดเว็บที่ `http://localhost:3000` และ Swagger ที่ `http://localhost:8000/docs`

## Public endpoints

- `POST /api/v1/search/moment` — Fast/Accurate moment search
- `GET /api/v1/videos/{video_id}/caption-status` — caption progress/status
- `POST /api/v1/videos/{video_id}/captions/rebuild` — resumable rebuild; ใช้ `force=true` เฉพาะเมื่อจำเป็น
- `GET /api/v1/system/telemetry` — worker/model/queue/VRAM telemetry

Search response มี model provenance, caption status, stage latency, warnings,
`cascade_path` และ `models_attempted` สำหรับตรวจสอบการทำงานจริง

## Evidence ในหน้าเว็บ

ใน Moment Card:

- กด `Caption เต็ม` เพื่อดู structured caption ทั้ง JSON ของ scene
- กด `ผล Accurate verifier` เพื่อดู raw VLM JSON, `event_present`, confidence,
  reason และ model ID
- SAM overlay/evidence จะไม่แสดงเมื่อ feature flag ปิด

## Warning และ fallback ที่พบบ่อย

| Warning | ความหมาย |
| --- | --- |
| `caption_backfill_pending` | ยังสร้าง caption ไม่ครบ แต่ Fast ใช้ได้ |
| `caprl_q6_worker_unavailable` | ติดต่อ Q6 worker ไม่ได้; คืน Fast |
| `caprl_q6_timeout_fallback` | verifier หมดเวลา; คืน Fast |
| `verifier_invalid_json_fallback` | ผล verifier parse ไม่ได้; คืน candidate เดิม |
| `accurate_partial_budget` | เหลือเวลาไม่พอสำหรับ verifier เพิ่ม |
| `calibration_missing` | ยังไม่มี calibration ที่ตรงกับ model/version |

## สิ่งที่ไม่รองรับใน production รอบนี้

- SAM 3.1 production grounding/tracking
- Identity, face recognition และ Re-ID
- OCR หรือ subtitle เป็นหลักฐานค้นหา
- audio เป็นหลักฐานค้นหา
- cloud inference หรือการส่งวิดีโอออกนอกเครื่อง
- real-time และ multi-camera

## Troubleshooting

### Worker unavailable

ตรวจว่า worker ทำงานที่ `127.0.0.1:8011`, token ใน API/worker ตรงกัน และเรียก
`/health` ก่อนค้น Accurate ระบบยังควรคืน Fast result ได้

### GGUF หรือ mmproj path ผิด

ตรวจ `LLAMA_CPP_PATH`, `VLM_GGUF_DIR` และชื่อไฟล์ Q6/mmproj ให้ตรงกับ registry
ห้ามแก้ชื่อ backend เป็น model ID อื่นที่ไม่ได้อยู่ใน allowlist

### VRAM ไม่พอ

ปิด preload ที่ไม่จำเป็น, ตรวจว่าไม่มี worker ซ้ำหลาย process และให้ model manager
unload โมเดลก่อนเริ่ม stage ถัดไป ค่า budget production คือ `11800 MB`

### Caption status ค้าง

ดู `/caption-status` และ `/system/telemetry`, ตรวจ worker log แล้วสั่ง backfill
ด้วย `--resume` งานเดิมจะไม่สร้าง artifact ซ้ำ

### PowerShell แจ้งว่า npm script ถูกปิด

ใช้ `npm.cmd run dev` แทน `npm run dev` เพื่อหลีกเลี่ยง execution policy ของ
`npm.ps1`

## License and research limitation

CapRL Q6 เป็นโมเดลสำหรับ research/demo ตามเงื่อนไขของ upstream repository ต้อง
ตรวจสอบ license และสิทธิ์การใช้งานเชิงพาณิชย์ก่อนนำไปใช้จริง
