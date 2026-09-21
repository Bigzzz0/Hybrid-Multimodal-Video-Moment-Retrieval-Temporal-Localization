# Pure-Visual Video Moment Retrieval & Temporal Localization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Next.js_14-App_Router-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/LanceDB-Serverless_Vector_DB-00D2B4?style=for-the-badge" alt="LanceDB" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <b>Natural Language Video Moment Retrieval & Temporal Boundary Localization System</b><br />
  Powered by <b>SigLIP 2 (NaFlex)</b>, <b>SAM 3.1</b>, <b>Qwen3-VL-2B (4-bit)</b>, calibrated temporal proposals, and <b>LanceDB (IVF-PQ & FTS)</b>.<br />
  <i>100% Local On-Premise Execution on an RTX 5070 12GB with Zero Cloud API Costs.</i>
</p>

---

## 🌟 จุดเด่นของระบบ (Key Highlights)

* 🔒 **100% Local On-Premise & Complete Data Privacy:** ประมวลผลและจัดเก็บข้อมูลเวกเตอร์ภายในเครื่องทั้งหมด ข้อมูลวิดีโอไม่รั่วไหลสู่คลาวด์ภายนอก และไม่มีค่าใช้จ่าย API รายเดือน
* ⚡ **Consumer GPU Optimized (RTX 5070 12GB):** ใช้ SigLIP2 เป็น retrieval หลัก และแยก Qwen3/SAM ไว้ใน local inference worker เพื่อควบคุม VRAM และป้องกัน OOM
* 🚀 **Progressive Visual Ingestion:** สกัด scene, keyframe และ frame embeddings ก่อนค้นหา พร้อมสร้าง dense visual captions แบบ background
* 📈 **Dynamic Relevance Density Heatmap:** แถบเรืองแสงแสดงระดับความเกี่ยวข้องของเนื้อหาตลอดทั้งวิดีโอแบบ 1-Hz Canvas Visualizer ช่วยให้ผู้ใช้เห็นภาพรวมของทั้งคลิปได้ในเสี้ยววินาที
* ⏱️ **Calibrated Multi-scale Temporal Localization:** สกัดช่วงเวลาเริ่มต้น-สิ้นสุด $[t_{start}, t_{end}]$ ด้วย rolling proposals, boundary/transition refinement และ Gaussian Soft-NMS ก่อนจัดลำดับหลายเหตุการณ์

---

## 🏛️ สถาปัตยกรรมระบบ (System Architecture)

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                PROGRESSIVE INGESTION PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                     [ Video File (.mp4/.mov) ]
                                                 │
                                                 ▼
                                   [ Decord GPU Decoder (NVDEC) ]
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                   ▼
     [ PySceneDetect Adaptive Cuts ]
                   │
                   ▼
      [ Visual Keyframe Sampling ]
                   │
         ┌─────────┴──────────────────────┐
         ▼                                ▼
  [ SigLIP 2 (NaFlex) ]       [ Qwen3-VL-2B ]       [ SAM 3.1 ]
   (768-dim Retrieval)       (Caption / VQA)    (Grounding / Mask / Track)
         │                                │
         └────────────────┬───────────────┘
                          ▼
    =================================================================================
       [ LanceDB Serverless Columnar Vector Database (Apache Arrow & Disk-based) ]
        • Disk-based IVF-PQ Cosine Vector Index (video_frames_v2)
        • Tantivy Full-Text Search (BM25) Index (scenes_v2 captions)
        • Per-video index_metadata: schema/model/version/dimension/timestamp
    =================================================================================
                                                 ▲
                                                 │
┌────────────────────────────────────────────────┴────────────────────────────────────────┐
│                        HYBRID RETRIEVAL & TEMPORAL BOUNDARY ENGINE                      │
└─────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                [ Natural Language Query: "ฉากที่มีคนอธิบายสไลด์กราฟแท่ง" ]
                                                 │
        ┌────────────────────────────────────────┼────────────────────────────────────────┐
        ▼                                        ▼                                        ▼
  [ Visual Vector Similarity ]         [ Scene Caption BM25 / FTS ]
        │                                        │                                        │
        └────────────────────────────────────────┼────────────────────────────────────────┘
                                                 ▼
  [ Weighted Reciprocal Rank Fusion (RRF) ]
                                                 │
                                                 ▼
                      [ 1D Gaussian Temporal Convolution: S(t) * G_σ ]
                                                 │
                                                 ▼
                   [ Dynamic Threshold Boundary Extraction: [t_start, t_end] ]
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           NEXT.JS 14 INTERACTIVE VIDEO DASHBOARD                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│  • Dynamic Relevance Density Heatmap Canvas Bar                                         │
│  • Timeline Auto-Seek Video Player (measured in benchmark)                             │
│  • Ranked Moment Cards with Timestamp Badges & Previews                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ รายการเทคโนโลยีที่เลือกใช้ (Tech Stack)

| ส่วนประกอบ | เทคโนโลยีที่เลือกใช้ | บทบาทและจุดเด่น |
| :--- | :--- | :--- |
| **Visual-Text Backbone** | `google/siglip2-base-patch16-naflex` | สกัดเวกเตอร์ภาพ 768-dim โดยรักษา aspect ratio |
| **Dense Visual Captioner / VQA**| `Qwen/Qwen3-VL-2B-Instruct` (NF4 4-bit) | caption, action/relation verification และ Video VQA |
| **Object Grounding** | `facebook/sam3.1` | text grounding, bounding box, mask และ temporal evidence |
| **Video Decoding**        | `Decord` (NVDEC GPU Hardware Fallback) | ถอดรหัสเฟรมจริงพร้อมรักษา timestamp และ fallback บน CPU |
| **Vector Storage**        | `LanceDB` (Apache Arrow Format) | Vector DB แบบ Serverless บน SSD พร้อมดัชนี IVF-PQ และ FTS |
| **Temporal Algorithm**    | `1D Gaussian Convolution & RRF` | กรองสัญญาณรบกวนและสกัดช่วงเวลาต่อเนื่อง $[t_s, t_e]$ |
| **Backend API**           | `FastAPI` + `Uvicorn` + `WebSockets` | REST API, HTTP 206 Byte-Range Streaming, Live Telemetry |
| **Frontend UI**           | `Next.js 14` + `React 18` + `Tailwind CSS` | Dashboard สไตล์ Dark Glassmorphism พร้อม Canvas Heatmap |

---

## 🚀 วิธีการติดตั้งและเริ่มใช้งาน (Quick Start Guide)

### 1. ข้อกำหนดขั้นต่ำของระบบ (System Requirements)
* **OS:** Windows 10/11, Ubuntu 22.04+ หรือ macOS (Apple Silicon)
* **GPU:** NVIDIA GPU พร้อม VRAM 12GB แนะนำ RTX 5070 หรือสูงกว่า
* **Software:** Python 3.12 สำหรับ inference worker, Python 3.11+ สำหรับ API, Node.js 18+, FFmpeg

---

### 2. ติดตั้งและตั้งค่า Backend Service (FastAPI)

```bash
# 1. เข้าสู่โฟลเดอร์ backend
cd backend

# 2. สร้าง Virtual Environment และติดตั้ง Dependencies
python -m venv venv

# บน Windows:
.\venv\Scripts\activate
# บน Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

#### การตั้งค่า `.env`:
สร้างไฟล์ `backend/.env` (หรือแก้ไขจาก `.env.example`):
```env
HF_TOKEN=hf_your_token_here

SIGLIP2_MODEL_ID=google/siglip2-base-patch16-naflex
QWEN_VL_MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
SAM_MODEL_ID=facebook/sam3.1
INFERENCE_WORKER_ENABLED=true
INFERENCE_WORKER_URL=http://127.0.0.1:8011
INFERENCE_WORKER_TOKEN=change-this-local-secret
MODEL_VRAM_BUDGET_MB=11800
ACCURATE_MAX_SECONDS=60
INFERENCE_WARMUP_QWEN=false
SAM_SEARCH_TOP_K=2
QWEN_FALLBACK_TOP_K=1
QWEN_MIN_REMAINING_SECONDS=12
SAM_COMPILE=false
```

โมเดลหนักจะทำงานใน process แยกที่ bind เฉพาะ `127.0.0.1` เพื่อให้ main API
ไม่ต้องถือ SAM/Qwen พร้อมกันเอง:

```bash
# terminal แยก: ใช้ environment ที่ติดตั้ง requirements-inference.txt
cd backend
python -m inference_worker.main
```

ถ้ายังไม่ได้ติดตั้ง worker หรือยังไม่มีสิทธิ์ดาวน์โหลด SAM checkpoint ให้ตั้ง
`INFERENCE_WORKER_ENABLED=false` ระบบยังค้น Fast ได้ และ Accurate จะ fallback ตาม
warning ที่คืนใน API แทนการทำให้เซิร์ฟเวอร์ล้ม

#### วอร์มโมเดล AI ล่วงหน้า (One-Click Preload):
```bash
 # โหลดและแคชโมเดลภาพขึ้น GPU Memory ทันที (Zero Cold-Start)
python preload_models.py
```

#### เริ่มรันเซิร์ฟเวอร์ Backend:
```bash
python main.py
```
* **Swagger API Documentation:** `http://localhost:8000/docs`

วิดีโอที่ถูกสร้างด้วยดัชนี v1 จะตอบ HTTP `409 reindex_required` จนกว่าจะ ingest ใหม่เป็น
`video_frames_v2`/`scenes_v2` โดยจะไม่ผสมเวกเตอร์คนละรุ่นระหว่างค้นหา

หากฐานข้อมูลเดิมมีตารางข้อความที่ต้องเก็บไว้ ให้สำรองก่อนลบ (คำสั่งนี้ไม่ถูกเรียกจาก runtime):

```bash
python backend/scripts/migrate_legacy_index.py --db backend/data/lancedb \
  --output backend/data/backups/legacy-text.jsonl
# เมื่อตรวจสอบไฟล์ backup แล้วจึงค่อยเพิ่ม --drop-legacy
```

---

### 3. ติดตั้งและรัน Frontend Web Application (Next.js 14)

เปิด Terminal ใหม่:
```bash
# 1. เข้าสู่โฟลเดอร์ frontend
cd frontend

# 2. ติดตั้ง Node Dependencies
npm install

# 3. รัน Next.js Dev Server
npm run dev
```
* **Web Dashboard Application:** `http://localhost:3000`

### สัญญา Search API (v2)

`POST /api/v1/search/moment` รับ `query`, `video_id` (บังคับ), `top_k` 1–20 และ
`profile` เป็น `fast` หรือ `accurate` ค่า tuning รุ่นเก่าที่ส่งมาเกินจะถูก ignore
ชั่วคราวเพื่อให้ client เดิมไม่พัง แต่ server เป็นผู้กำหนดน้ำหนักและ threshold เอง

`fast` ใช้ frame embeddings + scene-caption RRF; `accurate` ใช้ GPU cascade แบบลำดับ
SigLIP2 → unload → SAM 3.1 → unload → Qwen3-VL เมื่อ SAM ยังตอบคำค้นไม่ครบ → unload
ภายใต้งบเวลารวม 60 วินาที SAM และ Qwen จะไม่อยู่ใน VRAM พร้อมกัน หาก Qwen เหลือเวลา
น้อยกว่า 12 วินาที ระบบจะคืน partial result จาก Fast/SAM ที่เสร็จแล้ว ผลลัพธ์คืน `score` ที่ calibrated
(เมื่อมี artifact), `modality_breakdown`, `occurrence_index`, `profile`,
`calibrated`, `index_version`, `strategy_used`, `models_used`, `models_attempted`,
`cascade_path`, `stage_status`, `planner_version`, `stage_latency_ms`, `cache_hits`
และ `warnings` โดยคืนได้หลาย occurrence หรือ `moments=[]` สำหรับ no-match.
ผล Accurate ที่ใช้ SAM จะมี `grounding_evidence` และ frontend overlay จะแสดง bbox/mask
ตาม timestamp ผ่าน `GET /api/v1/grounding/track/{track_id}`

### Model / dataset attribution

* [SAM 3 repository and checkpoint instructions](https://github.com/facebookresearch/sam3)
* [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
* [SigLIP2 NaFlex](https://huggingface.co/google/siglip2-base-patch16-naflex)
* [AIRC-SMARTCLASS Part 2](https://data.mendeley.com/datasets/fw5hs57z78/1)

ตรวจ license และเงื่อนไขการเข้าถึง checkpoint ก่อนใช้งานจริง ระบบรอบนี้เก็บเฉพาะ
full-body/object evidence และไม่ทำ face recognition, Identity หรือ Re-ID

---

## 📊 ตัวชี้วัดประสิทธิภาพและผลการประเมิน (Benchmark Results)

| ตัวชี้วัด (Evaluation Metric) | Baseline snapshot (28 queries) | สถานะ |
| :--- | :---: | :---: | :---: |
| **$R@1@\text{IoU}=0.5$** | **46.43%** | ต้องวัดใหม่หลังปรับปรุง |
| **Mean IoU (mIoU)** | **0.4419** | ต้องวัดใหม่หลังปรับปรุง |
| **Mean Delta ($\Delta t_{start}$)** | **12.10 s** | ต้องวัดใหม่หลังปรับปรุง |
| **Query Latency** | **458.72 ms avg** | ต้องรายงาน p50/p95 |

---

## 🧮 สูตรทางคณิตศาสตร์หลักของระบบ (Mathematical Formulations)

### 1. การคำนวณเวกเตอร์ความคล้ายคลึง (SigLIP 2 Cosine Similarity)
$$S_{\text{vis}}(t) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_{f_t}}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_{f_t}\|_2}$$

### 2. การรวมคะแนนหลายมิติ (Reciprocal Rank Fusion - RRF)
$$\text{RRF}(d) = \sum_{m \in \{\text{visual}, \text{caption}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

### 3. การกรองสัญญาณรบกวนบนเส้นเวลา (1D Gaussian Temporal Convolution)
$$\tilde{\mathcal{S}}(t) = \mathcal{S}(t) * G_\sigma(t) = \int_{-\infty}^{\infty} \mathcal{S}(\tau) \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(t-\tau)^2}{2\sigma^2}\right) d\tau$$

### 4. การตัดแบ่งช่วงเวลาเหตุการณ์ (Dynamic Threshold Boundary Extraction)
$$[t_{\text{start}}, t_{\text{end}}] = \arg\max_{[t_s, t_e]} \int_{t_s}^{t_e} (\tilde{\mathcal{S}}(t) - \theta_{\text{dyn}}) \, dt \quad \text{where} \quad \theta_{\text{dyn}} = \mu_{\mathcal{S}} + \lambda \cdot \sigma_{\mathcal{S}}$$

---

## 📁 ผังโครงสร้างโปรเจกต์ (Project Tree)

```text
Video Event Retrieval/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── v1/
│   │   │   │   ├── upload.py            # Video Upload & Worker Ingestion
│   │   │   │   ├── search.py            # Hybrid Moment Search Endpoint
│   │   │   │   ├── video.py             # Stream, Frame Preview, Clip Cut
│   │   │   │   └── websocket.py         # Real-time WebSocket Telemetry
│   │   │   └── router.py
│   │   ├── core/                        # Config, Logger, Device Maps
│   │   ├── db/                          # LanceDB Schemas & Tables Init
│   │   ├── pipeline/                    # Decord, SceneDetect, sampling, SigLIP 2, Qwen3
│   │   ├── retrieval/                   # RRF, router, SAM cache, temporal localization
│   │   ├── inference/                   # Main API client/contracts for local worker
│   │   └── ../inference_worker/         # Qwen3/SAM GPU process on 127.0.0.1:8011
│   │   └── utils/                       # HTTP 206 Byte-Range Video Streaming
│   ├── tests/                           # Pytest Test Suite
│   ├── preload_models.py                # Pre-warmer & Cache Script
│   ├── requirements.txt
│   └── main.py                          # FastAPI Server Entrypoint
│
├── frontend/
│   ├── src/
│   │   ├── app/                         # Next.js 14 App Router
│   │   ├── components/
│   │   │   ├── player/                  # VideoPlayer, TimelineHeatmap Canvas
│   │   │   ├── search/                  # MomentCards, SearchBar
│   │   │   └── upload/                  # Live Pipeline Tracker Dropzone
│   │   └── lib/                         # Axios Client & TypeScript Types
│   ├── package.json
│   └── tailwind.config.js
│
├── evaluation/
│   ├── build_airc_manifest.py         # deterministic 27/9/9 video split
│   ├── airc_annotations.schema.json   # temporal/VQA/mask annotation contract
│   ├── compute_metrics.py               # R@K@IoU, tF1, count/no-match metrics
│   ├── validate_dataset.py              # Held-out split/language/query contract
│   ├── calibrate.py                     # Platt calibration + no-match threshold
│   ├── tune_fusion.py                   # Dev-set RRF grid search
│   ├── compare_benchmarks.py            # Paired bootstrap CI + ablation comparison
│   └── run_real_video_benchmark.py      # Fast/Accurate real-video runner
│
├── Proposal.md                          # ข้อเสนอโครงงาน pure-visual retrieval
├── README.md                            # คู่มือและเอกสารประกอบโครงงานฉบับสมบูรณ์
└── .gitignore
```

---

## 🧪 การรันชุดทดสอบ (Unit Testing & Benchmarking)

```bash
# รัน Unit Tests สำหรับอัลกอริทึม RRF, Gaussian Convolution และ IoU
cd backend
pytest tests/

# ตรวจสัญญา held-out set (ต้อง >=100 queries, >=10 วิดีโอ, video-disjoint splits)
python evaluation/validate_dataset.py --dataset evaluation/datasets/heldout.json

# สร้าง calibration/fusion artifacts จาก dev split เท่านั้น
python evaluation/calibrate.py --input dev_predictions.jsonl --output backend/data/calibration_v2.json --model-id google/siglip2-base-patch16-naflex
python evaluation/tune_fusion.py --input dev_fusion.jsonl --output backend/data/fusion_v2.json

# รัน benchmark แยก Fast/Accurate และเปรียบเทียบด้วย paired bootstrap 95% CI
python evaluation/run_real_video_benchmark.py --acceptance --profile fast --dataset evaluation/datasets/heldout.json --output fast.json
python evaluation/run_real_video_benchmark.py --acceptance --profile accurate --dataset evaluation/datasets/heldout.json --output accurate.json
python evaluation/compare_benchmarks.py --baseline baseline.json --candidate accurate.json --acceptance --output comparison.json
# เพิ่ม --ablation name=report.json ซ้ำได้เพื่อเทียบ NaFlex/RRF/proposal/calibration/Qwen

# สร้าง manifest AIRC-SMARTCLASS แบบ video-disjoint 27/9/9
python evaluation/build_airc_manifest.py --source-dir path/to/AIRC-SMARTCLASS --output evaluation/datasets/airc_manifest.json
```

ไฟล์ `evaluation/datasets/real_video_benchmark.json` และค่าที่อยู่ใน
`evaluation/baseline_manifest.json` เป็น smoke/regression fixture เท่านั้น ไม่ใช่
ผล held-out หรือหลักฐาน SOTA ต้องใช้ชุด video-disjoint เดียวกันสำหรับ acceptance gate

---

### VLM A–E ablation branch

Branch `codex/vlm-model-ablation` เป็นงานวิจัย/เดโมแบบ local-only สำหรับเปรียบเทียบ
Qwen3-VL-2B, VISE, CapRL-Qwen3VL-2B, CapRL GGUF Q4/Q6 และ CapRL-Video-4B
โดยคง SigLIP2 vectors เดิมและปิด SAM ใน search flow ด้วย
`VLM_LAB_DISABLE_SAM=true` (โค้ดและตาราง SAM ยังเก็บไว้ ไม่ได้ลบ)

ติดตั้ง worker ใน environment แยก แล้วเปิดตามลำดับ:

```powershell
# terminal 1: API
cd backend
..\.venv-inference\Scripts\python.exe main.py

# terminal 2: local inference worker
cd backend
..\.venv-inference\Scripts\python.exe -m inference_worker.main

# สร้าง caption artifacts ต่อจากจุดที่ทำเสร็จแล้ว
..\.venv-inference\Scripts\python.exe -m scripts.backfill_vlm_ablation --all-videos --all-backends --resume
```

สำหรับการทดลอง B–E ให้สร้าง environment แยกจาก worker เดิมด้วย
`backend/requirements-vlm-lab.txt`; ไม่ควรติดตั้งทับ `.venv-inference`.

การค้นหา Accurate ใช้ `SigLIP2 → selected VLM`; Fast ใช้ SigLIP2 และ caption
ที่มีอยู่ทันที หาก artifact รุ่นที่เลือกยังไม่พร้อม UI จะแสดงสถานะและไม่หลอกว่า
ใช้รุ่นนั้นจริง. GGUF Q4/Q6 ต้องติดตั้ง llama.cpp v0.4.1 commit `b29c606`,
ตั้ง `LLAMA_CPP_PATH` และ `VLM_GGUF_DIR` ให้ชี้ไปยังไฟล์นอก Git.

คำสั่ง benchmark แบบ smoke (fixture ปัจจุบันไม่ใช่ held-out):

```powershell
python evaluation/run_vlm_ablation.py --backend qwen3_vl_2b
python evaluation/run_vlm_ablation.py --backend qwen3_vl_2b_vise
python evaluation/run_vlm_ablation.py --backend caprl_qwen3vl_2b
python evaluation/run_vlm_ablation.py --backend caprl_qwen3vl_4b_q4
python evaluation/run_vlm_ablation.py --backend caprl_qwen3vl_4b_q6
python evaluation/run_vlm_ablation.py --backend caprl_video_4b
```

แหล่งอ้างอิง runtime: [VISE](https://github.com/mbzuai-oryx/VISE),
[CapRL](https://github.com/InternLM/CapRL), [CapRL models](https://huggingface.co/internlm),
และ [llama.cpp multimodal](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md).
CapRL variants ให้ใช้ในขอบเขต research/demo ตามเงื่อนไขของโครงการเท่านั้น และ
ห้ามเปลี่ยน default จาก A หรือ merge branch จนกว่าจะมี AIRC video-disjoint
held-out benchmark กับ paired bootstrap ครบ.

---

## 🧠 AI Agent Context Layer (NanoNets Graft)

โปรเจกต์นี้ได้รับการผสานเข้ากับ **[NanoNets Graft](https://github.com/nanonets/graft)** ซึ่งเป็นระบบ Context Layer & Code Graph สำหรับ AI Coding Agents (Gemini, Antigravity, Claude Code, Cursor, Copilot, Codex, Windsurf) ช่วยให้โมเดลเข้าใจแผนผังซอร์สโค้ด ฟังก์ชัน และ Call Graph ของทั้งโปรเจกต์ได้อย่างแม่นยำโดยไม่ต้องอ่านไฟล์ทั้งหมด

### คำสั่งใช้งาน Graft CLI:

```bash
# สร้าง/อัปเดต Code Graph ทันที ($0, No API Key required)
npx @nanonets/graft build

# แสดงแผนภาพสรุป Cluster และ Hubs สำคัญในระบบ
npx @nanonets/graft map

# ค้นหาตำแหน่งฟังก์ชันหรือโค้ดตามความหมาย
npx @nanonets/graft ask "ingestion pipeline"

# ตรวจสอบว่ามีฟังก์ชันไหนเรียกใช้งาน symbol นี้บ้าง (Call Graph Blast Radius)
npx @nanonets/graft callers ProgressiveIngestionManager

# เปิดหน้าต่าง Interactive Web Visualizer แผนผังโค้ด
npx @nanonets/graft viz
```


---

## 📜 การอ้างอิงทางวิชาการ (Academic Citation)

```bibtex
@article{videomoment2026,
  title   = {Pure-Visual Video Moment Retrieval and Temporal Localization using SigLIP 2 and Local Dense Captioning},
  author  = {Senior Project Research Group},
  year    = {2026},
  journal = {Computer Science Senior Capstone Project}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
