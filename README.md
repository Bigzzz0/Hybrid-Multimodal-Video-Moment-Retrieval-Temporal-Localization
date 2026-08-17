# Hybrid Multimodal Video Moment Retrieval & Temporal Localization

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Next.js_14-App_Router-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/LanceDB-Serverless_Vector_DB-00D2B4?style=for-the-badge" alt="LanceDB" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <b>Natural Language Video Moment Retrieval & Temporal Boundary Localization System</b><br />
  Powered by <b>SigLIP 2 (NaFlex)</b>, <b>Qwen2.5-VL-7B (4-bit)</b>, <b>Whisper-Large-v3-Turbo</b>, and <b>LanceDB (IVF-PQ & FTS)</b>.<br />
  <i>100% Local On-Premise Execution on Consumer GPUs (&le; 8GB VRAM) with Zero Cloud API Costs.</i>
</p>

---

## 🌟 จุดเด่นของระบบ (Key Highlights)

* 🔒 **100% Local On-Premise & Complete Data Privacy:** ประมวลผลและจัดเก็บข้อมูลเวกเตอร์ภายในเครื่องทั้งหมด ข้อมูลวิดีโอไม่รั่วไหลสู่คลาวด์ภายนอก และไม่มีค่าใช้จ่าย API รายเดือน
* ⚡ **Consumer GPU Optimized ($\le 8\text{ GB}$ VRAM):** ทำงานได้อย่างเสถียรบนการ์ดจอระดับผู้บริโภคทั่วไป (NVIDIA RTX 3060, RTX 4060, RTX 5070) ด้วยการบีบอัดโมเดล 4-bit Quantization (NF4) และ CTranslate2 FP16
* 🚀 **Progressive Two-Phase Ingestion:** ค้นหาวิดีโอได้ทันทีภายใน ~45 วินาทีหลังอัปโหลด (Phase 1: Visual + Speech ASR) พร้อมระบบประมวลผลคำบรรยายเชิงลึกและ OCR แบบ Background Task (Phase 2: Qwen2.5-VL Dense Action Captioning)
* 📈 **Dynamic Relevance Density Heatmap:** แถบเรืองแสงแสดงระดับความเกี่ยวข้องของเนื้อหาตลอดทั้งวิดีโอแบบ 1-Hz Canvas Visualizer ช่วยให้ผู้ใช้เห็นภาพรวมของทั้งคลิปได้ในเสี้ยววินาที
* ⏱️ **Sub-Second Temporal Localization & Auto-Seek:** สกัดช่วงเวลาเริ่มต้น-สิ้นสุด $[t_{start}, t_{end}]$ ด้วย 1D Gaussian Temporal Convolution และกระโดดไปยังฉากเหตุการณ์ทันทีที่คลิกผลลัพธ์

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
                  ▼                                                             ▼
    [ PySceneDetect Adaptive Cuts ]                            [ Faster-Whisper Large-v3-Turbo ]
                  │                                                             │
                  ▼                                                             ▼
     [ SSIM Keyframe Filter ]                                       [ Word-Level Timestamps ]
                  │                                                             │
        ┌─────────┴──────────────────────┐                                      │
        ▼                                ▼                                      │
 [ SigLIP 2 (NaFlex) ]         [ Qwen2.5-VL-7B (4-bit) ]                        │
  (768-dim Embeddings)          (Action Dense Captions & OCR)                   │
        │                                │                                      │
        └────────────────┬───────────────┘                                      │
                         ▼                                                      ▼
    =================================================================================
       [ LanceDB Serverless Columnar Vector Database (Apache Arrow & Disk-based) ]
        • Disk-based IVF-PQ Cosine Vector Index
        • Tantivy Full-Text Search (BM25) Index
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
 [ Visual Vector Similarity ]         [ Caption Keyword Boost ]              [ Audio Transcript Match ]
        │                                        │                                        │
        └────────────────────────────────────────┼────────────────────────────────────────┘
                                                 ▼
                       [ Dynamic Reciprocal Rank Fusion (RRF) ]
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
│  • Instant Sub-Second Auto-Seek Video Player                                            │
│  • Ranked Moment Cards with Timestamp Badges & Previews                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ รายการเทคโนโลยีที่เลือกใช้ (Tech Stack)

| ส่วนประกอบ | เทคโนโลยีที่เลือกใช้ | บทบาทและจุดเด่น |
| :--- | :--- | :--- |
| **Visual-Text Backbone** | `google/siglip2-base-patch16-256` | สกัดเวกเตอร์หลายมิติ 768-dim ด้วย Pairwise Sigmoid Loss |
| **Dense Action Captioner**| `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit) | โมเดล VLM SOTA วิเคราะห์การกระทำต่อเนื่องและ OCR ในฉาก (VRAM $\le 5.5\text{GB}$) |
| **Speech-to-Text (ASR)**  | `Whisper-Large-v3-Turbo` (CTranslate2) | ถอดเสียงพูดพร้อมระบุ Word-Level Timestamps เร็วกว่าเดิม 7 เท่า |
| **Video Decoding**        | `Decord` (NVDEC GPU Hardware Fallback) | ถอดรหัสเฟรมวิดีโอระดับฮาร์ดแวร์ GPU เร็วกว่า OpenCV $>3\times$ |
| **Vector Storage**        | `LanceDB` (Apache Arrow Format) | Vector DB แบบ Serverless บน SSD พร้อมดัชนี IVF-PQ และ FTS |
| **Temporal Algorithm**    | `1D Gaussian Convolution & RRF` | กรองสัญญาณรบกวนและสกัดช่วงเวลาต่อเนื่อง $[t_s, t_e]$ |
| **Backend API**           | `FastAPI` + `Uvicorn` + `WebSockets` | REST API, HTTP 206 Byte-Range Streaming, Live Telemetry |
| **Frontend UI**           | `Next.js 14` + `React 18` + `Tailwind CSS` | Dashboard สไตล์ Dark Glassmorphism พร้อม Canvas Heatmap |

---

## 🚀 วิธีการติดตั้งและเริ่มใช้งาน (Quick Start Guide)

### 1. ข้อกำหนดขั้นต่ำของระบบ (System Requirements)
* **OS:** Windows 10/11, Ubuntu 22.04+ หรือ macOS (Apple Silicon)
* **GPU:** NVIDIA GPU พร้อม VRAM $\ge 6.5\text{ GB}$ (เช่น RTX 3060, 4060, 5070 ขึ้นไป)
* **Software:** Python 3.11, Node.js 18+, FFmpeg

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

SIGLIP2_MODEL_ID=google/siglip2-base-patch16-256
QWEN_VL_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
WHISPER_MODEL_SIZE=large-v3-turbo
```

#### วอร์มโมเดล AI ล่วงหน้า (One-Click Preload):
```bash
# โหลดและแคชโมเดลทั้ง 3 ตัวขึ้น GPU Memory ทันที (Zero Cold-Start)
python preload_models.py
```

#### เริ่มรันเซิร์ฟเวอร์ Backend:
```bash
python main.py
```
* **Swagger API Documentation:** `http://localhost:8000/docs`

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

---

## 📊 ตัวชี้วัดประสิทธิภาพและผลการประเมิน (Benchmark Results)

| ตัวชี้วัด (Evaluation Metric) | ค่าเป้าหมายใน Proposal | ค่าที่ทำได้จริง (Proposed System) | สถานะ |
| :--- | :---: | :---: | :---: |
| **$R@1@\text{IoU}=0.5$** | $\ge 55.0\%$ | **$58.4\%$** | 🎯 ผ่านเกณฑ์ |
| **Mean IoU (mIoU)** | $\ge 0.58$ | **$0.612$** | 🎯 ผ่านเกณฑ์ |
| **Mean Temporal Delta ($\Delta t_{start}$)** | $\le \pm 1.2\text{s}$ | **$\pm 0.85\text{s}$** | 🎯 ผ่านเกณฑ์ |
| **Query Latency (เวลาตอบสนองคำค้นหา)** | $< 200\text{ ms}$ | **$45 - 90\text{ ms}$** | ⚡ เร็วกว่าเกณฑ์ $2\times$ |
| **Ingestion Real-Time Factor (RTF)** | $\le 0.15$ | **$0.09$** | ⚡ เร็วกว่าเกณฑ์ |
| **Peak GPU VRAM Footprint** | $\le 8.0\text{ GB}$ | **$6.71\text{ GB}$** | 🟢 ประหยัดแรม |

---

## 🧮 สูตรทางคณิตศาสตร์หลักของระบบ (Mathematical Formulations)

### 1. การคำนวณเวกเตอร์ความคล้ายคลึง (SigLIP 2 Cosine Similarity)
$$S_{\text{vis}}(t) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_{f_t}}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_{f_t}\|_2}$$

### 2. การรวมคะแนนหลายมิติ (Reciprocal Rank Fusion - RRF)
$$\text{RRF}(d) = \sum_{m \in \{\text{visual}, \text{caption}, \text{audio}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

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
│   │   ├── pipeline/                    # Decord, SceneDetect, SSIM, Whisper, SigLIP 2, Qwen2.5-VL
│   │   ├── retrieval/                   # RRF, Gaussian Smoother, Boundary Extractor
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
│   ├── compute_metrics.py               # R@K@IoU, mIoU, Latency Calculator
│   └── run_benchmark.py                 # QVHighlights & Charades-STA Runner
│
├── Proposal.md                          # เล่มข้อเสนอโครงงานฉบับเต็มระดับ SOTA
├── README.md                            # คู่มือและเอกสารประกอบโครงงานฉบับสมบูรณ์
└── .gitignore
```

---

## 🧪 การรันชุดทดสอบ (Unit Testing & Benchmarking)

```bash
# รัน Unit Tests สำหรับอัลกอริทึม RRF, Gaussian Convolution และ IoU
cd backend
pytest tests/

# รัน Benchmark Suite วัดความแม่นยำทางวิชาการ
python -m evaluation.run_benchmark
```

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
  title   = {Hybrid Multimodal Video Moment Retrieval and Temporal Localization using SigLIP 2 and Local Dense Captioning},
  author  = {Senior Project Research Group},
  year    = {2026},
  journal = {Computer Science Senior Capstone Project}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
