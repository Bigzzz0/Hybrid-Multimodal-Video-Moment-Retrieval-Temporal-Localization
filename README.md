# Pure-Visual Video Moment Retrieval & Temporal Localization

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows_10%2F11-0078D6?style=for-the-badge&logo=windows&logoColor=white" alt="Windows" />
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Next.js_14-App_Router-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" alt="Next.js" />
  <img src="https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA" />
  <img src="https://img.shields.io/badge/LanceDB-Serverless_Vector_DB-00D2B4?style=for-the-badge" alt="LanceDB" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<p align="center">
  <b>ระบบค้นหาช่วงเวลาและเหตุการณ์ในวิดีโอด้วยภาษาธรรมชาติ (Natural Language Video Moment Retrieval & Temporal Boundary Localization)</b><br />
  ขับเคลื่อนด้วย <b>SigLIP 2 (NaFlex)</b>, แคปชันล่วงหน้าด้วย <b>CapRL-Qwen3VL-4B Q6</b>, ระบบคัดกรอง <b>Accurate Verifier</b> และ <b>LanceDB Vector Database</b><br />
  <i>ทำงานแบบ Local On-Premise 100% บนเครื่องของคุณ ไม่ส่งข้อมูลออกภายนอก และไม่มีค่าใช้จ่าย Cloud API</i>
</p>

---

## 🌟 จุดเด่นของระบบ (Key Highlights)

* 🔒 **100% Local On-Premise & Complete Privacy:** ประมวลผลและจัดเก็บฐานข้อมูลเวกเตอร์ทั้งหมดในเครื่อง วิดีโอไม่รั่วไหลสู่ภายนอก
* ⚡ **Consumer Hardware Optimized:** ออกแบบมาสำหรับเครื่องคอมพิวเตอร์ทั่วไปและ GPU สำหรับผู้ใช้ทั่วไป (เช่น RTX 3060, 4070, 5070 หรือ CPU Mode) พร้อมระบบจัดการ VRAM ป้องกัน Out-of-Memory
* 🚀 **Progressive Visual Ingestion:** เมื่ออัปโหลดวิดีโอ ระบบจะสร้างเวกเตอร์ SigLIP 2 และเปิดให้ค้นหาแบบ **Fast Search ได้ทันที** ในช่วงไม่กี่สิบวินาทีแรก จากนั้นระบบจะสร้างแคปชันละเอียด (CapRL Q6) ในเบื้องหลังโดยไม่บล็อกการใช้งาน
* 📈 **Dynamic Relevance Density Heatmap:** แถบเรืองแสงแสดงความเข้มข้นของเหตุการณ์ที่ตรงกับคำค้นหาตลอดทั้งวิดีโอแบบ 1-Hz Canvas ช่วยให้เห็นจุดสำคัญได้ทันที
* ⏱️ **Calibrated Multi-scale Temporal Localization:** สกัดช่วงเวลาเริ่มต้น-สิ้นสุด $[t_{start}, t_{end}]$ อย่างแม่นยำด้วย Gaussian Convolution, Soft-NMS และ Dynamic Thresholding

---

## 🚀 วิธีการติดตั้งและเริ่มใช้งานบน Windows (Quick Start Guide)

### 💻 ข้อกำหนดของระบบ (System Requirements)
1. **Windows 10 หรือ Windows 11** (64-bit)
2. **Python 3.11 หรือ 3.12** ([ดาวน์โหลดจาก python.org](https://www.python.org/downloads/))  
   *(⚠️ สำคัญ: ตอนติดตั้งต้องติ๊กเลือก **"Add python.exe to PATH"** ด้วย)*
3. **Node.js 18+ หรือ 20+** ([ดาวน์โหลดจาก nodejs.org](https://nodejs.org/))
4. **NVIDIA GPU** (แนะนำ VRAM 8GB–12GB+ เช่น RTX 3060, 4070, 5070) หรือสามารถรันบน **CPU** ได้โดยตรง

---

### 📦 ขั้นตอนที่ 1: ติดตั้งครั้งแรก (First-Time Setup)

เปิดโปรแกรม **PowerShell** หรือ **Command Prompt** แล้วรันคำสั่งตามลำดับ (ทำเพียงครั้งแรกครั้งเดียว):

#### 1. ติดตั้ง Backend (Python)
```powershell
# 1. เข้าสู่โฟลเดอร์ backend
cd backend

# 2. สร้าง Virtual Environment
python -m venv venv

# 3. เปิดใช้งาน Virtual Environment
.\venv\Scripts\activate

# 4. ติดตั้ง dependencies ทั้งหมด
pip install -r requirements.txt

# 5. สร้างไฟล์ตั้งค่า .env จากตัวอย่าง
copy .env.example .env

# 6. กลับมาที่โฟลเดอร์หลัก
cd ..
```

#### 2. ติดตั้ง Frontend (Next.js)
```powershell
# 1. เข้าสู่โฟลเดอร์ frontend
cd frontend

# 2. ติดตั้งแพ็กเกจ Node.js
npm install

# 3. กลับมาที่โฟลเดอร์หลัก
cd ..
```

---

### ⚡ ขั้นตอนที่ 2: วิธีสั่งรันระบบ (How to Run)

เลือกใช้งานได้ **2 วิธี** ตามความสะดวก:

---

### 🌟 วิธีที่ 1: รันคำสั่งเดียวจบ (One-Click Launcher - แนะนำที่สุด)

เพียงเปิด PowerShell ที่โฟลเดอร์หลัก แล้วพิมพ์คำสั่งเดียว:

```powershell
npm start
```

*(หรือสามารถใช้เมาส์ **ดับเบิลคลิกไฟล์ `start_all.bat`** ใน File Explorer ได้เลย)*

> 💡 **สคริปต์นี้จัดการทุกอย่างให้แบบอัตโนมัติในคลิกเดียว:**
> 1. **Auto Preload AI Models:** โหลดและวอร์มโมเดล SigLIP 2 Vision และตรวจสอบไฟล์ CapRL Q6 ให้ทันที (ไม่ต้องสั่ง preload แยกเอง)
> 2. **เปิด Backend API Server:** รันที่ `http://localhost:8000`
> 3. **เปิด AI Inference Worker:** รันที่ `http://localhost:8011`
> 4. **เปิด Frontend Web App:** รันที่ `http://localhost:3000`
> 5. **Auto Wait & Open Browser:** มีระบบตรวจเช็คสถานะจนกระทั่ง Backend พร้อมใช้งาน 100% แล้วจะเปิดเว็บเบราว์เซอร์ให้ทันที หมดปัญหาเปิดเว็บแล้วกดอัปโหลดไม่ติด!

🛑 **วิธีสั่งหยุดการทำงานทั้งหมด (Stop All):**
```powershell
npm run stop
```
*(หรือดับเบิลคลิกไฟล์ `stop_all.bat` หรือกดปิดหน้าต่าง Command Prompt แต่ละบาน)*

---

### 🛠️ วิธีที่ 2: รันแยกหน้าต่างด้วยตัวเอง (Manual Mode สำหรับ Developer / Debug)

หากต้องการแก้ไขโค้ด หรือต้องการดูข้อความ Log แยกหน้าต่างแบบละเอียด ให้เปิด **3 หน้าต่าง PowerShell**:

* **หน้าต่างที่ 1 — Backend API Server (`:8000`):**
  ```powershell
  cd backend
  .\venv\Scripts\activate
  python preload_models.py    # (ทางเลือก) วอร์มโมเดลเข้าหน่วยความจำล่วงหน้า
  python main.py              # เริ่มต้น Backend Server
  ```
  * ตรวจสอบ API Documentation ได้ที่: `http://localhost:8000/docs`

* **หน้าต่างที่ 2 — AI Inference Worker (`:8011`):**
  ```powershell
  cd backend
  .\venv\Scripts\activate
  python -m inference_worker.main
  ```

* **หน้าต่างที่ 3 — Frontend Web UI (`:3000`):**
  ```powershell
  cd frontend
  npm run dev
  ```
  * เข้าใช้งานหน้าเว็บที่: `http://localhost:3000`

---

## ⚙️ การตั้งค่าที่สำคัญ (Configuration: `backend/.env`)

ไฟล์ `backend/.env` เก็บค่าคอนฟิกหลักของระบบ สามารถเปิดแก้ไขได้ด้วย Notepad หรือ VS Code:

| ตัวแปร | ค่าที่แนะนำ | คำอธิบาย |
| :--- | :--- | :--- |
| `DEVICE` | `cuda` หรือ `cpu` | เลือกรันโมเดลบน GPU (`cuda`) หรือ CPU (`cpu`). หากใช้การ์ดรุ่นใหม่มากที่ PyTorch ตัวปัจจุบันยังไม่รองรับ ให้ตั้งเป็น `cpu` ระบบจะรันได้ตามปกติ |
| `HF_TOKEN` | `hf_xxx...` | Hugging Face Token (ใส่เมื่อต้องการดาวน์โหลดโมเดลครั้งแรก) |
| `INFERENCE_WORKER_ENABLED` | `true` | เปิด/ปิด AI Worker สำหรับสร้าง Dense Action Captions ในพื้นหลัง |
| `SIGLIP2_MODEL_ID` | `google/siglip2-base-patch16-naflex` | โมเดล Vision-Text ตัวหลักสำหรับค้นหาภาพและเหตุการณ์ |
| `LLAMA_CPP_PATH` | `backend\bin\llama\llama-server.exe` | Path ไฟล์รัน llama-server สำหรับโมเดล CapRL Q6 GGUF |
| `VLM_GGUF_DIR` | `backend\models\gguf` | โฟลเดอร์เก็บไฟล์ weights ของโมเดล CapRL Q6 GGUF |

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
  [ SigLIP 2 (NaFlex) ]       [ CapRL Q6 ]
   (768-dim Retrieval)       (Caption + Accurate Verify)
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
│  • Timeline Auto-Seek Video Player                                                      │
│  • Ranked Moment Cards with Timestamp Badges & Previews                                 │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ รายการเทคโนโลยีหลัก (Tech Stack)

| ส่วนประกอบ | เทคโนโลยีที่เลือกใช้ | บทบาทและหน้าที่ |
| :--- | :--- | :--- |
| **Visual-Text Backbone** | `google/siglip2-base-patch16-naflex` | สกัดเวกเตอร์ภาพ 768 มิติ โดยรักษา Aspect Ratio เดิมของวิดีโอ |
| **Dense Scene Captioner** | `CapRL-Qwen3VL-4B Q6_K` (llama.cpp) | สรุปเนื้อหาการกระทำและวัตถุในแต่ละซีนล่วงหน้าแบบเบื้องหลัง |
| **Online Verifier / VQA** | `CapRL-Qwen3VL-4B Q6_K` (llama.cpp) | ตรวจสอบความถูกต้องของเหตุการณ์และตอบคำถาม Visual Evidence |
| **Video Decoding**        | `Decord` (Hardware/Software) | ถอดรหัสเฟรมวิดีโออย่างแม่นยำและรวดเร็ว |
| **Vector Storage**        | `LanceDB` (Apache Arrow Format) | ฐานข้อมูลเวกเตอร์แบบ Serverless ประสิทธิภาพสูง ไม่เปลือง RAM |
| **Temporal Algorithm**    | `1D Gaussian Convolution & RRF` | กรองสัญญาณรบกวนและสกัดช่วงเวลาต่อเนื่อง $[t_s, t_e]$ |
| **Backend API**           | `FastAPI` + `Uvicorn` + `WebSockets` | REST API รองรับ Video Streaming (HTTP 206) และ Live Progress |
| **Frontend UI**           | `Next.js 14` + `React 18` + `Tailwind` | หน้าเว็บแบบ Responsive Dark Modern พร้อม Canvas Heatmap |

---

## 📖 ข้อมูลเชิงลึกสำหรับนักพัฒนา (Developer Deep Dive)

### 1. Progressive Two-Phase Ingestion
1. **Phase 1: Fast Ingestion (~30–45 วินาที)**  
   ถอดรหัสวิดีโอ, ตัดแบ่งซีนด้วย PySceneDetect, สุ่มคีย์เฟรมด้วย SSIM, สกัดเวกเตอร์ด้วย SigLIP 2 และบันทึกลง LanceDB เมื่อขั้นตอนนี้เสร็จ วิดีโอจะเข้าสู่สถานะ `phase1_ready` และสามารถค้นหาแบบ Fast Search ได้ทันที
2. **Phase 2: Background Dense Captioning**  
   AI Inference Worker จะนำซีนต่าง ๆ ไปสร้างแคปชันการกระทำ (Dense Captions) ด้วย CapRL Q6 ในพื้นหลัง เมื่อเสร็จสิ้นวิดีโอจะเปลี่ยนสถานะเป็น `caption_ready` เพื่อรองรับการค้นหาแบบข้อความเต็มรูปแบบ (Full-Text Search)

### 2. Search Profiles: Fast vs. Accurate
* **Fast Profile:** ค้นหาเวกเตอร์ภาพจาก SigLIP 2 ร่วมกับ BM25 แคปชันที่มีอยู่ผ่าน Reciprocal Rank Fusion (RRF) ตอบสนองอย่างรวดเร็วในระดับไม่กี่ร้อยมิลลิวินาที
* **Accurate Profile:** เสริมการวิเคราะห์ด้วย CapRL Q6 Verifier เพื่อยืนยันความสัมพันธ์ของการกระทำและตำแหน่งวัตถุที่ซับซ้อน พร้อมแสดงเหตุผลประกอบ (Visual Evidence)

### 3. สคริปต์จัดการแคปชันย้อนหลัง (Backfill Captions)
หากต้องการสั่งสร้างแคปชันสำหรับวิดีโอเดิมที่เคยอัปโหลดไว้:
```powershell
cd backend
.\venv\Scripts\activate
# สร้างแคปชันให้ทุกวิดีโอที่ยังไม่มี
python -m scripts.backfill_primary_captions --all-videos --resume

# สร้างแคปชันเฉพาะวิดีโอที่ระบุ
python -m scripts.backfill_primary_captions --video-id <VIDEO_ID> --resume
```

---

## 🧮 สูตรทางคณิตศาสตร์หลัก (Mathematical Formulations)

### 1. การคำนวณความคล้ายคลึงของเวกเตอร์ (Cosine Similarity)
$$S_{\text{vis}}(t) = \frac{\mathbf{e}_Q \cdot \mathbf{e}_{f_t}}{\|\mathbf{e}_Q\|_2 \|\mathbf{e}_{f_t}\|_2}$$

### 2. การรวมคะแนนหลายมิติ (Reciprocal Rank Fusion - RRF)
$$\text{RRF}(d) = \sum_{m \in \{\text{visual}, \text{caption}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

### 3. การกรองสัญญาณบนเส้นเวลา (1D Gaussian Temporal Convolution)
$$\tilde{\mathcal{S}}(t) = \mathcal{S}(t) * G_\sigma(t) = \int_{-\infty}^{\infty} \mathcal{S}(\tau) \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(t-\tau)^2}{2\sigma^2}\right) d\tau$$

### 4. การสกัดช่วงเวลาเหตุการณ์ (Dynamic Threshold Boundary Extraction)
$$[t_{\text{start}}, t_{\text{end}}] = \arg\max_{[t_s, t_e]} \int_{t_s}^{t_e} (\tilde{\mathcal{S}}(t) - \theta_{\text{dyn}}) \, dt \quad \text{where} \quad \theta_{\text{dyn}} = \mu_{\mathcal{S}} + \lambda \cdot \sigma_{\mathcal{S}}$$

---

## 📁 ผังโครงสร้างโปรเจกต์ (Project Tree)

```text
Hybrid-Multimodal-Video-Moment-Retrieval/
├── backend/
│   ├── app/
│   │   ├── api/v1/                      # Upload, Search, Video Streaming, WebSocket
│   │   ├── core/                        # Configuration (.env), Logger
│   │   ├── db/                          # LanceDB Vector Database schemas
│   │   ├── pipeline/                    # Decord, SceneDetect, SigLIP 2, CapRL Q6
│   │   └── retrieval/                   # RRF Fusion, Temporal Localization
│   ├── inference_worker/                # Local Worker รัน CapRL Q6 GGUF (:8011)
│   ├── tests/                           # Pytest Test Suite
│   ├── preload_models.py                # สคริปต์โหลดและแคชโมเดลล่วงหน้า
│   ├── main.py                          # FastAPI Server Entrypoint (:8000)
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/                         # Next.js 14 App Router
│   │   ├── components/
│   │   │   ├── player/                  # VideoPlayer, Timeline Heatmap Canvas
│   │   │   ├── search/                  # SearchBar, Moment Cards
│   │   │   └── upload/                  # Dropzone & Live Pipeline Tracker
│   │   └── lib/                         # API Client & Data Types
│   └── package.json
│
├── start_all.bat                        # สคริปต์เปิดทั้งระบบในคลิกเดียว (Windows CMD)
├── start_all.ps1                        # สคริปต์เปิดทั้งระบบในคลิกเดียว (PowerShell)
├── stop_all.bat                         # สคริปต์ปิดทุกเซอร์วิส
├── package.json                         # npm start / npm run stop
└── README.md
```

---

## 🧪 การรันชุดทดสอบ (Unit Testing)

ทดสอบการทำงานของฟังก์ชันสำคัญ เช่น RRF, Gaussian Convolution และ IoU:
```powershell
cd backend
.\venv\Scripts\activate
pytest tests/
```

---

## 🧠 AI Agent Context Layer (NanoNets Graft)

โปรเจกต์นี้รองรับ **[NanoNets Graft](https://github.com/nanonets/graft)** ช่วยให้ AI Coding Assistants เข้าใจโครงสร้างและ Call Graph ของโค้ดได้อย่างแม่นยำ:

```powershell
# อัปเดต Code Graph ของโปรเจกต์
npx @nanonets/graft build

# แสดงแผนภาพสรุป Modules สำคัญ
npx @nanonets/graft map

# ค้นหาตำแหน่งฟังก์ชันตามความหมาย
npx @nanonets/graft ask "ingestion pipeline"
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
