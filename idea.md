# ข้อเสนอแนะและบันทึกแนวคิด: Object-Level Symbolic Mapping ด้วย YOLO / Open-Vocabulary Detection

> **สถานะแนวคิด:** เสนอปรับปรุงและบันทึกข้อเสนอ (Proposed Architecture Idea)  
> **วันที่บันทึก:** 27 ตุลาคม 2026  
> **ไฟล์เป้าหมาย:** `idea.md`  

---

## 1. สำรวจสถาปัตยกรรมปัจจุบันในโปรเจกต์ (Current System Audit)

จากการตรวจสอบ Source Code ทั้งหมดใน Repository:
* **ปัจจุบัน "ไม่มี YOLO" อยู่ในโปรเจกต์เลย**
* สถาปัตยกรรมปัจจุบันประกอบด้วย:
  1. **SigLIP 2 (NaFlex) [768-dim]:** สกัดเวกเตอร์ระดับเฟรม (Dense Visual Embedding) บันทึกลงตาราง LanceDB `video_frames_v2`
  2. **CapRL-Qwen3VL-4B (GGUF):** วิเคราะห์ภาพระดับ Scene เพื่อสร้าง Dense Caption สรุปเหตุการณ์ บันทึกลงตาราง `scenes_v2` และทำหน้าที่ Verify ช่วงเวลาในโหมด Accurate
  3. **SAM 3.1 (Segment Anything):** เตรียมไว้สำหรับ Spatio-Temporal Mask Tracking (ปัจจุบันถูกปิดไว้เป็นค่า Default ด้วย `ENABLE_SAM_GROUNDING=false`)
  4. **Hybrid Retrieval:** นำ Cosine Similarity จาก SigLIP2 มาผสานน้ำหนัก (RRF) กับ BM25/FTS ของ Scene Caption แล้วเกลี่ยด้วย Gaussian Smoothing

---

## 2. ประเมินแนวคิดของผู้ใช้ (Idea Evaluation)

### 💡 สรุปแนวคิดที่เสนอ:
1. นำ **YOLO** เข้ามาช่วยตรวจจับวัตถุและบุคคลในแต่ละช่วงของวิดีโอ
2. บันทึกและ Mapping สิ่งที่ปรากฏในวิดีโออย่างละเอียด เช่น *"คนใส่เสื้อสีฟ้า"*, *"คนใส่หมวกสีเหลือง"*
3. ไม่พึ่งพาเพียง Vector Search เพียงอย่างเดียว แต่สร้าง Inverted Index / Text Tag จากวัตถุที่พบจริงในเฟรม
4. สุ่มตรวจจับเมื่อเฟรมมีความเปลี่ยนแปลง (Frame Difference / New Object) เพื่อไม่ให้สิ้นเปลืองการประมวลผล
5. เก็บข้อมูลเป็น Metadata / Text File ประจำวิดีโอ

---

### ✅ จุดแข็งและข้อดี (Why this is a Great Idea):
1. **แก้ปัญหา Semantic Bleed ของ Vector Search โดยตรง:**  
   โมเดลอย่าง CLIP/SigLIP รวมข้อมูลทั้งเฟรมเข้าเป็นเวกเตอร์เดียว (Global Pooling) ทำให้เวลาค้นหาคำที่มีความละเอียดเฉพาะเจาะจง (Attribute-level) เช่น *"หมวกสีเหลือง"* เวกเตอร์มักจะเพี้ยนหากฉากหลังมีแสงสีเหลือง หรือคนใส่เสื้อสีอื่นแต่มีวัตถุสีเหลืองปน การมี **Object-level Symbolic Tags** จะทำหน้าที่เป็น Hard Filter หรือ Exact Match ที่แม่นยำ 100%
2. **ตรวจจับวัตถุขนาดเล็ก (Small Objects) ได้ดีกว่า:**  
   Object Detector ถูกออกแบบมาพร้อม Feature Pyramid Network (FPN) สแกนวัตถุทุกสเกลตั้งแต่ขนาดเล็กถึงใหญ่ จึงเก็บรายละเอียดสิ่งของชิ้นเล็ก เช่น หมวก กระเป๋า แก้วน้ำ โทรศัพท์ ได้ครบถ้วนกว่าโมเดลแคปชันระดับภาพรวม
3. **แนวคิด Change Detection มีประสิทธิภาพสูง:**  
   การเช็ค Frame Difference / Significant Change ช่วยลดความซ้ำซ้อน ไม่ต้องประมวลผลทุกเฟรม ประหยัดเวลา Ingestion ได้มหาศาล

---

## 3. ข้อจำกัดของ Standard YOLO และทางเลือกที่แนะนำ (Critical Considerations)

### ⚠️ ข้อจำกัดสำคัญของ Standard YOLO (เช่น YOLOv8, YOLOv11):
* **Class ติดกรอบ COCO 80 Classes:** YOLO ทั่วไปถูกเทรนมาเพื่อตรวจจับเฉพาะ 80 หมวดหมู่มาตรฐาน (เช่น `person`, `car`, `chair`, `dog`)  
* **ไม่รู้จัก Attribute และสี:** Standard YOLO จะบอกได้เพียงว่าเป็น `person` **แต่ไม่สามารถบอกได้ว่า "ใส่เสื้อสีฟ้า" หรือ "ใส่หมวกสีเหลือง"** (หมวก หรือ `hat` ไม่ได้อยู่ใน 80 คลาสของ COCO ด้วยซ้ำ)

---

### 🚀 ทางออกทางเทคนิคที่เหมาะสมที่สุด (Recommended Solutions):

| แนวทาง | โมเดลที่แนะนำ | ข้อดี | ข้อพิจารณา |
| :--- | :--- | :--- | :--- |
| **แนวทาง A (แนะนำที่สุด)** | **YOLO-World** (Open-Vocabulary Detection) | • รันได้เร็วมาก (50–70+ FPS บน GPU)<br>• ค้นหาด้วย Free-text prompt เช่น `["blue shirt", "yellow hat", "backpack"]`<br>• ไม่จำกัดแค่ 80 คลาส | ต้องกำหนด Prompt Vocabulary หรือดึงคลาสพื้นฐาน + สกัดจาก Text Query |
| **แนวทาง B** | **Standard YOLO + Color/Attribute Extractor** | • YOLOv8/v11 ตัด Bounding Box ของ `person`<br>• ส่ง Crop ต่อให้ Color Histogram หรือ Mini-CLIP วิเคราะห์สีเสื้อ/หมวก | โค้ดมี Pipeline 2 ชั้น แต่ได้ผลลัพธ์สีเสื้อที่แน่นอนและแม่นยำมาก |
| **แนวทาง C** | **Florence-2 (Dense Region Captioning)** | • โมเดลเล็ก (~0.23B) จาก Microsoft<br>• บอกพิกัด Box พร้อมบรรยายวัตถุและสีครบในรอบเดียว | ความเร็วปานกลาง (~10-15 FPS) ช้ากว่า YOLO-World เล็กน้อย |

---

## 4. โครงสร้างการจัดเก็บข้อมูลที่แนะนำ (Data Storage Strategy)

เพื่อไม่ให้เปลืองพื้นที่จัดเก็บ และยังคงค้นหาได้เร็วในระดับมิลลิวินาที:

### โครงสร้างตารางใน LanceDB: `frame_objects_v1`
```json
{
  "video_id": "vid_abc123",
  "timestamp": 14.5,
  "frame_idx": 435,
  "objects": [
    {
      "label": "person",
      "attributes": ["blue shirt", "black pants"],
      "confidence": 0.92,
      "bbox": [0.15, 0.22, 0.45, 0.88]
    },
    {
      "label": "hat",
      "attributes": ["yellow"],
      "confidence": 0.85,
      "bbox": [0.24, 0.22, 0.35, 0.32]
    }
  ],
  "object_text": "person blue shirt black pants yellow hat"
}
```

* **การค้นหา (Retrieval):**
  - คอลัมน์ `object_text` สามารถทำ **Full-Text Search (FTS / BM25)** ใน LanceDB ได้ทันที
  - สามารถ Query ร่วมกับ SigLIP2 ในลักษณะ **Triple-Hybrid Fusion**:
    $$\text{Score} = w_1 \cdot \text{SigLIP2(Visual)} + w_2 \cdot \text{Qwen(Scene Caption)} + w_3 \cdot \text{YOLO(Object Tags)}$$

---

## 5. แผนการนำไปปฏิบัติ (Implementation Roadmap)

1. **Phase 1: Ingestion Pipeline Enrichment**
   - ใน `backend/app/pipeline/keyframe_filter.py` นำภาพ Keyframes มาผ่าน **YOLO-World**
   - สกัด Bounding Box และ Label / Attribute Tags บันทึกลง Metadata ประจำวิดีโอ (`data/objects/{video_id}.json`) หรือ LanceDB
2. **Phase 2: Search Engine Integration**
   - ปรับ `backend/app/retrieval/search_engine.py` เพิ่มสเตจตรวจค้นจาก Object Index
   - หากคำค้นมีชื่อวัตถุหรือสีเจาะจง ให้ดึง Timestamps ที่พบคู่วัตถุนั้นมา Boost คะแนนช่วงเวลา
3. **Phase 3: Frontend Visualization (Optional)**
   - ส่งผล Bounding Box กลับไปแสดงเป็น Overlay กรอบไฮไลต์บนตัวเล่นวิดีโอ (Video Player) เมื่อผู้ใช้คลิกดู Moment นั้นๆ
