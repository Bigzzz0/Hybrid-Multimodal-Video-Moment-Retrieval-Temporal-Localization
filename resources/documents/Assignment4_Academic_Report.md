# รายงานทางวิชาการ (Assignment 4)
## การศึกษาทฤษฎีและงานวิจัยที่เกี่ยวข้องเพื่อการพัฒนาโครงงาน

> **สถานะเอกสาร:** รายงานนี้เป็น literature review และ academic snapshot ของช่วงก่อน production implementation ไม่ใช่คู่มือ runtime ปัจจุบัน ให้ดูภาคผนวก `Current Implementation Status` และ [docs/current-runtime.md](docs/current-runtime.md) สำหรับระบบที่ใช้งานจริง

---

# [ปกนอก / ปกใน]

<br />

<div align="center">

### รายงานการศึกษาทฤษฎีและงานวิจัยที่เกี่ยวข้อง
### (Literature Review and Theoretical Framework Report)

<br />

## **ระบบสืบค้นและระบุช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติแบบ Pure-Visual**
## **(Pure-Visual Video Moment Retrieval and Temporal Localization System)**

<br /><br />

**จัดทำโดย**  
กลุ่มนักศึกษาโครงงานวิจัยระดับปริญญาตรี  
สาขาวิชาวิทยาการคอมพิวเตอร์  

<br />

**เสนอ**  
ผู้ช่วยศาสตราจารย์ ดร. สิลดา อินทรโสธรฉันท์  

<br /><br />

**รายงานนี้เป็นส่วนหนึ่งของรายวิชาโครงงานวิทยาการคอมพิวเตอร์**  
**สาขาวิชาวิทยาการคอมพิวเตอร์ คณะวิทยาศาสตร์ มหาวิทยาลัยขอนแก่น**  
**ภาคการศึกษาปลาย ปีการศึกษา 2568**

</div>

<br /><br />

---

# บทคัดย่อ

รายงานฉบับนี้มีวัตถุประสงค์เพื่อศึกษา สังเคราะห์ และเปรียบเทียบทฤษฎีของระบบสืบค้นและระบุตำแหน่งช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติ โดยจำกัดขอบเขตไว้ที่ข้อมูลภาพและ dense visual scene captions การศึกษาครอบคลุม SigLIP 2 NaFlex, Qwen visual temporal reasoning, LanceDB และ Reciprocal Rank Fusion ร่วมกับ multi-scale temporal proposals, calibration และ Soft-NMS ผ่านกรอบ 5W1H และตารางสังเคราะห์วรรณกรรม

ผลการศึกษานี้ใช้เพื่อกำหนดแผนพัฒนาและวิธี benchmark ไม่ใช่การอ้างผลลัพธ์ล่วงหน้า ตัวเลข R@K, mIoU, temporal F1, latency และ VRAM ต้องมาจาก held-out test ที่แบ่งตามวิดีโอ และรายงานพร้อม confidence interval เท่านั้น ระบบเป้าหมายทำงาน local และรองรับ no-match กับหลาย occurrence โดยไม่ใช้ข้อมูลเสียงหรือข้อความบนภาพ

**คำสำคัญ:** การสืบค้นช่วงเวลาในวิดีโอ (Video Moment Retrieval), แบบจำลองภาษาภาพ (Vision-Language Models), การระบุขอบเขตเวลา (Temporal Localization), ฐานข้อมูลเวกเตอร์ (Vector Database), การปรับเทียบความเชื่อมั่น (Calibration)

<br />

---

# Abstract

This academic report investigates pure-visual Natural Language Video Moment Retrieval and temporal boundary localization for long-form video. The review focuses on SigLIP 2 NaFlex frame/text embeddings, dense visual scene captions, Qwen visual temporal verification, LanceDB, Reciprocal Rank Fusion, multi-scale proposals, and confidence calibration using a 5W1H framework and synthesis matrix.

The review defines an implementable pure-visual architecture rather than claiming unmeasured accuracy. Its acceptance criteria are held-out R@1/R@5 at IoU .3/.5/.7, mIoU, temporal F1 for repeated events, no-match precision/recall, p50/p95 latency, and peak VRAM. All claims must be regenerated from an immutable baseline and a video-disjoint test set; the target is local execution within 8 GB VRAM.

**Keywords:** Video Moment Retrieval, Vision-Language Models, Temporal Localization, Vector Database, Confidence Calibration

<br />

---

# คำนำ

รายงานฉบับนี้จัดทำขึ้นเพื่อรวบรวม ศึกษา และวิเคราะห์องค์ความรู้สำหรับพัฒนาโครงงาน "ระบบสืบค้นและระบุช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติแบบ Pure-Visual (Pure-Visual Video Moment Retrieval and Temporal Localization System)"

เนื้อหาภายในรายงานครอบคลุมความเป็นมาของปัญหาการสืบค้นวิดีโอความยาวสูง การวิเคราะห์วรรณกรรมวิจัยสำคัญด้วยกรอบคำถาม 5W1H การศึกษาสูตรและทฤษฎีทางคณิตศาสตร์ที่เกี่ยวข้อง การเปรียบเทียบข้อดีและข้อจำกัดของแต่ละแนวทางผ่านตารางสังเคราะห์วรรณกรรม (Synthesis Matrix) ตลอดจนการสรุปประเด็นสำคัญเพื่อนำไปประยุกต์ใช้ต่อยอดในการพัฒนาระบบจริง

ผู้จัดทำขอขอบพระคุณ ผู้ช่วยศาสตราจารย์ ดร. สิลดา อินทรโสธรฉันท์ อาจารย์ประจำรายวิชา ที่ได้ให้คำแนะนำ หลักเกณฑ์การจัดทำเอกสารทางวิชาการ และข้อเสนอแนะอันเป็นประโยชน์อย่างยิ่ง ผู้จัดทำหวังเป็นอย่างยิ่งว่ารายงานฉบับนี้จะเป็นประโยชน์ต่อการศึกษาค้นคว้าและการพัฒนาเทคโนโลยีด้านการประมวลผลสื่อวิดีโอหลายมิติต่อไป

<div align="right">
<b>คณะผู้จัดทำ</b><br />
สาขาวิชาวิทยาการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
</div>

<br />

---

# สารบัญ

* **บทคัดย่อ**
* **Abstract**
* **คำนำ**
* **สารบัญ**
* **สารบัญตาราง**
* **สารบัญภาพ**
* **บทที่ 1 บทนำ**
  * 1.1 ความเป็นมาและความสำคัญของปัญหา
  * 1.2 จุดประสงค์ของการศึกษา
  * 1.3 ขอบเขตของการศึกษา
  * 1.4 ประโยชน์ที่คาดว่าจะได้รับ
* **บทที่ 2 ทฤษฎีและงานวิจัยที่เกี่ยวข้อง**
  * 2.1 งานวิจัยที่เกี่ยวข้อง (การสังเคราะห์วรรณกรรมตามกรอบ 5W1H)
    * 2.1.1 งานวิจัยด้านการโมเดลความสัมพันธ์เชิงเวลาและการตรวจจับด้วย Transformer/DETR
  * 2.1.2 งานวิจัยด้านแบบจำลองภาษาภาพและวิดีโอ
    * 2.1.3 งานวิจัยด้านการตรวจสอบลำดับการกระทำเชิงภาพ
    * 2.1.4 งานวิจัยด้านการตรวจจับรอยต่อฉาก (Shot Boundary Detection)
  * 2.2 ทฤษฎีและหลักการทางคณิตศาสตร์ที่เกี่ยวข้อง
    * 2.2.1 ทฤษฎี Vision-Language Joint Latent Space และ NaFlex Resolution
    * 2.2.2 ทฤษฎี Pairwise Sigmoid Loss ในการฝึกแบบจำลอง
    * 2.2.3 ทฤษฎีการรวมผลลัพธ์ด้วย Reciprocal Rank Fusion (RRF)
    * 2.2.4 ทฤษฎี multi-scale temporal proposals และ boundary refinement
    * 2.2.5 ทฤษฎีการปรับเทียบความเชื่อมั่นและ no-match threshold
    * 2.2.6 ทฤษฎีการจัดเก็บและสร้างดัชนีเวกเตอร์แบบ Disk-based IVF-PQ
  * 2.3 การเปรียบเทียบผลการศึกษาและช่องว่างของงานวิจัย
    * 2.3.1 ตารางสังเคราะห์วรรณกรรม (Synthesis Matrix)
    * 2.3.2 บทวิเคราะห์สังเคราะห์วรรณกรรม (Literature Review Synthesis)
* **บทที่ 3 สรุปผลการศึกษาและแนวทางการนำไปใช้**
  * 3.1 สรุปผลการศึกษาทฤษฎีและวรรณกรรม
  * 3.2 แนวทางการนำไปประยุกต์ใช้ต่อยอดในการพัฒนาโครงงาน
* **เอกสารอ้างอิง**
* **ภาคผนวก**
  * ภาคผนวก ก: รายละเอียดชุดข้อมูลทดสอบมาตรฐานสากล (Benchmark Datasets)
  * ภาคผนวก ข: ข้อกำหนดฮาร์ดแวร์และการใช้หน่วยความจำ VRAM ของแต่ละแบบจำลอง
  * ภาคผนวก ค: ตัวอย่างคำค้นหาภาษาธรรมชาติและผลการสกัดช่วงเวลา [t_start, t_end]

<br />

---

# สารบัญตาราง

* **ตารางที่ 2.1:** สรุปการวิเคราะห์งานวิจัยที่เกี่ยวข้องด้วยกรอบ 5W1H
* **ตารางที่ 2.2:** ตารางสังเคราะห์วรรณกรรม (Synthesis Matrix) เปรียบเทียบสถาปัตยกรรมและมิติข้อมูล
* **ตารางที่ 3.1:** การกำหนดค่าพารามิเตอร์ทางทฤษฎีสำหรับการประยุกต์ใช้ในระบบสืบค้นไฮบริด

<br />

---

# สารบัญภาพ

* **รูปที่ 2.1:** สถาปัตยกรรม Joint Latent Space และกลไก NaFlex Dynamic Resolution ของ SigLIP 2
* **รูปที่ 2.2:** กระบวนการรวมผลคะแนนหลายมิติด้วย Reciprocal Rank Fusion (RRF)
* **รูปที่ 2.3:** การปรับความเรียบของสัญญาณเวลาด้วย Multi-Scale 1D Gaussian Temporal Convolution
* **รูปที่ 2.4:** การสกัดขอบเขตเวลา [t_start, t_end] ด้วยกลไก Dynamic Threshold และ Valley Detection
* **รูปที่ 3.1:** แผนผังวงจรการทำงานเชิงแนวคิดของระบบสืบค้นวิดีโอแบบไฮบริดสองเฟส (Progressive Data Flow)

<br />

---

# บทที่ 1
# บทนำ

### 1.1 ความเป็นมาและความสำคัญของปัญหา
ในยุคสารสนเทศปัจจุบัน ข้อมูลวิดีโอความยาวสูง (Long-form Video) ได้กลายเป็นสื่อหลักในการบันทึกและถ่ายทอดความรู้ในหลากหลายบริบท เช่น วิดีโอบันทึกการเรียนการสอนและการนำเสนอทางวิชาการ (Lecture & Presentation Archives), วิดีโอบันทึกการประชุม (Meeting Recordings), ตลอดจนฟุตเทจจากกล้องวงจรปิดและกล้องติดหน้ารถยนต์ (Surveillance & Driving Footage) อย่างไรก็ตาม ปัญหาคอขวดสำคัญที่เกิดขึ้นคือ **"ภาระในการค้นหาและระบุตำแหน่งช่วงเวลาที่เกิดเหตุการณ์เฉพาะเจาะจง" (Data Overload and Manual Video Scrubbing)** ซึ่งผู้ใช้งานจำเป็นต้องเสียเวลาเปิดรับชมหรือเลื่อนแถบเวลา (Time Bar) ยาวนานหลายสิบนาทีถึงหลายชั่วโมงเพื่อค้นหาเหตุการณ์สั้น ๆ เพียงไม่กี่วินาที

ระบบสืบค้นวิดีโอแบบดั้งเดิมส่วนใหญ่ยังคงพึ่งพา metadata ระดับไฟล์ จึงเข้าถึงเนื้อหาเชิงช่วงเวลาได้จำกัด งานนี้มุ่งให้ embedding และ scene caption เข้าใจวัตถุ การกระทำ ลำดับการเคลื่อนไหว และการเปลี่ยนสถานะจากภาพจริง เช่น *"ช่วงที่คนสวมเสื้อแดงเดินเข้ามาหยิบกระเป๋า"* หรือ *"ตอนที่รถจักรยานยนต์เลี้ยวตัดหน้า"* โดยไม่ใช้ข้อมูลนอกภาพเป็นหลักฐาน

ในช่วงปี ค.ศ. 2024–2026 งานด้าน visual-language foundation models มีความก้าวหน้าที่เหมาะกับการทำงาน local ได้แก่:
1. **SigLIP 2 (Google DeepMind, 2025):** แบบจำลอง Vision-Language ที่รองรับ Native Flexible Dynamic Resolution (NaFlex) ช่วยให้รักษารายละเอียดภาพและตำแหน่งเชิงพื้นที่ในวิดีโออัตราส่วน 16:9 ได้อย่างแม่นยำ
2. **Qwen2.5-VL-7B:** โมเดลภาษาภาพสำหรับ dense visual scene captions และ temporal verification แบบจำกัดเวลา
3. **LanceDB (2024):** ฐานข้อมูลเวกเตอร์แบบฝังตัวบน Apache Arrow ที่รองรับ vector search และ caption FTS/BM25

การศึกษาทฤษฎีและวรรณกรรมจึงมีความสำคัญเพื่อสังเคราะห์กรอบ pure-visual ที่วัดผลได้จริง โดยแยก Fast retrieval ออกจาก Accurate verifier และไม่อ้างประสิทธิภาพที่ยังไม่มี benchmark รองรับ

### 1.2 จุดประสงค์ของการศึกษา
1. เพื่อศึกษาทฤษฎี สถาปัตยกรรม และวิวัฒนาการของแบบจำลองการสืบค้นช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติ (Video Moment Retrieval & Temporal Grounding)
2. เพื่อศึกษาหลักการทำงานของ SigLIP 2 NaFlex, dense visual captions, Qwen temporal verifier และฐานข้อมูลเวกเตอร์แบบ local
3. เพื่อศึกษาการผสานคะแนน RRF, multi-scale proposals, boundary refinement, Soft-NMS และ Platt calibration สำหรับระบุ [t_start, t_end]
4. เพื่อสังเคราะห์ข้อดี ข้อจำกัด และช่องว่างทางวิจัยสำหรับออกแบบระบบ pure-visual ที่รองรับหลาย occurrence และ no-match

### 1.3 ขอบเขตของการศึกษา
* **ขอบเขตด้านรูปแบบข้อมูล:** ศึกษาข้อมูลวิดีโอประเภท Long-form Video รูปแบบไฟล์ดิจิทัล (.mp4, .mov, .mkv) ครอบคลุมวิดีโอบรรยายการสอน วิดีโอการประชุม และฟุตเทจกล้องหน้ารถยนต์
* **ขอบเขตด้านแบบจำลองและอัลกอริทึม:** ศึกษา Vision-Language (CLIP, SigLIP 1, SigLIP 2 NaFlex, Qwen2.5-VL), LanceDB/Apache Arrow, RRF, multi-scale proposals, Soft-NMS และ calibration
* **ขอบเขตด้านชุดข้อมูลมาตรฐานสากล:** ศึกษาเกณฑ์การวัดผลและลักษณะข้อมูลบนชุดข้อมูลมาตรฐาน QVHighlights (Lei et al., 2021) และ Charades-STA (Gao et al., 2017)
* **ขอบเขตด้านสภาพแวดล้อมการประมวลผล:** ศึกษาแนวทางการประมวลผลแบบ 100% Local On-Premise ภายใต้ข้อจำกัดของฮาร์ดแวร์ระดับผู้บริโภค (VRAM ไม่เกิน 8 GB)

### 1.4 ประโยชน์ที่คาดว่าจะได้รับ
1. ได้รับชุดความรู้เชิงลึกเกี่ยวกับทฤษฎีและวิวัฒนาการของระบบสืบค้นวิดีโอด้วยภาษาธรรมชาติ
2. ทราบถึงจุดเด่น ข้อจำกัด และช่องว่างทางวิจัยของแบบจำลองแต่ละประเภทผ่านตารางสังเคราะห์วรรณกรรม (Synthesis Matrix)
3. ได้รับพิมพ์เขียวเชิงสถาปัตยกรรม (Architectural Blueprint) สำหรับนำไปใช้ต่อยอดในการพัฒนาซอฟต์แวร์ระบบสืบค้นวิดีโอจริงที่มีความแม่นยำสูง รวดเร็ว และประหยัดทรัพยากร

<br />

---

# บทที่ 2
# ทฤษฎีและงานวิจัยที่เกี่ยวข้อง

### 2.1 งานวิจัยที่เกี่ยวข้อง (การสังเคราะห์วรรณกรรมตามกรอบ 5W1H)

เพื่อทำความเข้าใจวิวัฒนาการ ระเบียบวิธีวิจัย และข้อค้นพบสำคัญในสาขาการสืบค้นช่วงเวลาในวิดีโอด้วยภาษาธรรมชาติอย่างลึกซึ้ง การศึกษานี้ได้ทำการทบทวนวรรณกรรมสำคัญที่เกี่ยวข้องจำนวน 12 งานวิจัย โดยใช้กรอบการคิดวิเคราะห์ **5W1H (Who, What, Where, When, Why, How)** เพื่อศึกษารายละเอียดของผู้วิจัย สิ่งที่พัฒนา แหล่งเผยแพร่ ปัญหาที่เป็นแรงจูงใจ และวิธีการดำเนินงานอย่างเป็นระบบ โดยสามารถจำแนกงานวิจัยออกเป็น 4 กลุ่มหลักตามความเรียงทางวิชาการดังต่อไปนี้:

#### 2.1.1 งานวิจัยด้านการโมเดลความสัมพันธ์เชิงเวลาและการตรวจจับด้วย Transformer/DETR
1. **โครงข่าย 2D-TAN สำหรับการระบุตำแหน่งช่วงเวลาในวิดีโอ (Zhang et al., 2020)**  
   Zhang และคณะ (2020) จาก University of Rochester และ Microsoft Research ได้นำเสนอโครงข่าย *2D Temporal Adjacent Networks (2D-TAN)* ในการประชุมวิชาการระดับนานาชาติ AAAI Conference on Artificial Intelligence (AAAI 2020; [arXiv:1912.03590](https://arxiv.org/abs/1912.03590)) เพื่อแก้ไขข้อจำกัดของระเบียบวิธีแบบ 1D Sliding Window หรือ Segment Selection ในอดีตที่ไม่สามารถมองเห็นและวิเคราะห์ความสัมพันธ์ของช่วงเวลาเหตุการณ์ที่เหลื่อมซ้อนหรืออยู่ติดกันได้อย่างเป็นระบบ คณะผู้วิจัยได้พัฒนาวิธีการแปลงช่วงเวลาทั้งหมดของวิดีโอให้อยู่ในรูปของ 2D Temporal Feature Map โดยกำหนดให้แกนหนึ่งแทนจุดเริ่มต้นของช่วงเวลา (t_start) และอีกแกนแทนจุดสิ้นสุด (t_end) จากนั้นจึงประยุกต์ใช้ 2D Convolutional Layers ในการดึงความสัมพันธ์ของบริบทเชิงเวลาก่อนจับคู่กับเวกเตอร์คำค้นหาภาษาธรรมชาติ ส่งผลให้แบบจำลองสามารถค้นหาและระบุตำแหน่งของ Moment ที่ตรงกับคำค้นหาได้อย่างมีประสิทธิภาพสูงขึ้น

2. **ชุดข้อมูลมาตรฐาน QVHighlights และแบบจำลอง Moment-DETR (Lei et al., 2021)**  
   Lei, Berg, และ Bansal (2021) จาก University of North Carolina at Chapel Hill ได้นำเสนอชุดข้อมูลมาตรฐานสากล *QVHighlights* พร้อมทั้งแบบจำลอง *Moment-DETR* ในการประชุม Advances in Neural Information Processing Systems (NeurIPS 2021; [arXiv:2107.09609](https://arxiv.org/abs/2107.09609)) เพื่อแก้ไขปัญหาของชุดข้อมูลการสืบค้นวิดีโอแบบดั้งเดิม (เช่น Charades-STA หรือ ActivityNet-Captions) ที่มุ่งเน้นเฉพาะการตัดขอบเขตเวลาอย่างหยาบโดยไม่มีการประเมินคะแนนความโดดเด่นของเนื้อหารายวินาที (Fine-grained Saliency Score) โดยผู้วิจัยได้ประยุกต์ใช้สถาปัตยกรรม DEtection TRansformer (DETR) แปลงภารกิจการตรวจจับช่วงเวลาให้เป็นปัญหา Direct Set Prediction ที่สามารถทำนายทั้งพิกัดช่วงเวลา [t_start, t_end] และเส้นโค้งคะแนนความโดดเด่น (Highlight Score) ตลอดทั้งคลิปวิดีโอไปพร้อมกัน ซึ่งกลายเป็นหมุดหมายสำคัญของเกณฑ์การประเมินระบบสืบค้นวิดีโอสมัยใหม่

3. **แบบจำลอง QD-DETR สำหรับการสร้าง Video Representation ตามคำค้นหา (Moon et al., 2023)**  
   Moon และคณะ (2023) จาก Korea University ได้นำเสนอแบบจำลอง *QD-DETR (Query-Dependent DETR)* ในการประชุมวิชาการ IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023) เพื่อแก้ไขข้อจำกัดของแบบจำลองกลุ่ม DETR ก่อนหน้าที่มักเข้ารหัสคุณลักษณะของวิดีโอโดยไม่ขึ้นกับคำค้นหา (Query-Agnostic) ส่งผลให้แบบจำลองไม่สามารถแยกแยะฉากที่ไม่เกี่ยวข้องออกจากฉากเป้าหมายได้ดีเท่าที่ควร โดยคณะผู้วิจัยได้เสนอกลไก Query-Dependent Cross-Attention เพื่อนำเวกเตอร์คำค้นหาไปปรับแต่งการเข้ารหัสวิดีโอตั้งแต่ชั้นแรกๆ ร่วมกับการใช้ Loss Function พิเศษที่ฝึกฝนด้วย Negative Video-Text Pairs เพื่อกดคะแนนความโดดเด่นของฉากที่ไม่เกี่ยวข้องลง ทำให้ระบบสามารถสืบค้นและระบุตำแหน่งช่วงเวลาได้อย่างแม่นยำสูงขึ้น

4. **กรอบการทำงานแบบรวมศูนย์ UniVTG สำหรับการระบุตำแหน่งเชิงเวลา (Lin et al., 2023)**  
   Lin และคณะ (2023) จาก Sea AI Lab และ Microsoft ได้นำเสนอกรอบการทำงาน *UniVTG (Unified Video-Language Temporal Grounding)* ในการประชุม IEEE/CVF International Conference on Computer Vision (ICCV 2023) เพื่อแก้ไขปัญหาความกระจัดกระจายของสถาปัตยกรรมโมเดลในอดีตที่มักแยกการพัฒนาสำหรับแต่ละภารกิจย่อย เช่น Video Moment Retrieval, Highlight Detection, และ Video Summarization ออกจากกัน ทำให้ไม่สามารถถ่ายโอนองค์ความรู้ข้ามชุดข้อมูลได้อย่างเต็มศักยภาพ ผู้วิจัยจึงได้ออกแบบสถาปัตยกรรมแบบรวมศูนย์ (Unified Transformer Backbone) ที่รับสัญญาณภาพและข้อความคำค้น แล้วทำนายทั้งขอบเขตเวลาและเส้นโค้งคะแนนความเกี่ยวข้องไปพร้อมกัน ส่งผลให้ได้ประสิทธิภาพระดับแนวหน้าในหลายชุดข้อมูลทดสอบ

5. **สถาปัตยกรรม ActionFormer สำหรับการตรวจจับการกระทำแบบ Anchor-Free (Zhang et al., 2022)**  
   Zhang, Wu, และ Li (2022) จาก University of Wisconsin-Madison และ Nanjing University ได้นำเสนอสถาปัตยกรรม *ActionFormer* ในการประชุม European Conference on Computer Vision (ECCV 2022) เพื่อแก้ไขปัญหาความไม่ยืดหยุ่นของแบบจำลองแบบดั้งเดิมที่ต้องอาศัยกรอบเวลาคงที่ (Anchor Windows) ในการตรวจจับเหตุการณ์ ส่งผลให้ประสิทธิภาพลดลงเมื่อเผชิญกับกิจกรรมที่มีความยาวแตกต่างกันอย่างมาก โดยคณะผู้วิจัยได้พัฒนาโครงสร้างแบบ Anchor-Free ผสาน Multi-scale 1D Temporal Convolution เข้ากับ Self-Attention Mechanism เพื่อสร้าง Feature Pyramid และทำนายขอบเขตเวลาแบบจุดต่อจุด ส่งผลให้สามารถตรวจจับการกระทำสั้นและการกระทำยาวได้อย่างแม่นยำ

#### 2.1.2 งานวิจัยด้านแบบจำลองภาษาภาพและวิดีโอ (Vision-Language & Video Models)
6. **แบบจำลอง SigLIP 2 และกลไก NaFlex Dynamic Resolution (Google DeepMind, 2025)**  
   Google DeepMind (Tschannen et al., 2025; [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)) ได้นำเสนอแบบจำลอง *SigLIP 2* ซึ่งเป็น Vision-Language Encoder เจเนอเรชันใหม่ เพื่อแก้ไขข้อจำกัดของแบบจำลอง CLIP และ SigLIP 1 เดิมที่บังคับปรับภาพให้เป็นสี่เหลี่ยมจัตุรัสและขนาดคงที่ ทำให้สูญเสียรายละเอียดภาพและมิติเชิงพื้นที่ในวิดีโออัตราส่วน 16:9 คณะผู้วิจัยได้พัฒนานวัตกรรม *NaFlex (Native Flexible Dynamic Resolution)* ที่รองรับการประมวลผลภาพตามอัตราส่วนจริงโดยไม่ต้องยืดหรือตัดขอบภาพ ควบคู่กับการใช้ฟังก์ชันการสูญเสียแบบ Pairwise Sigmoid Loss, การฝึกฝนด้วย Masked Prediction, และ Self-Distillation ส่งผลให้ SigLIP 2 มีความสามารถในการเข้าใจความหมายเชิงพื้นที่ (Spatial Awareness) และรายละเอียดของวัตถุในวิดีโอเหนือกว่าแบบจำลองในอดีตอย่างมีนัยสำคัญ

7. **Qwen2.5-VL สำหรับ dense visual captions และ temporal verification**
   Qwen2.5-VL ใช้สร้างคำบรรยายระดับ scene จากเฟรมจริงและตรวจสอบ top candidates ใน Accurate mode การส่ง timestamp เป็น metadata และการจำกัดผลลัพธ์เป็น JSON ที่อ้างอิงดัชนีเฟรมช่วยให้ boundary ใหม่อยู่บนเวลาที่สังเกตได้จริง และทำให้ระบบ fallback ได้เมื่อ JSON ผิดหรือเกิน time budget

8. **แบบจำลอง TimeChat สำหรับการเข้าใจลำดับเวลาในวิดีโอยาว (Ren et al., 2024)**  
   Ren และคณะ (2024) จาก Peking University และ Microsoft Research ได้นำเสนอแบบจำลอง *TimeChat* ในการประชุม CVPR 2024 เพื่อแก้ไขปัญหาความไม่เข้าใจมิติเวลา (Time-Blindness) ของ Large Language Models ทั่วไปที่ไม่สามารถผูกข้อความเข้ากับลำดับเวลาของเฟรมวิดีโอได้อย่างแม่นยำ โดยคณะผู้วิจัยได้ออกแบบ Time-Aware Frame Encoder ร่วมกับการสร้าง Time Binding Tokens เพื่อป้อนตำแหน่งเวลาเข้าสู่ LLM ส่งผลให้โมเดลสามารถตอบคำถามและให้เหตุผลเชิงลำดับเวลาพร้อมระบุพิกัด Timestamp ของเหตุการณ์ในวิดีโอยาวได้อย่างละเอียด

9. **แบบจำลอง Video-LLaVA สำหรับการรวม Representation ภาพและวิดีโอ (Lin et al., 2024)**  
   Lin และคณะ (2024) จาก Peking University ได้นำเสนอแบบจำลอง *Video-LLaVA* ในการประชุมวิชาการ EMNLP 2024 เพื่อแก้ไขข้อจำกัดจากการแยกกระบวนการเรียนรู้ภาพนิ่งและวิดีโอออกจากกัน ซึ่งทำให้โมเดลไม่สามารถถ่ายทอดความรู้เชิงแนวคิด (Concept Transfer) ได้อย่างสมบูรณ์ ผู้วิจัยได้ออกแบบกระบวนการ Joint Representation Alignment โดยจัดวางเวกเตอร์ภาพนิ่งและลำดับเฟรมวิดีโอให้อยู่ในพื้นที่ Representation เดียวกันก่อนส่งผ่าน Language Projection Layer เข้าสู่ LLM ส่งผลให้โมเดลมีความเข้าใจเหตุการณ์และการกระทำในวิดีโอได้อย่างลึกซึ้งยิ่งขึ้น

#### 2.1.3 งานวิจัยด้านการตรวจสอบลำดับการกระทำเชิงภาพ (Visual Temporal Verification)
10. **การตรวจสอบแบบสองขั้นและการผูก timestamp กับเฟรมจริง**
    แนวทาง two-stage retrieval ช่วยแยกการค้นหาผู้สมัครจำนวนมากออกจากการให้เหตุผลเชิงภาพที่มีต้นทุนสูง งานนี้จึงใช้ frame/caption retrieval ใน Fast และส่งเพียง top 3 ช่วงไปยัง Qwen ใน Accurate โดยให้โมเดลเลือก index ของเฟรมที่ส่งเข้าไปเท่านั้น

11. **ข้อกำหนดของระบบ pure-visual**
    การวิเคราะห์ทั้งหมดของโครงงานต้องยึดจากภาพเฟรมและ caption ที่สร้างจากภาพเท่านั้น ไม่ใช้เสียงหรือข้อความบนภาพเป็นสัญญาณค้นหา

12. **แบบจำลอง ImageBind สำหรับการผูกมัด 6 โมดัลลิตีด้วยภาพ (Girdhar et al., 2023)**  
    Girdhar และคณะ (2023) จาก Meta AI ได้นำเสนอแบบจำลอง *ImageBind* ในการประชุม CVPR 2023 ซึ่งเป็นงานวิจัยบุกเบิกการสร้างพื้นที่เวกเตอร์ร่วมเดี่ยว (Unified Embedding Space) ที่ผูกมัดข้อมูล 6 โมดัลลิตี ได้แก่ ภาพ/วิดีโอ, ข้อความ, เสียง, ข้อมูลเชิงลึก (Depth), ภาพความร้อน (Thermal), และข้อมูลเซนเซอร์การเคลื่อนไหว (IMU) โดยใช้ภาพเป็นแกนกลางในการเชื่อมโยง (Binding Modality) ซึ่งพิสูจน์ให้เห็นว่าการผสานข้อมูลหลายมิติสามารถทำได้โดยไม่จำเป็นต้องมีข้อมูลจับคู่โดยตรงระหว่างทุกโมดัลลิตี อันเป็นรากฐานทฤษฎีสำคัญของการสืบค้นแบบไฮบริด

#### 2.1.4 งานวิจัยด้านการตรวจจับรอยต่อฉาก (Shot Boundary Detection)
13. **แบบจำลอง TransNet V2 สำหรับการตรวจจับรอยต่อฉากความเร็วสูง (Souček & Lokoč, 2020)**  
    Souček และ Lokoč (2020) จาก Charles University ได้นำเสนอแบบจำลอง *TransNet V2* สำหรับงานตรวจจับรอยตัดต่อฉากในวิดีโอ (Shot Transition Detection) เพื่อแก้ไขปัญหาคอขวดด้านเวลาในการประมวลผลทุกเฟรมของวิดีโอความยาวสูง โดยใช้สถาปัตยกรรม Dilated 3D Convolutional Neural Network ที่สามารถประมวลผลด้วยความเร็วสูงพิเศษมากกว่า 500 เฟรมต่อวินาที เพื่อตรวจจับทั้งรอยตัดฉากแบบฉับพลัน (Hard Cuts) และรอยตัดฉากแบบค่อยเป็นค่อยไป (Gradual Transitions) ซึ่งช่วยให้ระบบสามารถแบ่งวิดีโอออกเป็นฉากย่อยและลดปริมาณเฟรมซ้ำซ้อนลงได้มากกว่าร้อยละ 75

<br />

**ตารางที่ 2.1:** สรุปการวิเคราะห์งานวิจัยที่เกี่ยวข้องด้วยกรอบ 5W1H

| งานวิจัย | ผู้พัฒนา (Who) | ปี (When) | แหล่งเผยแพร่ / ลิงก์ | สาระสำคัญที่พัฒนา (What) | ปัญหาที่แก้ไข (Why) | วิธีการหลัก (How) |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **2D-TAN** | Zhang et al. | 2020 | AAAI / [arXiv:1912.03590](https://arxiv.org/abs/1912.03590) | 2D Temporal Map Modeling | 1D Window ขาดความสัมพันธ์ข้ามช่วงเวลา | 2D Adjacent Feature Map |
| **QVHighlights / Moment-DETR** | Lei et al. | 2021 | NeurIPS / [arXiv:2107.09609](https://arxiv.org/abs/2107.09609) | Joint Moment & Saliency Grounding | ขาดเกณฑ์ประเมินระดับวินาที | Transformer Set Prediction |
| **QD-DETR** | Moon et al. | 2023 | CVPR | Query-Dependent Video DETR | ฉากที่ไม่เกี่ยวข้องได้คะแนนสูง | Query Cross-Attention & Negative Pairs |
| **UniVTG** | Lin et al. | 2023 | ICCV | Unified Temporal Grounding | ขาดกรอบการทำงานร่วมกัน | Unified Transformer Backbone |
| **ActionFormer** | Zhang et al. | 2022 | ECCV | Anchor-free 1D Temporal Grounding | Anchor Box ขาดความยืดหยุ่น | Multi-scale 1D Temporal Pyramid |
| **SigLIP 2** | Google DeepMind | 2025 | arXiv / [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) | NaFlex Vision-Language Model | ภาพ 16:9 ถูกยืดและสูญเสียมิติ | NaFlex Dynamic Resolution & Sigmoid Loss |
| **Qwen2.5-VL** | Qwen Team | 2024 | Technical report | Dense visual caption + temporal verification | VLM ใช้เวลาสูง | Sequential top-3 rerank |
| **TimeChat** | Ren et al. | 2024 | CVPR | Time-Sensitive Video-LLM | LLM ขาดความเข้าใจเรื่องเวลา | Time Binding Tokens & Frame Encoder |
| **Video-LLaVA** | Lin et al. | 2024 | EMNLP | Unified Image-Video Representation | การแยกภาพกับวิดีโอออกจากกัน | Joint Representation Alignment |
| **Pure-visual constraint** | โครงงานนี้ | 2026 | Implementation specification | ใช้เฉพาะภาพเฟรมและ scene captions | ไม่รองรับสัญญาณนอกภาพ | Explicit modality contract |
| **LanguageBind** | Zhu et al. | 2024 | ICLR | Multi-modal Binding Space | การสืบค้นพึ่งพาเฉพาะมิติภาพ | Language-Centric Contrastive Learning |
| **ImageBind** | Girdhar et al. | 2023 | CVPR / [arXiv:2305.05665](https://arxiv.org/abs/2305.05665) | One Embedding for 6 Modalities | การผสานข้ามโมดัลลิตีต้องการคู่ข้อมูลมหาศาล | Image-Centric Joint Binding Space |
| **TransNet V2** | Souček & Lokoč | 2020 | arXiv | Fast Shot Boundary Detection | ประมวลผลทุกเฟรมทำให้ระบบช้า | Dilated 3D Convolutional Network |

<br />

---

### 2.2 ทฤษฎีและหลักการทางคณิตศาสตร์ที่เกี่ยวข้อง

#### 2.2.1 ทฤษฎี Vision-Language Joint Latent Space และ NaFlex Resolution
แบบจำลอง Vision-Language สมัยใหม่ทำงานโดยการแปลงข้อมูลภาพคีย์เฟรม f_t และคำค้นหาข้อความ Q ให้อยู่ในปริภูมิต่างมิติเดียวกัน (Joint Latent Space) ขนาด d = 768 มิติ โดยคำนวณคะแนนความคล้ายคลึงด้วย Cosine Similarity:

```text
Cosine_Similarity(f_t, Q) = (v_t · u_q) / (||v_t|| × ||u_q||)
```

โดยที่:
* **v_t** = E_vis(f_t) ∈ ℝ^768 คือ เวกเตอร์แทนคุณลักษณะของภาพคีย์เฟรม
* **u_q** = E_text(Q) ∈ ℝ^768 คือ เวกเตอร์แทนคำค้นหาภาษาธรรมชาติ
* **กลไก NaFlex (Native Flexible Dynamic Resolution)** ใน SigLIP 2 ช่วยให้ตัวเข้ารหัสภาพสามารถรับภาพอัตราส่วน 16:9 โดยไม่ต้องตัดขอบหรือปรับยืดภาพ ทำให้สามารถรักษารายละเอียดเชิงพื้นที่ของวัตถุและตัวอักษรในภาพได้อย่างสมบูรณ์

```
┌────────────────────────────────────────────────────────────────────────┐
│  Keyframe f_t (16:9) ──► NaFlex Vision Encoder (ViT) ──► v_t ∈ ℝ^768   │
│                                                            ▲           │
│                                                Cosine      │           │
│                                              Similarity    │           │
│                                                            ▼           │
│  Text Query Q        ──► Multilingual Text Encoder   ──► u_q ∈ ℝ^768   │
└────────────────────────────────────────────────────────────────────────┘
```
**รูปที่ 2.1:** สถาปัตยกรรม Joint Latent Space และกลไก NaFlex Dynamic Resolution ของ SigLIP 2

#### 2.2.2 ทฤษฎี Pairwise Sigmoid Loss ในการฝึกแบบจำลอง
SigLIP 2 ใช้ฟังก์ชันการสูญเสียแบบ Pairwise Sigmoid Loss แทนที่ Softmax Contrastive Loss แบบดั้งเดิม ซึ่งช่วยขจัดปัญหาการพึ่งพาขนาด Global Batch Size และทำให้การแยกแยะเวกเตอร์มีความเสถียรสูงขึ้น:

```text
L_SigLIP = - ∑_{i=1}^{N} ∑_{j=1}^{N} [ y_ij · log σ(t · u_i · v_j + b) + (1 - y_ij) · log(1 - σ(t · u_i · v_j + b)) ]
```

โดยที่:
* **y_ij = 1** เมื่อคู่ภาพและข้อความเป็นคู่ที่ตรงกัน (Positive Pair)
* **y_ij = 0** เมื่อคู่ภาพและข้อความเป็นคู่ตรงข้าม (Negative Pair)
* **σ(z) = 1 / (1 + e^(-z))** คือ ฟังก์ชัน Sigmoid Activation
* **t** คือ Temperature Parameter (ค่าสัมประสิทธิ์ปรับอุณหภูมิความชัน)
* **b** คือ Learnable Bias Parameter (ค่าไบแอสที่ปรับตามการเรียนรู้)
* **N** คือ จำนวนตัวอย่างข้อมูลในมินิแบตช์ (Batch Size)

#### 2.2.3 ทฤษฎีการสร้าง temporal proposals จาก visual relevance
ระบบสร้าง raw relevance timeline ที่ 2Hz จากทุกเฟรมที่ indexed แล้วคำนวณ rolling windows หลายสเกล 2, 4, 8, 16 และ 32 วินาที โดย candidate center ต้องเป็น local maximum ของแต่ละสเกล คะแนนรวม peak relevance, mean relevance และ boundary contrast เพื่อรองรับทั้งเหตุการณ์สั้น ยาว และเหตุการณ์ซ้ำ

#### 2.2.4 ทฤษฎีการรวมผลลัพธ์ด้วย Reciprocal Rank Fusion (RRF)
เพื่อรวมผลการค้นหาจากสองแหล่งที่มีมาตรวัดต่างกัน (visual frame similarity และ generated scene-caption BM25) ทฤษฎี RRF (Cormack et al., 2009) ถูกนำมาใช้โดยไม่ต้องปรับสเกลคะแนน:

```text
RRF(d) = ∑_{m ∈ M} [ w_m / (k + r_m(d)) ]
```

โดยที่:
* **M = {Visual, Caption}** คือ เซตของแหล่งสัญญาณภาพในระบบ
* **r_m(d)** คือ ลำดับอันดับ (Rank) ของข้อมูล d ในโมดัลลิตี m
* **k** คือ ค่าคงที่ปรับความเรียบ (กำหนดค่า k = 60 เพื่อป้องกันอันดับต้นมีน้ำหนักสูงเกินไป)
* **w_m** คือค่าน้ำหนักจาก tuning artifact (ค่าเริ่มต้น visual = 0.80, caption = 0.20) และ normalize ใหม่เมื่อไม่มี caption ที่ generated

```
             ┌─────────────────────────┐
             │ Text Query: "คนยกมือถาม" │
             └────────────┬────────────┘
        ┌─────────────────┬─────────────────┐
        ▼                 ▼
 ┌──────────────┐  ┌──────────────┐
 │ Visual Match │  │ Caption Match│
 │ (Rank r_vis) │  │ (Rank r_cap) │
 └──────┬───────┘  └──────┬───────┘
        │                 │
        └─────────────────┼─────────────────┘
                          ▼
            ┌───────────────────────────┐
            │   Reciprocal Rank Fusion  │
            │   RRF(d) = Σ w / (k + r)  │
            └─────────────┬─────────────┘
                          ▼
             [ Fused Temporal Score S(t) ]
```
**รูปที่ 2.2:** กระบวนการรวมผลคะแนนหลายมิติด้วย Reciprocal Rank Fusion (RRF)

#### 2.2.5 ทฤษฎี multi-scale proposal และ boundary refinement
แทนการตีความ camera motion ว่าเป็น semantic relevance ระบบรักษา raw timeline แยกจาก display-normalized timeline แล้วใช้ scene boundaries, จุดที่คะแนนลดต่ำกว่า 60% ของ peak และ transition-energy peaks เพื่อปรับขอบเขต proposal การปรับเรียบเป็นเพียงเครื่องมือแสดงผลและไม่ถูกใช้เป็น confidence โดยตรง

Raw timeline ใช้สำหรับ calibration และ proposal scoring ส่วน normalized timeline ใช้แสดง heatmap เท่านั้น รูปแบบการแสดงผลอาจใช้ smoothing ได้ แต่ห้ามใช้ normalized value เป็น no-match confidence

#### 2.2.6 ทฤษฎีการปรับขอบเขตด้วย energy-quantile refinement
การกำหนดจุดเริ่มต้นและสิ้นสุดของ [t_start, t_end] ใช้จุดที่ relevance ลดต่ำกว่า 60% ของ peak ร่วมกับ scene boundary และ transition-energy peak ส่วน quantile 8%/92% เป็นเพียง **energy-quantile refinement**

```text
boundary = refine(peak, scene_boundary, transition_energy, quantile=[0.08, 0.92])
```

โดยที่:
* **peak** คือจุด local maximum ของ proposal
* **scene_boundary** คือขอบเขตฉากที่ได้จาก ingestion
* **transition_energy** คือความต่างของเฟรม ใช้ช่วยวาง boundary เท่านั้น

จากนั้นขยายขอบเขตจาก local peak ออกไปจนคะแนนลดต่ำกว่า 60% ของ peak หรือชน scene boundary แล้วใช้ Gaussian Soft-NMS ที่ IoU 0.5 เพื่อลดช่วงซ้ำ โดยคงเหตุการณ์ที่ไม่ทับกันทั้งหมด

```text
[t_start, t_end] = boundary_refinement(relevance, peak, scenes, transition_energy)
```

```
relevance peak ──► expand while score >= 0.60 × peak
       │          ├─ clamp to scene boundaries
       │          └─ refine with transition-energy peaks
       └─► Gaussian Soft-NMS (IoU 0.5) ──► disjoint occurrences
```
**รูปที่ 2.4:** การสกัดขอบเขตเวลาและการรักษาเหตุการณ์ที่ไม่ทับซ้อนกัน

#### 2.2.7 ทฤษฎีการจัดเก็บและสร้างดัชนีเวกเตอร์แบบ Disk-based IVF-PQ
เพื่อรองรับการจัดเก็บเวกเตอร์จำนวนมากในระดับ On-Premise สถาปัตยกรรม LanceDB ใช้การจัดเก็บข้อมูลแบบ Columnar บน Apache Arrow Format ร่วมกับดัชนี Inverted File with Product Quantization (IVF-PQ):
1. **Inverted File (IVF):** แบ่งพื้นที่เวกเตอร์ออกเป็น K กลุ่มคลัสเตอร์ด้วย k-means เพื่อจำกัดขอบเขตการค้นหาเฉพาะคลัสเตอร์ที่ใกล้เคียง
2. **Product Quantization (PQ):** แบ่งเวกเตอร์ขนาด 768 มิติออกเป็น M ส่วนย่อย และบีบอัดแต่ละส่วนเป็นตัวแทนรหัส (Codebook) ทำให้ประหยัดหน่วยความจำได้กว่า 8–16 เท่า และสามารถคำนวณระยะทาง Cosine Distance ได้บนหน่วยความจำดิสก์โดยตรงโดยไม่ต้องโหลดข้อมูลทั้งหมดขึ้น RAM

<br />

---

### 2.3 การเปรียบเทียบผลการศึกษาและช่องว่างของงานวิจัย

#### 2.3.1 ตารางสังเคราะห์วรรณกรรม (Synthesis Matrix)
เพื่อเปรียบเทียบแนวทางการพัฒนาและชี้ให้เห็นช่องว่างทางงานวิจัย (Research Gaps) ตารางที่ 2.2 ได้สรุปการสังเคราะห์วรรณกรรมในแต่ละมิติ:

<br />

**ตารางที่ 2.2:** ตารางสังเคราะห์วรรณกรรม (Synthesis Matrix) เปรียบเทียบสถาปัตยกรรมและมิติข้อมูล

| แนวทาง / งานวิจัย | มิติข้อมูลที่รองรับ (Modalities) | สถาปัตยกรรมหลัก (Core Backbone) | วิธีการสกัดพิกัดเวลา (Temporal Localization) | จุดเด่นสำคัญ (Key Strengths) | ข้อจำกัดและช่องว่างวิจัย (Gaps & Limitations) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposal-based VMR**<br>*(2D-TAN, TALL)* | Visual Only (2D CNN) | Sliding Window & 2D Temporal Feature Map | Exhaustive Proposal Intersection | เข้าใจความสัมพันธ์ของช่วงเวลาซ้อนทับได้ดี | ไม่รองรับ Open-Vocabulary, คำนวณช้ามากบนวิดีโอยาว |
| **DETR-based Set Prediction**<br>*(Moment-DETR, QD-DETR)* | Visual + Text | Transformer Encoder-Decoder | DETR Bounding Coordinate Regression | ทำนายช่วงเวลาและ Highlight Score พร้อมกันได้ดี | ต้องการการ Fine-tune บนชุดข้อมูลปิด, กินหน่วยความจำ GPU สูง |
| **Unified Temporal Grounding**<br>*(UniVTG, ActionFormer)* | Visual + Text | Multi-scale 1D Temporal Convolution | Point-wise Anchor-free Regression | สกัดเหตุการณ์สั้นและยาวได้ยืดหยุ่นสูง | ต้องปรับให้เหมาะกับ index แบบ zero-shot และหลาย occurrence |
| **Time-Sensitive Video-LLM**<br>*(TimeChat, Video-LLaVA)* | Visual + Text | Large Language Model | Time Binding Query Tokens | เข้าใจการกระทำระดับสูงและตอบคำถามเชิงเหตุผลได้ | ใช้เวลาและหน่วยความจำสูง จึงเหมาะกับ rerank top candidates |
| **Two-Stage Selective Filtering**<br>*(SeViLA, LanguageBind)* | Multi-modal Binding Space | Localizer-Filter + Answerer VLM | Top-K Keyframe Selection | ประหยัด Token ในงานตอบคำถาม Video QA | ขาดระบบดัชนีเวกเตอร์ที่รองรับการค้นหาระดับคลังข้อมูลขนาดใหญ่ |
| **Proposed Pure-Visual Architecture**<br>*(โครงงานนี้)* | **Visual Vector + Dense Scene Caption** | **SigLIP 2 NaFlex + Qwen (4-bit)** | **Tuned RRF + Multi-scale Proposals + Platt Calibration + Soft-NMS** | **รักษา aspect ratio, รองรับหลาย occurrence/no-match, local VRAM ≤ 8 GB** | ต้อง reindex ข้อมูล v1 และวัดผลบน held-out test |

<br />

#### 2.3.2 บทวิเคราะห์สังเคราะห์วรรณกรรม (Literature Review Synthesis)
จากการทบทวนวรรณกรรมข้างต้น สามารถสรุปประเด็นสังเคราะห์สำคัญได้ 3 ประการ:
1. **การใช้ visual channels สองแบบ:** frame embeddings ให้ coverage ของ timeline ส่วน dense scene captions ช่วยจับความสัมพันธ์เชิงการกระทำ การ fuse ต้องตรวจ caption status และไม่ให้ fallback generic caption เพิ่มคะแนน
2. **ความสำคัญของอัตราส่วนภาพและการแปลงเวกเตอร์ (Spatial Resolution Awareness):** การที่แบบจำลองรุ่นใหม่อย่าง SigLIP 2 นำเสนอ NaFlex ช่วยแก้ปัญหาภาพผิดเพี้ยนจากการยืดขยายภาพแบบดั้งเดิม ทำให้การจับคู่คำค้นหากับวัตถุในวิดีโอ 16:9 มีความแม่นยำสูงขึ้นอย่างก้าวกระโดด
3. **ความสมดุลระหว่างความเร็วและความละเอียด:** Fast ทำ frame retrieval และ proposal generation ส่วน Accurate ส่ง top 3 candidates ให้ Qwen ตรวจสอบแบบ sequential ภายใต้ 15 วินาทีและ fallback เมื่อ timeout/OOM

<br />

---

# บทที่ 3
# สรุปผลการศึกษาและแนวทางการนำไปใช้

### 3.1 สรุปผลการศึกษาทฤษฎีและวรรณกรรม
การศึกษาทฤษฎีและงานวิจัยที่เกี่ยวข้องสามารถสรุปสาระสำคัญได้ดังนี้:
1. **แบบจำลองภาพและภาษา:** SigLIP 2 NaFlex ทำ frame/text embeddings ส่วน Qwen ทำ scene captions และ visual temporal verification โดยยังคง local execution
2. **การผสานคะแนนและระบุขอบเขต:** RRF, multi-scale proposals, boundary refinement และ Soft-NMS คืนเหตุการณ์ที่ไม่ทับกันหลายรายการ จากนั้น Platt calibration แยก rank score ออกจาก probability และ no-match gate
3. **การประมวลผล local:** Decord และ LanceDB ช่วยทำงานในเครื่อง โดยต้องวัด latency และ peak VRAM จริงใน benchmark ไม่สรุปจากสเปกโมเดล

### 3.2 แนวทางการนำไปประยุกต์ใช้ต่อยอดในการพัฒนาโครงงาน
จากการศึกษาทฤษฎีข้างต้น ผู้จัดทำได้กำหนดพิมพ์เขียวเชิงสถาปัตยกรรมสำหรับนำไปต่อยอดในการพัฒนาโครงงานจริง ดังแสดงในรูปที่ 3.1:

```
========================= PROGRESSIVE PURE-VISUAL INGESTION PIPELINE =========================

 [ ไฟล์วิดีโอต้นฉบับ (.mp4 / .mov) ]
                 │
      ▼
 [ Decord Video Reader ]
      │
      ▼
 [ Scene + Transition Sampling ]
      │
   ┌──┴───────────────────────────┐
   ▼                              ▼
[ SigLIP 2 NaFlex Encoder ] [ Qwen2.5-VL (4-bit) ]
(เฟรม embeddings จริง)       (scene captions / verifier)
   │                              │
   └──────────────┬───────────────┘
                  ▼
 ==============================================================================
  [ LanceDB Serverless Columnar Vector Database (Apache Arrow & Disk IVF-PQ) ]
   - Table: videos, scenes_v2, video_frames_v2, index_metadata
 ==============================================================================

============================ PURE-VISUAL REAL-TIME RETRIEVAL ============================

 [ คำค้นหาภาษาธรรมชาติ: "ช่วงที่มีการนำเสนอสไลด์เปรียบเทียบผลลัพธ์" ]
                 │
      ┌──────────┴──────────────────────────────┐
      ▼                                         ▼
 [ SigLIP 2 Text Variants ]              [ Caption BM25/FTS ]
      │                                         │
      └───────────────────┬─────────────────────┘
                          ▼
            [ Tuned Visual/Caption RRF + 2Hz Timeline ]
                          │
                          ▼
        [ Multi-Scale Proposals + Boundary Refinement + Soft-NMS ]
                          │
                          ▼
        [ Platt Calibration + No-Match Threshold ]
                          │
                          ▼
       [ ส่งผลลัพธ์หลาย Moments, score probability และ 2-Hz Heatmap ]
```
**รูปที่ 3.1:** แผนผังวงจรการทำงานเชิงแนวคิดของระบบสืบค้นวิดีโอแบบ Pure-Visual สองโปรไฟล์

<br />

**ตารางที่ 3.1:** การกำหนดค่าพารามิเตอร์ทางทฤษฎีสำหรับการประยุกต์ใช้ในระบบสืบค้นไฮบริด

| องค์ประกอบทางเทคนิค | แบบจำลอง / อัลกอริทึมที่เลือกใช้ | ค่าพารามิเตอร์ที่กำหนด | เหตุผลเชิงทฤษฎีรองรับ |
| :--- | :--- | :--- | :--- |
| **ตัวสกัดเวกเตอร์ภาพ** | `google/siglip2-base-patch16-naflex` | เฟรม embedding จริง, dimension ตรวจจาก checkpoint | รักษา aspect ratio และ timeline coverage |
| **ตัวบรรยายภาพ/ตรวจสอบ** | `Qwen2.5-VL-7B-Instruct` | 4-bit ตาม memory budget | สร้าง scene captions และตรวจ top-3 candidates จากเฟรมจริง |
| **การผสานคะแนน RRF** | Reciprocal Rank Fusion | k = 60, visual/caption = 0.80/0.20 ก่อน tune | caption unavailable จะไม่เพิ่มคะแนน |
| **การสร้าง proposals** | Multi-scale rolling windows | 2, 4, 8, 16, 32 วินาที และขยายตามความยาว | รองรับ short/long และ repeated events |
| **การปรับขอบเขต** | Energy-quantile refinement + Soft-NMS | 60% ของ peak, IoU = 0.5 | รักษาเหตุการณ์ที่ไม่ทับกันและ clamp ตาม scene |
| **การปรับเทียบ** | Platt calibration | positive เมื่อ IoU ≥ 0.5, มี no-match queries | แยก rank score ออกจาก probability และเลือก threshold จาก dev |
| **ฐานข้อมูลเวกเตอร์** | LanceDB (Apache Arrow) | index v2 + FTS/BM25 captions | ตรวจ schema/model metadata ก่อนสลับใช้งาน |

<br />

---

# เอกสารอ้างอิง

<div style="padding-left: 2em; text-indent: -2em;">

Bai, S., et al. (2024). *Qwen2.5-VL: Enhancing Vision-Language Models for Fine-Grained Multimodal Understanding and Localization*. arXiv preprint.

Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual machine learning methods for search result fusion. In *Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval* (pp. 758–759). https://doi.org/10.1145/1571941.1572114

Gao, J., Sun, C., Yang, Z., & Nevatia, R. (2017). TALL: Temporal activity localization via language query. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 5267–5275). https://doi.org/10.1109/ICCV.2017.563

Girdhar, R., El-Nouby, A., Liu, Z., Mann, M., Rabbat, M., Singh, M., ... & Misra, I. (2023). ImageBind: One embedding space to bind them all. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 15180–15190). https://doi.org/10.1109/CVPR52729.2023.01457

Google DeepMind. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic and Spatial Awareness*. arXiv preprint arXiv:2502.14786. https://arxiv.org/abs/2502.14786

LanceDB Authors. (2024). *LanceDB: Serverless, Developer-friendly Vector Database for Multimodal AI*. LanceDB Whitepaper & Documentation.

Lei, J., Berg, T. L., & Bansal, M. (2021). QVHighlights: Detecting moments and highlights in videos via natural language queries. In *Advances in Neural Information Processing Systems (NeurIPS 2021)* (Vol. 34, pp. 11846–11858). https://arxiv.org/abs/2107.09609

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., & Yuan, L. (2024). Video-LLaVA: Learning united visual representation by alignment before projection. In *Findings of the Association for Computational Linguistics: EMNLP 2024* (pp. 6784–6798).

Lin, K. Q., Zhang, P., Chen, J., Pramanick, S., Gao, X., Dai, P., & Yan, S. (2023). UniVTG: Towards unified video-language temporal grounding. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 2794–2804). https://doi.org/10.1109/ICCV51070.2023.00262

Moon, W., Hyun, S., Park, S., Park, D., & Heo, J. P. (2023). Query-dependent video representation for moment retrieval and highlight detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 23023–23033). https://doi.org/10.1109/CVPR52729.2023.02206

Ren, S., Yao, L., Li, Z., Sun, Y., & Zhu, X. (2024). TimeChat: A time-sensitive multimodal large language model for long video understanding. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 14313–14323).

Souček, T., & Lokoč, J. (2020). TransNet V2: An effective deep network for fast video shot boundary detection. *arXiv preprint arXiv:2008.05238*.

Yu, S., Cho, J., Yadav, P., & Bansal, M. (2023). Self-chained video-language alignment for fast video qa and moment retrieval. In *Advances in Neural Information Processing Systems (NeurIPS 2023)* (Vol. 36, pp. 24874–24890).

Zhang, C. L., Wu, J., & Li, Y. (2022). ActionFormer: Localizing moments of actions with transformers. In *European Conference on Computer Vision (ECCV)* (pp. 492–510). Springer. https://doi.org/10.1007/978-3-031-19772-7_29

Zhang, S., Peng, H., Fu, J., & Luo, J. (2020). Learning 2D temporal adjacent networks for moment localization with natural language. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(07), 12870–12877. https://doi.org/10.1609/aaai.v34i07.7053 (arXiv: https://arxiv.org/abs/1912.03590)

</div>

<br />

---

# ภาคผนวก (Appendices)

<br />

## ภาคผนวก ก: รายละเอียดชุดข้อมูลทดสอบมาตรฐานสากล (Benchmark Datasets)

| ชุดข้อมูลมาตรฐาน | ผู้พัฒนา / ปี | จำนวนคลิป / ชั่วโมง | ลักษณะคำค้นหา (Query Characteristics) | ตัวชี้วัดหลักที่ใช้ประเมิน |
| :--- | :--- | :---: | :--- | :--- |
| **QVHighlights** | Lei et al. (2021) | 10,148 วิดีโอ (~150 ชม.) | วิดีโอ Vlog และข่าวภาษาอังกฤษ พร้อมทั้ง Moment Intervals [t_start, t_end] และ 1-5 Saliency Score | R@1@IoU=0.5, 0.7, mIoU, HIT@1 |
| **Charades-STA** | Gao et al. (2017) | 9,848 ช่วงเวลา (~82 ชม.) | วิดีโอกิจกรรมภายในบ้าน เน้นการปฏิสัมพันธ์กับสิ่งของ (Object Interaction) | R@1@IoU=0.3, 0.5, 0.7, mIoU |
| **ActivityNet Captions** | Krishna et al. (2017) | 20,000 วิดีโอ (~849 ชม.) | กิจกรรมกลางแจ้งและการกีฬา มีคำบรรยายต่อเนื่องหลายประโยคต่อคลิป | R@1@IoU=0.3, 0.5, mIoU |

<br />

## ภาคผนวก ข: ข้อกำหนดฮาร์ดแวร์และการใช้หน่วยความจำ VRAM ของแต่ละแบบจำลอง

| แบบจำลอง AI | ฟังก์ชันการทำงาน | รูปแบบการ Quantization | ขนาดพารามิเตอร์ | Peak VRAM ขณะรัน | Latency เฉลี่ยต่อ 1 นาทีวิดีโอ |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Decord** | อ่านเฟรมวิดีโอตาม timeline | Video Decoder | - | วัดจาก benchmark | วัดจาก benchmark |
| **SigLIP 2 (NaFlex)** | สกัด frame/text embeddings | FP16/BF16 | ตาม checkpoint | วัดจาก benchmark | วัดจาก benchmark |
| **Qwen2.5-VL** | Dense scene caption และ Accurate verifier | 4-bit ตาม budget | ตาม checkpoint | ต้องไม่เกิน 8 GB รวมระบบ | วัด p95 จาก benchmark |
| **LanceDB Engine** | Vector search และ caption FTS | Columnar Disk Storage | - | วัดจาก benchmark | วัดจาก benchmark |

<br />

## ภาคผนวก ค: ตัวอย่างคำค้นหาภาษาธรรมชาติและผลการสกัดช่วงเวลา [t_start, t_end]

```text
ตัวอย่างที่ 1: วิดีโอบรรยายการสอน (ตัวอย่างรูปแบบผลลัพธ์ ไม่ใช่ผล benchmark)
  • คำค้นหา (Query): "ช่วงที่อาจารย์ชี้กราฟแท่งเปรียบเทียบผลลัพธ์โมเดล"
  • ผลลัพธ์: moments หลายรายการได้เมื่อผ่าน calibrated threshold
  • Breakdown: visual + caption + temporal + verifier

ตัวอย่างที่ 2: วิดีโอกล้องติดหน้ารถยนต์
  • คำค้นหา (Query): "รถจักรยานยนต์เลี้ยวตัดหน้าก่อนถึงทางแยก"
  • ผลลัพธ์: occurrence_index แยกเหตุการณ์ที่ไม่ทับกัน และคืน moments=[] หากต่ำกว่า threshold
  • Accurate mode: Qwen เลือกเฉพาะ timestamp ของเฟรมที่ส่งเข้าโมเดล
```

---

## ภาคผนวก: Current Implementation Status (main @ `6de9468`)

ภาคผนวกนี้แยกสถานะ implementation จริงออกจากทฤษฎี สมมติฐาน และ benchmark design ในรายงานฉบับเดิม

- **Production stack:** `SigLIP2 NaFlex` เป็น retrieval หลัก และ `CapRL-Qwen3VL-4B Q6` เป็นทั้ง precomputed scene captioner และ Accurate verifier
- **SAM/Qwen2B:** SAM 3.1 ปิดใน production ด้วย `ENABLE_SAM_GROUNDING=false`; Qwen3-VL-2B ไม่ถูกโหลดใน production flow และคงไว้เฉพาะ compatibility/ประวัติของระบบ
- **Hardware/runtime:** ทดสอบบน RTX 5070 12GB ด้วย physical VRAM budget 11.8GB และ Accurate budget 60 วินาที โดย CapRL Q6 ใช้ llama.cpp local worker; checkpoint GGUF และ mmproj ไม่อยู่ใน repository
- **Lifecycle:** หลัง Phase 1 ระบบเปิด Fast Search ได้ทันที จากนั้นสร้าง CapRL Q6 caption แบบ background, upsert แบบ resumable และ activate version แบบ atomic เมื่อ artifact ครบ
- **ความแตกต่างจากเนื้อหาหลัก:** Qwen2.5-VL, budget 7–8GB/15 วินาที และ flow ที่อธิบาย SAM/Qwen เป็นระบบปัจจุบันในส่วนเดิม ให้ตีความเป็น proposal baseline หรือ historical design เท่านั้น ไม่ใช่ production contract
- **สิ่งที่ยังต้องวัดใหม่:** R@K, temporal IoU/mIoU, verifier accuracy, latency, VRAM และ stability ต้องรายงานจาก video-disjoint held-out set; ตัวเลขในส่วนทฤษฎีหรือ smoke test ไม่ใช่ผล production ที่ยืนยันแล้ว

ดูรายละเอียด endpoint, configuration, caption status/rebuild, provenance ของ caption/verifier และ troubleshooting ได้ที่ [docs/current-runtime.md](docs/current-runtime.md)
