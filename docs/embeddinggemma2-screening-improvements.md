# วิธีปรับความแม่นของการคัดกรอง — 8 ตุลาคม 2026

สถานะ: ข้อเสนอจากการตรวจ source และทดลองคะแนนบน embeddings ที่มีอยู่ ยังไม่ได้พิสูจน์ว่า configuration ใหม่ผ่าน validation หรือเปลี่ยนระบบคัดกรองที่กำลังใช้งาน

## หลักฐานจากระบบเรา

ใช้ prompt เดิมและเวกเตอร์ของคลิปเต็ม โดยไม่ทำดัชนี media ใหม่ ผลดิบอยู่ที่ `scratch/event-retrieval/long-clips/screening-diagnosis.json`

| Fighting011 (92 ช่วง) | จำนวนช่วงผลัก/ชกที่ผ่านเกณฑ์ |
|---|---:|
| ภาพผ่าน แต่วิดีโอไม่ผ่าน | 2 |
| วิดีโอผ่าน แต่ภาพไม่ผ่าน | 8 |
| ทั้งสองช่องทางผ่าน | 76 |
| กฎปัจจุบัน: อย่างใดอย่างหนึ่งผ่าน | 86 |

Normal634 มี 150 ช่วง ไม่ผ่านทั้งสองช่องทางเลย แต่เป็นฉากถนนไกลจากกล้อง จึงไม่ใช่ hard negative ที่เหมาะกับคนอยู่ใกล้กันในห้อง

ใน Fighting011 คะแนน positive ภาพทุกช่วงอยู่ประมาณ 0.644–0.728 และวิดีโอ 0.675–0.757 ดังนั้น threshold=0.35 ไม่ช่วยแยกช่วงในคลิปนี้ ตัวที่แยกจริงส่วนใหญ่คือ margin=0.02 เทียบกับชุดคำอธิบายพฤติกรรมปกติที่ยังมีน้อย ไม่มีการนำตัวเลขนี้ไปตั้ง threshold ใหม่หรืออ้างเป็นความแม่น

Source ที่เกี่ยวข้อง: `backend/app/gemma2/ranking.py` (screen/merge_candidates), `backend/app/gemma2/service.py` (refine/verification_frames), `evaluation/embeddinggemma2/worker.py` (verify_batch)

## ลำดับปรับที่แนะนำ

1. **ตรวจเหตุในช่วงสั้นก่อนรวมเป็นเหตุใหญ่** ใช้หน่วยตรวจเริ่มต้น 6 วินาทีที่มีอยู่ ไม่เอาเหตุ 254 วินาทีไปตรวจด้วย 8 เฟรมชุดเดียว เก็บ parent candidate พร้อม child verification units และตรวจช่วงย่อย 3 วินาทีเพื่อหาขอบเขต รวมเฉพาะ child intervals ที่มีหลักฐานยืนยันจริง ไม่ลาก min(start) ถึง max(end) ครอบช่องว่างที่ไม่ผ่าน ยังคงเหตุที่ตรวจไม่สำเร็จเป็น pending และไม่ลบผล review เดิม

2. **เพิ่มพฤติกรรมปกติที่แยกยากและอยู่ในฉากเดียวกัน** เช่น คนยืน/นั่งใกล้กัน คนชี้มือ คนกอด/จับมือ เจ้าหน้าที่ประคองคน คนที่นอนอยู่แล้วโดยไม่ได้เกิดการล้ม สำหรับเสียงใช้พูดดัง หัวเราะ เพลง/ทีวี และเสียงกระแทกที่ไม่ใช่เสียงคนตะโกน ให้ผู้ตรวจยืนยัน label ของคลิปเหล่านี้ก่อนใช้ อย่าใช้ความดังเพียงอย่างเดียวเป็นหลักฐานความวุ่นวาย

3. **เรียนรู้ตัวคัดกรองขนาดเล็กจาก embeddings ที่ตรึงไว้** เปรียบเทียบ zero-shot prompt bank ที่ปรับแล้วกับ logistic regression/linear head ใช้ video embedding และคะแนนเทียบ hard negatives เป็นจุดเริ่มต้น แยกการเรียนรู้แต่ละเหตุและแต่ละช่องทาง โดยไม่ fine-tune น้ำหนัก Gemma ใช้ภาพเป็นข้อมูลประกอบ ไม่กำหนดว่า video ต้องผ่านเสมอจนกว่าจะตรวจ recall ของเหตุสั้นแล้ว ความเปลี่ยนแปลงตามเวลาอาจเพิ่มเป็น feature แต่ motion ไม่ใช่หลักฐานว่ามีการชกโดยตัวมันเอง

4. **หาขอบเขตจากคะแนนตามเวลา** ทดลองเกณฑ์เริ่มเหตุที่สูงกว่าเกณฑ์คงเหตุ (hysteresis) และแบ่งเมื่อหลักฐานลดลง แยก minimum duration ตามชนิดเหตุ ไม่บังคับให้ทุกเหตุผ่านหลาย window เพราะอาจทำให้พลาดการล้มหรือชกสั้น ๆ จำกัดความยาวหน่วยตรวจเพื่อคุมการสุ่มเฟรม ไม่ใช้เพดานความยาวนั้นอ้างว่าเหตุยาวเป็นหลายเหตุจริง

5. **ให้ Qwen ตรวจการกระทำก่อนสร้าง Caption ยาว** ใช้คำถามเจาะจงตามชนิดเหตุ พร้อม timestamps และเฟรมต่อเนื่อง: การล้มต้องเห็นการเปลี่ยนท่าลงพื้น; ผลัก/ชกต้องมีการเคลื่อนมือ/แขนเข้าหาอีกคนและหลักฐานการสัมผัสหรือการตอบสนอง พร้อม uncertain หากมองไม่เห็น เก็บ decision+evidence สั้นก่อนสร้าง Caption ของเหตุที่ผ่าน ลดการเสีย calls กับ candidate ปกติ ไม่ให้ Caption ที่สร้างเองเป็น ground truth

6. **แยกความครบของคำตอบออกจากความแม่นของการตรวจ** ใช้ JSON schema/grammar ถ้า binary ที่ใช้อยู่รองรับ, จำกัดจำนวนรายการและความยาวข้อความ, ตรวจ finish reason/token budget และ retry เฉพาะงานที่ไม่ครบหนึ่งครั้ง JSON ที่รูปแบบถูกต้องยังอาจบรรยายผิดได้ และ grammar ไม่รับประกันว่าคำตอบจะจบเมื่อ token budget หมด ห้ามเปลี่ยน parse failure เป็น event_present=false

## วิธีพิสูจน์

- ทำ onset/offset และ label ของชุด development/validation ให้ครบก่อน โดยมีคลิปปกติที่มีคนและบริบทใกล้เคียง แบ่งตามวิดีโอ/การถ่ายจริง
- รันเปรียบเทียบทีละปัจจัย: เดิม → แบ่งหน่วยตรวจ → hard negatives → classifier → temporal boundaries → Qwen decision-first ใช้เฟรมและ prompt ที่เกี่ยวข้องเหมือนกันสำหรับการเทียบแต่ละคู่
- รายงาน candidate recall ก่อน Qwen, precision/recall สุดท้ายที่ temporal IoU≥0.3, false alarms ต่อชั่วโมง, temporal boundary error, unverified rate และ actual calls/เวลา/VRAM
- เลือก threshold ต่อชนิดเหตุด้วย validation เท่านั้น แล้วล็อกก่อน test คงเป้าหมายเดิม: screening recall≥95%, final recall≥85%, precision≥80%, false alarms≤2/h ตัวเลขเป็นเป้าหมาย ไม่ใช่ผลที่ทำได้แล้ว
- คลิป Fighting011 ที่เปิดดูและใช้วิเคราะห์นี้เป็น development/example ต้องไม่กลับไปอ้างเป็น held-out test เพื่อเลือกค่าชนะ

## แหล่งวิจัยและขอบเขตการนำมาใช้

- [คู่มือ Google EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers): อธิบาย embeddings หลายช่องทางและ semantic similarity; ข้อเสนอของเราคือเพิ่มตัวตัดสินเหตุและเวลาแยกจาก retrieval การมี unified embeddings ไม่ยืนยันความแม่นของพฤติกรรมเฉพาะ
- [ActionFormer, ECCV 2022](https://arxiv.org/abs/2202.07925): แยกการจำแนกและหาขอบเขตการกระทำจากลำดับ features ตามเวลา เป็นแนวทางอ้างอิงสำหรับพัฒนาตัว temporal head ในระยะถัดไป ไม่ใช่หลักฐานว่าโมเดล pretrained ใช้ตรวจโรงพยาบาลได้ทันที
- [Post-Processing Temporal Action Detection, CVPR 2023](https://arxiv.org/abs/2211.14924): ศึกษาความคลาดเคลื่อนขอบเขตจาก temporal downsampling และการปรับขอบเขตหลังโมเดล แนว hysteresis/แบ่งหน่วยที่เสนอที่นี่เป็นการออกแบบของเรา ไม่ใช่การติดตั้ง GAP ของบทความนี้
- [llama.cpp grammar guide](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md): constrained generation/JSON schema รองรับเพียงบางส่วน และต้องตรวจความเข้ากันได้ของ binary ที่ pin ไว้ก่อนใช้

เริ่มจากข้อ 1, 2 และ 6 ก่อน แล้วใช้ชุด label ที่ครบประเมินข้อ 3–5 เพื่อเลือกค่าที่คุ้มกับทรัพยากร ไม่มีการรับประกันว่าการเพิ่มโมเดลหรือ vision budget เพียงอย่างเดียวจะแก้ false positives

## ข้อสรุปหลังค้นงานวิจัยเพิ่มเติม

ทางเลือกที่เหมาะกับระบบและข้อมูลที่มีตอนนี้ คือ **Gemma คัดช่วงที่อาจมีเหตุแยกตามพฤติกรรม + แบ่งหน่วยตามเวลา + Qwen ตรวจคำถามเจาะจงการกระทำ + รวมเฉพาะหน่วยที่ยืนยันแล้ว** นี่เป็นข้อเสนอการประยุกต์สำหรับโปรเจกต์ ไม่ใช่ข้อสรุปว่าดีที่สุดในทุก benchmark หรือมีผลรับรองการย้ายงานวิจัยมาสู่ Gemma/Qwen รุ่นเรา

| งานต้นทาง | หลักที่ใช้ประโยชน์ | ขอบเขต/ข้อจำกัดสำหรับเรา |
|---|---|---|
| [EventVAD](https://arxiv.org/html/2504.13092v1) | แบ่งเหตุและใช้ความเปลี่ยนแปลงของ features ตามเวลาก่อนให้ VLM ตัดสิน | รุ่นที่อ่านเป็น preprint ใช้ CLIP+RAFT+VideoLLaMA2.1-7B และรายงานเครื่อง A800 80 GB; ไม่คัดลอกทั้งชุดแล้วถือว่าจะรันบน 12 GB ได้ ใช้หลัก temporal segmentation กับ Gemma ที่มี และมี short-window fallback หากไม่มีจุดเปลี่ยนชัด |
| [VERA, CVPR 2025](https://arxiv.org/html/2412.01095v3) | ใช้คำถามนำที่อิงรูปแบบเหตุและข้อมูลติดป้ายกำกับ โดยไม่แก้น้ำหนัก VLM | ต้องมี coarsely labeled training data และกระบวนการปรับคำถาม; คำถามที่เรียนรู้กับ VLM หนึ่งไม่รับประกันผลกับ CapRL Qwen 4B เรา ต้องเลือกจาก validation ของเรา |
| [ASK-Hint](https://arxiv.org/html/2510.02155v1) | ใช้คำถามการกระทำที่เฉพาะและกระชับกับ frozen VLM | preprint ระบุว่ายังขาด temporal modeling จึงใช้ prompt เพียงอย่างเดียวไม่แก้ขอบเขตยาวในระบบเรา |
| [VadCLIP, AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28423) | ตรึง encoder แล้วฝึก temporal adapter/ตัวจำแนกจาก weak labels | ไม่ใช่วิธีไม่ต้องฝึกทั้งหมด มีโมดูลและ prompts ที่เรียนรู้; เป็นต้นแบบสำหรับ phase ถัดไปเมื่อข้อมูลพอ ต้องฝึก/ทดสอบ head บน Gemma features ใหม่ ไม่ใช้ CLIP weights ของ adapter โดยตรง |
| [Holmes-VAU, CVPR 2025](https://github.com/pipixin321/HolmesVAU) | เลือกเฟรมที่มีหลักฐานเหตุพร้อมบริบทด้วย anomaly-focused sampling | ชุดเต็มมี trained scorer และการ fine-tune โมเดล; ใช้เป็น reference การสุ่มเฟรม ไม่อ้างว่าแทนที่ด้วย uniform 8 frames แล้วได้ผลเดียวกัน |

### ขั้นตอนที่เลือกสำหรับรอบถัดไป

1. เปลี่ยนจาก merge-before-verification เป็น bounded child units (เริ่ม 6 วินาที และทดสอบบริบทที่ยาวขึ้นอย่างควบคุม) พร้อม status แยกแต่ละหน่วย ใช้ features ตามเวลาช่วยแบ่งและหาขอบเขต แต่การเปลี่ยน features ไม่ใช่เหตุผิดปกติโดยตัวมันเอง และอย่าใช้เป็น hard gate ที่ทำให้พลาดเหตุช้า/ถูกบัง
2. Qwen ตรวจ 2–4 คำถามต่อชนิดเหตุภายในคำตอบสั้นชุดเดียว เช่น แขนเคลื่อนเข้าหาบุคคลอื่นไหม เห็นการสัมผัส/ตอบสนองไหม หรืออาจเป็นการประคอง; สำหรับการล้มต้องเห็นการเปลี่ยนจากท่าตั้งตรงลงพื้น เก็บ evidence frames/timestamps และ uncertain ห้ามกำหนดว่าต้องตอบ true ทุกคำถามเมื่อหลักฐานถูกบัง
3. เมื่อยืนยันการกระทำจึงสร้าง Caption รายละเอียดแยก การเปลี่ยนลำดับ decision-first เป็นการออกแบบของเราเพื่อควบคุมงาน Qwen ไม่ใช่ pipeline ต้นฉบับ EventVAD ซึ่งใช้ description ก่อน score ใช้ JSON grammar ถ้ารองรับร่วมกับ token budget/retry และรักษา raw failed outputs สำหรับตรวจย้อนหลัง
4. เพิ่ม hard negatives ที่มีคนในฉากใกล้เคียงและปรับ threshold ต่อเหตุบน validation ตรึง Gemma; เมื่อข้อมูลเพียงพอเทียบ linear head baseline กับ temporal head ขนาดเล็ก โดยนับหน่วยที่ไม่ยืนยันเป็น unverified และรายงาน recall/precision/false alarms/time ตามเดิม

งานวิจัยที่อ้างไม่ทดลอง EmbeddingGemma 2 + CapRL Qwen3VL-4B-Q6 ของเราโดยตรง และส่วนใหญ่รายงาน frame-level AUC/AP ซึ่งไม่ใช่ final event Precision/Recall หรือ false alarms/h ห้ามนำคะแนนจากบทความมาอ้างเป็นผลของระบบเรา เลือกวิธีจาก validation โดยรักษา recall ก่อน แล้วลด false positives และต้นทุนภายในเกณฑ์ที่ตั้งไว้
