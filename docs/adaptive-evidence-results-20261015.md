# ผลทดลองเลือกเฟรมและคำสั่งตรวจหลักฐาน — 15 ตุลาคม 2026

**ยังไม่มีวิธีใหม่ชนะ baseline จึงคงวิธีเดิมสำหรับ demo**

## ผลหลักตามเกณฑ์เดิม

| วิธี | พบเหตุ | แจ้งผิด | Recall | Precision | Qwen wall | จำนวนคำขอ | VRAM รวมทั้งการ์ดสูงสุด | Qwen RAM สูงสุด |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| วิธีเดิม: เฟรมสม่ำเสมอ | 6/7 | 1 | 85.7% | 85.7% | 130.77s | 16 | 11.24 GiB | 9.57 GiB |
| เลือกเฟรมตามการเปลี่ยนแปลง | 6/7 | 1 | 85.7% | 85.7% | 130.13s | 16 | 11.43 GiB | 9.57 GiB |
| เฟรมเดิม + คำสั่งเน้นหลักฐาน | 6/7 | 3 | 85.7% | 66.7% | 131.87s | 16 | 11.30 GiB | 9.64 GiB |
| เฟรมใหม่ + คำสั่งเน้นหลักฐาน | 6/7 | 3 | 85.7% | 66.7% | 132.47s | 16 | 11.47 GiB | 9.63 GiB |

ไม่มี uncertain assessment หรือ technical error ในทั้งสี่แขน แต่ข้อความ uncertainty ภายในคำตอบบางรายการยังมีอยู่ ไม่ควรตีความว่าโมเดลมั่นใจถูกต้องทุกช่วง

## การควบคุมและต้นทุน

- ใช้ Gemma gate ที่ล็อกไว้เดิม 16 หน่วยจาก 10 validation videos, 7 labeled events ของ UBnormal ภาพจำลอง ไม่มีเสียง; normal references จาก development เดิม ไม่เปิด official test
- แต่ละหน่วยไม่เกิน 12s มีภาพในเหตุ 8 เฟรม บวกภาพบริบทและ normal references เดิม; baseline ใหม่และ adaptive รัน Qwen จริงในรอบนี้ ไม่ใช้คำตอบเก่ามาแทน
- เลือกภาพจาก candidate ประมาณ 4FPS เพิ่มความถี่เฉพาะช่วงท้ายสั้นเพื่อให้ครบอย่างน้อย 8 candidates; ผสมมวล uniform 50% กับ mean absolute grayscale image difference 50% แล้วเลือก quantiles แบบไม่ซ้ำ ไม่ใช้ label เวลา/actor มาเลือกภาพ
- เปลี่ยน timestamps 15/16 หน่วย อีกหนึ่งหน่วยช่วงสั้นใช้ครบทุก candidate; คงจำนวนภาพ ขนาดภาพ context prompt และ references เท่ากันในการเทียบ sampling
- แขน evidence เพิ่มคำสั่งให้บรรยายการเปลี่ยนแปลงที่เห็น เทียบกิจกรรมกับบริบทปกติและคำอธิบายทางเลือก แล้วตัดสินด้วย schema เดิม ข้อความเต็มเก็บใน protocol
- แต่ละการเปรียบเทียบสลับลำดับแขนทุก batch 4 หน่วย; สองแขน evidence รันต่อจากรอบ sampling จึงไม่ได้สุ่มทั้ง 4 แขนรวมกัน เวลาเป็นเพียงหนึ่งรอบต่อแขน
- การเตรียม candidate/decode/JPEG สำหรับ adaptive ใช้ 11.89s เพิ่มเติม ไม่รวมในคอลัมน์ Qwen wall; Gemma media index/gate และ baseline JPEG มีอยู่แล้ว ไม่ใช่เวลา end-to-end ตั้งแต่วิดีโอใหม่
- VRAM sampled ทั้งการ์ดรวมโปรแกรมอื่น; RAM เป็น process-tree RSS ของ Qwen ไม่ใช่ peak RAM preprocessing; Qwen ใช้ llama.cpp ไม่ใช้ค่า PyTorch allocated ของ broker มาอ้างเป็นทรัพยากร Qwen
- batch size รอบนี้เป็น 4 ต่างจากการทดลองก่อนหน้าที่เป็น 8 จึงห้ามตีความเวลาหรือ RAM ที่ต่างจากรายงานเก่าว่าเป็นผลดีของวิธีใหม่
- ตรวจ frame hashes และ script hashes หลังรันตรงเดิมทั้งสองรอบ; tests ที่เกี่ยวข้องผ่าน 37 ข้อ มี deprecation warnings เดิม 2 ข้อ

## วิเคราะห์ขอบเขตเหตุโดยไม่เปลี่ยนเกณฑ์กลางทาง

ใน abnormal_scene_4_scenario_7 หน่วยท้ายเปลี่ยนจาก normal ใน baseline เป็น suspicious ใน adaptive และทั้งสองแขน evidence แต่ช่วงต้นและท้ายติดกันจึงถูกรวมเป็นช่วงเดียวก่อนประเมิน ผลตาม event matching เดิมจึงยังเป็น 6/7 ไม่ใช่ 7/7
หากวิเคราะห์โดยไม่รวมหน่วย baseline ได้ TP=6/FP=4/FN=1 และ adaptive ได้ TP=7/FP=5/FN=0 นี่เป็นเพียง sensitivity diagnostic เพราะหน่วยติดกันอาจซ้ำเหตุเดียว ทำให้ FP เปลี่ยนด้วย ห้ามนำตัวเลข 7/7 นี้มาแทนผลหลักหรืออ้างว่าระบบแม่นขึ้นแล้ว
คำอธิบายของ adaptive ในหน่วยท้ายยังอ้างถึง medical attire และปฏิสัมพันธ์ที่ผิดธรรมดา จึงไม่ยืนยันว่าจับพฤติกรรมจริงตาม label ได้ถูกต้อง ต้องตรวจหลักฐานและขอบเขตแยกต่อ

## ข้อสรุปและรอบที่ควรทำต่อ

1. คง uniform sampling และ camera-normality policy เดิม: sampling ใหม่ไม่ลด FP หรือเพิ่ม event recall ตามเกณฑ์เดิม ส่วน evidence prompt เพิ่ม FP จาก 1 เป็น 3 และ precision ลด 85.7%→66.7%
2. แยกตรวจขอบเขตภายในกลุ่มเหตุที่ต่อกันด้วยข้อมูลภาพ/คะแนนจริง ไม่ใช้ label มาตัดช่วง และไม่แก้กฎ merge ให้เข้ากับคำตอบของคลิปนี้ ควรทดลอง refinement แบบมีบริบทเฉพาะกลุ่ม แทนการแบ่งทุกคลิปเป็น 3s ซึ่งงานก่อนหน้าเคยให้ผลด้อยกว่า
3. เพิ่มข้อมูล validation ที่กันไว้ใหม่และ normal exposure ยาวขึ้นก่อนเลือกวิธีใหม่ ชุดนี้ถูกตรวจซ้ำและใช้วิเคราะห์แล้ว มีเพียง 7 เหตุ/normal ประมาณหนึ่งนาที ไม่รองรับข้อสรุปทั่วไปหรืออัตราแจ้งผิดรายชั่วโมงที่เสถียร
4. Bootstrap ตาม recording บันทึกใน summary.json เพื่อแสดงความไม่แน่นอนในชุดเล็ก ไม่ใช่หลักฐาน generalization หรือ independent test

รอบนี้ไม่แก้ production defaults, demo jobs/reviews, model weights, thresholds หรือ main; ผลดิบเก็บแยกจาก Git ไม่รวมวิดีโอ/environment/weights ใน commit

## ทำซ้ำ

`python -m evaluation.embeddinggemma2.research_adaptive_frames --output <fresh-output>`

`python -m evaluation.embeddinggemma2.research_evidence_frames --source <completed-adaptive-output> --output <fresh-evidence-output>`

ผลดิบ: scratch/event-retrieval/adaptive-frames-20261015-v2 และ scratch/event-retrieval/evidence-frames-20261015-v1; v1 ของ sampling หยุดก่อนเรียก Qwen เพราะ candidate ช่วงท้ายไม่ครบ แก้แล้วรันใหม่ใน v2 ไม่ปะปนผลกัน

Graft CLI ไม่พร้อมใช้งาน จึง refresh graph ไม่ได้ใน environment นี้
