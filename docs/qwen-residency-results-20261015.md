# ผลทดลองลดการโหลด Qwen ซ้ำ — 15 ตุลาคม 2026

**เวลาเฉลี่ยลด 3.9% โดยคำตอบรายช่วงตรงกันทุกตัวอักษรในสองรอบที่ทดลอง**

## ผลที่วัดได้

| รอบ | วิธี | เวลา Qwen phase | โหลดโมเดล | พบเหตุ | แจ้งผิด | Peak VRAM ทั้งการ์ด | Peak Qwen RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | ปล่อยทุก batch | 123.68s | 2 | 6/7 | 1 | 11.41 GiB | 12.67 GiB |
| 1 | เก็บไว้จนจบ phase | 117.16s | 1 | 6/7 | 1 | 11.01 GiB | 12.67 GiB |
| 2 | เก็บไว้จนจบ phase | 119.10s | 1 | 6/7 | 1 | 11.06 GiB | 12.67 GiB |
| 2 | ปล่อยทุก batch | 122.23s | 2 | 6/7 | 1 | 11.04 GiB | 12.61 GiB |

เฉลี่ย baseline 122.96s → retained 118.13s ประหยัด 4.83s หรือ 3.93%; รายรอบประหยัด 5.27% และ 2.56% มีเพียงสองคู่ ยังไม่ใช่ผล speedup ที่ยืนยันในทุก workload

## สิ่งที่คงเดิม

- ใช้ frozen uniform payload 16 หน่วยและ JPEG เดิมจาก adaptive-frames-20261015-v2; โมเดล Q6, prompt, timestamps, normal references, max_tokens และลำดับเหตุเหมือนกัน
- batch ละ8 เหมือน production; baseline โหลดสองครั้งสำหรับสอง batch ส่วน retained โหลดครั้งเดียวตลอด16หน่วย ไม่รวมหลายเหตุให้โมเดลตัดสินในคำขอเดียว และไม่ใช้ verdict cache
- รัน Qwen จริงรวม64คำขอ: รอบ1 baseline→retained; รอบ2 retained→baseline บังคับ unloaded ก่อนเริ่มแต่ละแขน และตรวจ unloaded หลังจบแต่ละแขน
- ใช้ worker.verify_batch และ checked_verdict เดิมผ่าน isolated in-process module; เปลี่ยนเฉพาะ release hook ของ subprocess ทดลอง ไม่แก้ running broker/API หรือ production defaults
- ก่อนแต่ละ batch ตรวจว่าไม่มี demo job กำลังทำงาน ไม่มี Gemma resident และมี RAM เหลือ; ส่วนนี้เป็น offline experiment ไม่ใช่ production GPU lease manager และยังไม่ควรใช้กับการทำงานหลาย client พร้อมกัน
- เวลารวม cold model load ทั้งสอง batch, checks/telemetry และ final release ของแต่ละแขน; ไม่รวมการทำ media index/Gemma gate/เตรียม JPEG ที่มีอยู่แล้ว การบังคับ unload ก่อนเริ่มแต่ละแขนอยู่นอกช่วงจับเวลา

## คุณภาพและสถานะ

- เปรียบเทียบแบบจับคู่32หน่วย (16หน่วยเดิมในสองรอบ): assessment changes=0 และ raw JSON text changes=0 ทั้งสองคู่ จึงคงทั้ง verdict/reason/evidence ในข้อมูลที่ทดลอง ไม่ใช่เพียงคะแนนรวมเท่ากัน
- ผล event matching ตามกติกาเดิม: TP6/FP1/FN1, Recall85.7%, Precision85.7%, uncertain0 ทั้งสี่ run ไม่มี technical error
- code hashes และ frame hashes หลังรันไม่เปลี่ยน; Qwen ถูก unload ครบทั้งสี่ run และ cleanup หลังทดลองทำงานแล้ว
- Tests ผ่าน40ข้อ รวมกรณี defer releases, คืน release hook และปล่อยโมเดลเมื่อมี exception; มี deprecation warnings เดิม2ข้อ ไม่ใช่การทดสอบ production lease/cancel ทั้งระบบ

## ข้อจำกัดและการนำไปใช้

1. ชุดนี้เป็น UBnormal ภาพจำลอง validation ที่ตรวจซ้ำแล้วเพียง7เหตุ ไม่มีเสียง และ normal exposure60.13วินาที ผลตรงกันในชุดนี้ไม่รับประกันความแม่นทุกวิดีโอ
2. RAM สูงสุดใกล้เดิมประมาณ12.7GiB แต่ถือหน่วยความจำไว้นานกว่า ส่วน VRAM ที่วัดเป็นทั้งการ์ดรวมโปรแกรมอื่น จึงไม่อ้างว่าลด VRAM จากค่าที่ต่างเล็กน้อย
3. ได้ประโยชน์เมื่อมีมากกว่าหนึ่ง batch หรือมี retry ถ้างานไม่เกิน8หน่วยและไม่มี retry จะไม่มีการโหลดข้าม batch ให้ตัดออก จึงไม่คาดว่าจะได้ speedupเท่ารอบนี้
4. ยังไม่เปิดใช้ใน demo การใช้งานจริงต้องเพิ่ม phase/job lease ใน broker ที่มี owner, timeout, cancellation/error cleanup และส่งมอบ GPU ให้ Gemma ผ่านคิวเดิม ไม่ใช้ no-op release แบบใน subprocess ทดลอง
5. ไม่มีการลดภาพ/ความละเอียด เปลี่ยน quantization หรือเพิ่มเกณฑ์ Gemma เพื่อแลกความเร็ว จุดถัดไปสำหรับ first-pass throughput ยังเป็นเวลาประมวลผลภาพ/คำสั่ง ไม่ใช่ lifecycle อย่างเดียว

## ทำซ้ำ

`python -m evaluation.embeddinggemma2.research_qwen_residency --output <fresh-output> --repeats 2`

ผลดิบอยู่ scratch/event-retrieval/qwen-residency-20261015-v1 พร้อม protocol, partial และ results; รายงาน/CSV/JSON สำเนาอยู่ output/qwen-residency-20261015 ของ checkout หลัก ไม่เก็บวิดีโอ/weights/environment/output ใน Git

Graft CLI ไม่พร้อมใน environment นี้ จึงใช้ข้อมูล source ที่ตรงตำแหน่งเกี่ยวข้องและไม่สามารถ refresh graph ได้
