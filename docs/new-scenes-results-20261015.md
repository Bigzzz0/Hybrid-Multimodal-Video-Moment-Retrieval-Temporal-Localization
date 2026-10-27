# ผลทดสอบฉากใหม่ — UBnormal

คงวิธีและเกณฑ์เดิมก่อนเปิดผลชุดใหม่ ไม่ฝึกน้ำหนักและไม่เลือก threshold ใหม่

## ข้อมูล

- วิดีโอใหม่ 20 คลิป จากฉาก 6/8/14/16; training-normal 8 คลิป (7 การบันทึกต้นฉบับ) สำหรับสร้างคลังปกติ
- ประเมิน official validation 12 คลิป มี 6 ช่วงเหตุและ 6 คลิปปกติ รวม 174.90s; normal-only 90.53s
- เป็นภาพจำลองของผู้จัดทำ UBnormal ไม่ใช่กล้องจริง ไม่มีการเปิด official test; แยกตาม recording โดย fog variant ไม่ถูกแยกข้ามชุด
- ฉาก8 มี training original และ fog ของการบันทึกเดียวกัน จึงไม่ใช่ตัวอย่างปกติอิสระสองการบันทึก; context เป็น generic pedestrian-area เดิม คลังอ้างอิงแยกฉาก
- UCSD แหล่งทางการเชื่อมต่อหมดเวลา ส่วน ShanghaiTech ได้หน้า OneDrive แต่ยังไม่มีไฟล์ที่ตรวจสอบได้ จึงไม่มีผลกล้องจริงที่อ้างจากสองชุดนั้นในรายงานนี้
- แหล่งข้อมูลและสิทธิ์: [UBnormal official repository](https://github.com/lilygeorgescu/UBnormal), CC-BY-NC-ND-4.0; source revision และ SHA256 บันทึกใน manifest/selection และตรวจ CRC ขณะอ่านไฟล์จาก archive

## ผลหลัก (temporal IoU ≥0.3, one-to-one matching)

| ขั้นตอน | TP | FP | FN | Recall | Precision | Uncertain units |
|---|---:|---:|---:|---:|---:|---:|
| Gemma: หน้าต่างต้นทาง | 6 | 6 | 0 | 100.0% | 50.0% | 0 |
| Gemma: หน่วยตรวจ 12s | 6 | 6 | 0 | 100.0% | 50.0% | 0 |
| Gemma → Qwen | 3 | 2 | 3 | 50.0% | 60.0% | 0 |
| ตรวจช่วงที่ข้ามเพิ่ม (diagnostic) | 3 | 2 | 3 | 50.0% | 60.0% | 0 |

ส่งเข้า Qwen 22/23 หน่วย ลดคำขอรอบหลัก 4.3%; ตรวจหน่วยที่ถูกข้ามแยกภายหลังเพื่อวิเคราะห์ Gemma เท่านั้น ไม่ปะปนเวลา audit กับรอบหลัก

## แยกตามฉาก (วิธีหลัก Gemma → Qwen)

| ฉาก | TP | FP | FN | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| UBnormal-14 | 1 | 0 | 0 | 100.0% | 100.0% |
| UBnormal-16 | 0 | 2 | 1 | 0.0% | 0.0% |
| UBnormal-6 | 1 | 0 | 1 | 50.0% | 100.0% |
| UBnormal-8 | 1 | 0 | 1 | 50.0% | 100.0% |

## กรณีที่ผิดและข้อสรุป

- เหตุที่พลาดทั้งสามคลิป (abnormal_scene_6_scenario_6, abnormal_scene_8_scenario_9, abnormal_scene_16_scenario_1) ถูกส่งถึง Qwen แล้ว และ Qwen ตอบ normal ทุกหน่วยที่ตรวจ ไม่ใช่กรณีพบเหตุแล้วเสียจากการ union อย่างในการทดลองก่อน
- normal_scene_16_scenario_2 แจ้งผิดโดยอ้างหมวก/หนวดและลักษณะคนงาน ส่วน normal_scene_16_scenario_3 อ้างชุดสีแดงคล้าย superhero แสดงว่าคำสั่งห้ามใช้รูปลักษณ์อย่างเดียวยังไม่ทำให้โมเดลปฏิบัติตามได้สม่ำเสมอ
- การตรวจช่วงที่ Gemma ข้ามเพิ่มให้ผลรวมเท่าเดิม จึงยังไม่พบเหตุหายเพิ่มจาก gate ในชุดนี้ แต่ gate ยังลดงานได้น้อยมาก
- ผล85.7%จากชุดเดิมจึงไม่ควรนำไปกล่าวเป็นความแม่นทั่วไป ชุดใหม่ได้ Recall50%/Precision60%; ยังต้องปรับการเห็นและตีความการกระทำของ Qwen พร้อมทดสอบข้อมูลที่กันไว้ใหม่ การเปลี่ยน threshold ให้เข้ากับชุดนี้ภายหลังจะไม่ถือเป็นผลทดสอบอิสระ

## เวลาและทรัพยากร

- สร้างดัชนีครั้งแรกใน process นี้ 500.42s; ผลรวม decode ที่ jobs บันทึก 348.62s และ encode 142.00s
- สร้างคลัง/encode query/คัดช่วง/เตรียมภาพตรวจ 80.82s
- Qwen รอบหลัก 157.08s; audit ช่วงที่ข้ามเพิ่ม 10.96s (แยกออกจากวิธีหลัก)
- เวลาไม่รวมดาวน์โหลดข้อมูล; ใช้ phase retention แบบ offline ที่ผ่านการทดลองก่อน ไม่ใช่การเปลี่ยนค่าเริ่มต้นใน demo
- Qwen phase peak VRAM ทั้งการ์ด 11.38GiB; peak Qwen process-tree RSS 12.65GiB (sample ทุก0.5s, VRAM รวมโปรแกรมอื่น)
- Technical error ในรอบหลัก 0 หน่วย; uncertain แยกจาก normal ไม่แปลง OOM/ตรวจไม่สำเร็จเป็นปกติ

## ข้อจำกัด

- ชุดใหม่ไม่ซ้ำฉากและไฟล์กับชุดเก่า แต่ยังเป็นภาพจำลองจาก dataset เดียวกัน และมีจำนวนน้อย ยังไม่ใช่หลักฐานว่าผ่านเป้าหมายใช้งานจริง
- normal exposure ประมาณ1.5นาที ไม่เพียงพอรับรองเกณฑ์แจ้งผิด≤2เหตุต่อชั่วโมง; ค่า bootstrap ที่เก็บในผลดิบเป็นความไม่แน่นอนของชุดเล็ก ไม่ใช่การรับรอง generalization
- ผลหลักใช้ first-pass Qwen ตาม controlled baseline เดิม เก็บ uncertain ไว้ ไม่เพิ่มวิธี retry/threshold ใหม่หลังเห็นผล
- all_qwen_diagnostic รวมคำตอบจริงรอบหลักกับคำตอบของหน่วยที่ข้ามภายหลัง ไม่ใช่การรัน all-unit อีกครั้งที่ควบคุมลำดับและใช้เทียบเวลา
- ประเมินช่วงเหตุจากขอบเขตหน่วยเดิมก่อน union; temporal matching และความต่อเนื่องของเหตุมีผลต่อ TP/FP ห้ามตีความเป็นการยืนยันชนิดการกระทำทุกกรณี
- ไม่เปลี่ยน demo jobs/reviews, model weights, thresholds หรือ merge main

## การทำซ้ำ

`python -m evaluation.embeddinggemma2.evaluate_frozen_scenes --manifest <manifest> --selection-lock <pre-download-lock> --output <fresh-run>`

ผลดิบ/manifest/protocol/คะแนน/คำตอบรายช่วงแนบใน output และใน scratch ของ worktree; วิดีโอ/ดัชนี/weights ไม่เข้า Git
