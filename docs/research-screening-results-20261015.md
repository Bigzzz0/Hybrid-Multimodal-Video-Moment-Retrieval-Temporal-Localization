# ผลทดลองวิธีตรวจความผิดปกติจากงานวิจัย — 15 ตุลาคม 2026

รันจริงบน RTX 5070 ในบร้านช์ทดลอง ใช้ UBnormal ภาพจำลอง 16 คลิป 3 ฉาก รวม 210.93 วินาที
development ปกติ 6 คลิป; validation 10 คลิป มีเหตุ 7 ช่วง และช่วงปกติรวม 60.13 วินาที ไม่มีเสียง
ไม่เปิด official test ไม่ฝึกน้ำหนัก ไม่เปลี่ยน demo defaults และไม่ merge main

## ข้อสรุป

คัดกรองเมื่อกันฉากออกจาก threshold: gemma_normal_knn_smooth1 OR vpc_temporal_smooth1 จับได้ 6/7, FP 4; คัดเวลาวิดีโอ 82.4%
ผล Qwen ที่สังเกตดีที่สุดระหว่างสองรอบ: Qwen camera-normality 12s (historical) จับได้ 6/7, FP 1, uncertain/unverified 0 หน่วย
สองบรรทัดนี้เป็นคนละ protocol ห้ามนำมาอ้างว่าเป็นผล Gemma/VPC → Qwen แบบ end-to-end ที่รันแล้ว
ไม่พบหลักฐานพอให้เปลี่ยน demo defaults หรืออ้างว่าผ่านเป้าหมายใช้งานจริง

## วิธีเปรียบเทียบ

วิธีคัดกรองใหม่กันหนึ่งฉากไว้ประเมิน หมุนครบ 3 ฉาก เลือก threshold จาก validation อีกสองฉากเท่านั้น
คลังปกติของแต่ละฉากมาจาก development ของฉากนั้น ไม่ใช้ validation เติมคลัง
เลือก threshold ให้ recall >=95% ก่อน จากนั้น recall สูงกว่า, FP ต่ำกว่า และระยะเวลาที่คัดน้อยกว่า
ใช้ IoU >=0.3 จับคู่เหตุหนึ่งต่อหนึ่ง ไม่ใช้ frame AUC แทน event recall
ยังเป็น exploratory validation: ชุดนี้เคยถูกวิเคราะห์มาก่อน และเลือกวิธีจากผล pooled validation; ไม่ใช่ independent test

## ผลทั้งหมด

| วิธี | Protocol | TP/7 | FP | Recall | Precision | รอตรวจ | เวลาวิดีโอที่คัด |
|---|---|---:|---:|---:|---:|---:|---:|
| semantic_margin_smooth0 | cross-scene | 5 | 11 | 71.4% | 31.2% | 0 | 77.7% |
| semantic_margin_smooth1 | cross-scene | 4 | 3 | 57.1% | 57.1% | 0 | 60.0% |
| flashback_small_bank_smooth0 | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 57.9% |
| flashback_small_bank_smooth1 | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 57.9% |
| prism_raw_axes_smooth0 | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 57.9% |
| prism_raw_axes_smooth1 | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 57.9% |
| prism_whitened_axes_smooth0 | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 57.9% |
| prism_whitened_axes_smooth1 | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 57.9% |
| gemma_normal_knn_smooth0 | cross-scene | 5 | 2 | 71.4% | 71.4% | 0 | 33.3% |
| gemma_normal_knn_smooth1 | cross-scene | 5 | 2 | 71.4% | 71.4% | 0 | 31.9% |
| vpc_spatial_smooth0 | cross-scene | 4 | 7 | 57.1% | 36.4% | 0 | 71.9% |
| vpc_spatial_smooth1 | cross-scene | 4 | 5 | 57.1% | 44.4% | 0 | 64.9% |
| vpc_temporal_smooth0 | cross-scene | 5 | 7 | 71.4% | 41.7% | 0 | 95.0% |
| vpc_temporal_smooth1 | cross-scene | 5 | 5 | 71.4% | 50.0% | 0 | 81.7% |
| vpc_global_smooth0 | cross-scene | 5 | 7 | 71.4% | 41.7% | 0 | 76.8% |
| vpc_global_smooth1 | cross-scene | 5 | 5 | 71.4% | 50.0% | 0 | 70.5% |
| vpc_three_memory_smooth0 | cross-scene | 5 | 7 | 71.4% | 41.7% | 0 | 76.8% |
| vpc_three_memory_smooth1 | cross-scene | 5 | 5 | 71.4% | 50.0% | 0 | 70.5% |
| gemma_normal_knn_smooth1 OR vpc_temporal_smooth1 | cross-scene | 6 | 4 | 85.7% | 60.0% | 0 | 82.4% |
| gemma_normal_knn_smooth1 OR prism_whitened_axes_smooth1 | cross-scene | 5 | 3 | 71.4% | 62.5% | 0 | 64.8% |
| gemma_normal_knn_smooth1 OR semantic_margin_smooth0 | cross-scene | 5 | 11 | 71.4% | 31.2% | 0 | 77.7% |
| vpc_three_memory_smooth1 OR prism_whitened_axes_smooth1 | cross-scene | 6 | 4 | 85.7% | 60.0% | 0 | 92.3% |
| original_gemma_cross_scene | cross-scene | 4 | 2 | 57.1% | 66.7% | 0 | 49.5% |
| vpc_loro_spatial_smooth0 | cross-scene | 4 | 7 | 57.1% | 36.4% | 0 | 76.8% |
| vpc_loro_spatial_smooth1 | cross-scene | 4 | 4 | 57.1% | 50.0% | 0 | 73.3% |
| vpc_loro_temporal_smooth0 | cross-scene | 5 | 7 | 71.4% | 41.7% | 0 | 95.0% |
| vpc_loro_temporal_smooth1 | cross-scene | 5 | 5 | 71.4% | 50.0% | 0 | 81.7% |
| vpc_loro_global_smooth0 | cross-scene | 5 | 2 | 71.4% | 71.4% | 0 | 64.1% |
| vpc_loro_global_smooth1 | cross-scene | 5 | 2 | 71.4% | 71.4% | 0 | 64.8% |
| vpc_loro_three_memory_smooth0 | cross-scene | 5 | 7 | 71.4% | 41.7% | 0 | 82.4% |
| vpc_loro_three_memory_smooth1 | cross-scene | 5 | 5 | 71.4% | 50.0% | 0 | 81.7% |
| Qwen camera-normality 12s (historical) | historical validation | 6 | 1 | 85.7% | 85.7% | 0 | 62.2% |
| Qwen explicit-target 3s | new full validation | 4 | 1 | 57.1% | 80.0% | 19 | 15.7% |
| Qwen explicit-target 3s + one schema retry | new full validation | 5 | 2 | 71.4% | 71.4% | 1 | 19.9% |

## Replay การคัดก่อน Qwen 3s ก่อน schema retry

ใช้คำตอบ Qwen ที่รันจริงบนกริดเดียวกัน เลือกหน่วยที่ทับกับช่วงคัดโดยไม่ใช้ label เปลี่ยนเฟรมหรือคำสั่ง
ตาราง replay นี้ใช้ผลรอบก่อน schema retry ซึ่งมี uncertain/unverified 19 หน่วย ไม่ใช่ผล pipeline ที่แก้ schema แล้ว
ยังไม่ได้วัดเวลา pipeline ที่คัดก่อนจริง จำนวนด้านล่างคือจำนวนหน่วยที่น่าจะต้องตรวจ ไม่ใช่ calls/เวลาที่ประหยัดแล้ว

| ตัวคัดก่อน Qwen 3s | TP/7 | FP | uncertain | หน่วยที่ต้องตรวจโดยประมาณ /57 |
|---|---:|---:|---:|---:|
| semantic_margin_smooth0 | 4 | 1 | 16 | 49 |
| semantic_margin_smooth1 | 3 | 1 | 10 | 34 |
| flashback_small_bank_smooth0 | 3 | 0 | 10 | 33 |
| flashback_small_bank_smooth1 | 3 | 0 | 10 | 33 |
| prism_raw_axes_smooth0 | 3 | 0 | 10 | 33 |
| prism_raw_axes_smooth1 | 3 | 0 | 10 | 33 |
| prism_whitened_axes_smooth0 | 3 | 0 | 10 | 33 |
| prism_whitened_axes_smooth1 | 3 | 0 | 10 | 33 |
| gemma_normal_knn_smooth0 | 3 | 1 | 4 | 19 |
| gemma_normal_knn_smooth1 | 3 | 1 | 4 | 20 |
| vpc_spatial_smooth0 | 3 | 1 | 13 | 44 |
| vpc_spatial_smooth1 | 3 | 1 | 10 | 39 |
| vpc_temporal_smooth0 | 4 | 1 | 19 | 56 |
| vpc_temporal_smooth1 | 4 | 1 | 13 | 47 |
| vpc_global_smooth0 | 3 | 1 | 12 | 44 |
| vpc_global_smooth1 | 3 | 1 | 9 | 40 |
| vpc_three_memory_smooth0 | 3 | 1 | 12 | 44 |
| vpc_three_memory_smooth1 | 3 | 1 | 9 | 40 |
| gemma_normal_knn_smooth1 OR vpc_temporal_smooth1 | 4 | 1 | 13 | 47 |
| gemma_normal_knn_smooth1 OR prism_whitened_axes_smooth1 | 3 | 1 | 10 | 37 |
| gemma_normal_knn_smooth1 OR semantic_margin_smooth0 | 4 | 1 | 16 | 49 |
| vpc_three_memory_smooth1 OR prism_whitened_axes_smooth1 | 4 | 1 | 16 | 53 |
| original_gemma_cross_scene | 3 | 0 | 8 | 28 |
| vpc_loro_spatial_smooth0 | 3 | 1 | 16 | 47 |
| vpc_loro_spatial_smooth1 | 3 | 1 | 15 | 42 |
| vpc_loro_temporal_smooth0 | 4 | 1 | 19 | 56 |
| vpc_loro_temporal_smooth1 | 4 | 1 | 13 | 47 |
| vpc_loro_global_smooth0 | 3 | 0 | 10 | 36 |
| vpc_loro_global_smooth1 | 3 | 0 | 11 | 37 |
| vpc_loro_three_memory_smooth0 | 4 | 1 | 13 | 48 |
| vpc_loro_three_memory_smooth1 | 4 | 1 | 13 | 47 |

## จุดที่ต่างจากงานต้นฉบับ

- PRISM: ใช้สูตร text covariance/whitened anomaly axes จาก source ที่ pin revision แต่ใช้ Gemma แทน Qwen/PE พร้อม generic text bank 24 ข้อ และ sampling ของเรา
- Flashback-style: ใช้ top-10 retrieval, anomaly penalty .95 และ temperature .01 บน text bank เล็ก ไม่ใช่การทำซ้ำคลังหนึ่งล้านคำอธิบายของบทความ
- VideoPatchCore: ใช้ RN101 และ feature/coreset functions จากต้นฉบับจริง แต่แทน YOLOv5l ด้วย SSDLite CPU, ใช้ 2FPS/4เฟรม/stride1s, memory cap512 และ normalize จาก development
- รอบ VPC แรกพบ in-sample normal distances เป็นศูนย์ใน memory ที่เก็บครบ จึงเก็บผลเดิมและทำรอบ LORO แยก ไม่อ้างว่า scaling เดิมมีความน่าเชื่อถือ
- LORO ใช้ normal recording หนึ่งตรวจเทียบอีก recording เพื่อประมาณความแปรปรวน; ช่องที่ไม่มีความแปรปรวนถูกปิด ไม่ขยายคะแนนด้วยการหาร 1e-6
- Qwen 3s เปลี่ยนทั้งความยาวหน่วยและคำสั่งระบุ target interval ชัดเจน จึงไม่อ้างว่าพิสูจน์ผลของ sampling เพียงปัจจัยเดียว
- ยังไม่ได้รัน official AnomalyRuler แบบ induction/deduction เต็ม, Flashback เต็ม หรือ pretrained VadCLIP; ห้ามใช้ผล adaptations ตัดสินว่างานต้นฉบับแพ้

## เวลาและทรัพยากร

VPC feature extraction: 48.68s, load 3.74s; อ่าน JPEG ที่เคยเตรียมไว้ ไม่รวม decode MP4/ดาวน์โหลดน้ำหนัก
VPC PyTorch allocated peak 0.509 GiB, reserved 0.572 GiB
VPC sampled process RSS 1.549 GiB; sampled whole-card VRAM 2.620 GiB
Qwen 3s: 57 private HTTP requests; elapsed 723.15s หลังเตรียมเฟรม รวมโหลด/ปล่อยในแต่ละ batch
Gemma media embeddings ใช้ cache จากรอบเดิม เวลา scoring ในผลใหม่จึงไม่ใช่ end-to-end indexing time
VRAM ทั้งการ์ดรวมโปรแกรมอื่น วัดทุก .5s; Qwen ใช้ llama.cpp ไม่ใช่ PyTorch allocator ของ broker
ไม่เปรียบเทียบ inference-only กับ cold indexing แล้วอ้าง speedup

## ข้อจำกัดและหลักฐาน

Schema retry: รันซ้ำเฉพาะ technical unverified โดยเพิ่ม valid index range, ใช้ภาพและ parser เดิม; เพิ่ม 18 calls, 216.79s; รวม 75 calls
ข้อมูลยังมีเพียง 7 เหตุและ normal exposure ประมาณหนึ่งนาที ยังรับรอง <=2 false alerts/hour ไม่ได้
คำตอบ Qwen ที่ผิด schema/เลขเฟรม หรือไม่ชัดเจน คง uncertain แยกจากผลปกติ ไม่ถือเป็นการยืนยันถูก
รายการผิดพลาด Qwen: {"Qwen action evidence requires valid chronological frame indices": 18}
ตรวจ hash ระหว่างรัน Qwen: code_unchanged=True, frames_unchanged=True
เก็บ predictions รายคลิป, fold thresholds, raw scores, source/frame/manifest hashes และคำตอบ Qwen ครบใน scratch/event-retrieval/
ชุดทดสอบที่เกี่ยวข้องผ่าน 28 ข้อ (หลังเปลี่ยน pytest basetemp เป็น workspace เนื่องจาก Temp ของ Windows ไม่มีสิทธิ์)

## ทำซ้ำ

```powershell
$env:PYTHONPATH="$PWD/backend;$PWD"
python -m evaluation.embeddinggemma2.research_vpc --output <fresh-vpc-dir>
python -m evaluation.embeddinggemma2.research_screening --output <fresh-results-dir> --vpc-features <vpc-dir>
python -m evaluation.embeddinggemma2.research_reference_calibration
python -m evaluation.embeddinggemma2.research_fusion
python -m evaluation.embeddinggemma2.research_qwen_units --output <fresh-qwen-dir>
```

fusion/LORO/report scripts อ้างอิงผลของรอบนี้ตามชื่อ directory ที่ระบุใน source; ใช้ fresh output เพื่อเก็บประวัติ

## แหล่งต้นฉบับ

- https://github.com/SkiddieAhn/Paper-VideoPatchCore (eefdc8d7a84f8eb06a561e3c42ab26a1b944f9b6)
- https://github.com/ytC2026/ICML2026-PRISM (27c4533a45670f3bceddcf875de39a23dd09bc65)
- https://arxiv.org/abs/2505.15205
- https://github.com/openai/CLIP (d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)
- https://github.com/lilygeorgescu/UBnormal
