# VadCLIP เทียบระบบปัจจุบัน — ทดลองข้ามชุดข้อมูล

## ข้อมูลและวิธี

- ใช้ official UCF-Crime checkpoint จาก VadCLIP ไม่ฝึกน้ำหนักเพิ่ม; strict load ครบทุก parameter รวม CLIP ViT-B/16
- Official source revision `c41067f07d252efcda18008bea367886070c33b0`; checkpoint SHA256 `d064aba05190ce5f6652151e6259b24ca6dda83a1e6289bf343afb6f905a2c79`
- ตัวตรวจฝึกจาก UCF-Crime แต่ประเมินกับ UBnormal ภาพจำลอง จึงไม่ใช่การทำซ้ำคะแนนในบทความหรือหลักฐานกล้องจริง
- 12 validation videos, 6 events, 174.90s; normal-only 90.53s
- ใช้ training-normal 8 คลิป / 7 การบันทึกต้นฉบับเพื่อเกณฑ์ q95 เท่านั้น; fog และต้นฉบับอยู่ development เหมือนกัน ไม่ข้าม split
- ล็อกสอง sampling ก่อนรัน: stride16=เฟรมแรกทุกกลุ่ม16เฟรม; mean16=เฉลี่ย features ทั้ง16เฟรม; เก็บ tail ตาม FPS จริง
- ใช้ crop กลางตาม official image_crop(type=0), FP32, raw features ไม่เพิ่ม normalization; temporal context256 และ prompt labels UCF เดิม
- Primary: stride16/binary/threshold0.5; อีก7แบบเป็น diagnostic ที่ประกาศก่อนเห็นผล ไม่เลือกแบบชนะจาก validation เพื่อใช้งานจริง
- เกณฑ์อีกแบบคือ95th percentile (higher) ของคะแนน training-normal รวมทุกกล้อง ไม่ใช้ label validation และไม่เปลี่ยนตามผล validation
- upstreamเผยแพร่ features สำเร็จรูป แต่มีความไม่ชัดเจนของ raw extraction จึงต้องระวัง feature-distribution mismatch แม้ใช้ checkpoint และ forward เดิม
- Semantic score=1-softmax(normal), binary score=sigmoid(logit); เป็นคะแนนโมเดล ไม่รับรองว่าเป็น probability ที่ calibrate แล้ว

## Event metrics — temporal IoU≥0.3, one-to-one matching

| วิธี | TP | FP | FN | Recall | Precision | แจ้งผิดต่อชั่วโมงบนคลิปปกติ |
|---|---:|---:|---:|---:|---:|---:|
| gemma_windows | 6 | 6 | 0 | 100.0% | 50.0% | 238.59 |
| gemma_qwen | 3 | 2 | 3 | 50.0% | 60.0% | 79.53 |
| stride16/binary/fixed_0.5 | 0 | 0 | 6 | 0.0% | — | 0.00 |
| stride16/binary/training_normal_q95_higher | 0 | 2 | 6 | 0.0% | 0.0% | 0.00 |
| stride16/semantic/fixed_0.5 | 0 | 0 | 6 | 0.0% | — | 0.00 |
| stride16/semantic/training_normal_q95_higher | 1 | 1 | 5 | 16.7% | 50.0% | 39.76 |
| mean16/binary/fixed_0.5 | 0 | 0 | 6 | 0.0% | — | 0.00 |
| mean16/binary/training_normal_q95_higher | 0 | 2 | 6 | 0.0% | 0.0% | 0.00 |
| mean16/semantic/fixed_0.5 | 0 | 0 | 6 | 0.0% | — | 0.00 |
| mean16/semantic/training_normal_q95_higher | 1 | 1 | 5 | 16.7% | 50.0% | 39.76 |

Normal exposure เพียงประมาณ1.5นาที ค่าแจ้งผิดต่อชั่วโมงเป็นการคูณขยายจากชุดเล็ก ไม่เพียงพอรับรองเป้าหมาย≤2เหตุ/ชั่วโมง; precision ที่ไม่มี positive prediction แสดงเป็น —

## Frame ranking — ไม่ใช่ event recall

| Sampling/head | ROC-AUC | AP |
|---|---:|---:|
| stride16/binary/frame_ranking | 0.6618 | 0.6434 |
| stride16/semantic/frame_ranking | 0.6774 | 0.6090 |
| mean16/binary/frame_ranking | 0.6910 | 0.6706 |
| mean16/semantic/frame_ranking | 0.7140 | 0.6770 |

## เวลาและทรัพยากร

- GPU: NVIDIA GeForce RTX 5070; torch 2.10.0+cu128; FP32
- โหลด checkpoint และสร้างโมเดล 7.04s; รวมรันสอง sampling, calibration, repeated scoring และสร้าง metrics 128.67s
- Peak PyTorch allocated 0.925GiB / reserved 1.406GiB
- Peak process RAM 2.071GiB; GPUทั้งการ์ดที่สุ่มวัด 2850MiB (รวมโปรแกรมอื่น, sampled ไม่ใช่ exact peak)
- stride16: decode+CLIP+first scorer ของ12validation clips 18.42s; encode 336ภาพ; cached-feature scorer p50 29.31ms / p95 31.58ms
- mean16: decode+CLIP+first scorer ของ12validation clips 41.72s; encode 5247ภาพ; cached-feature scorer p50 29.28ms / p95 30.05ms
- cached-feature scorer ไม่รวม decode/CLIP; ไม่อ้างเวลาเฉพาะส่วนนี้เป็นความเร็วครบระบบ
- เวลาระบบเดิมมาจากรอบก่อนบนเครื่องเดียวกัน ไม่ได้รันสลับจับคู่ในรอบนี้ จึงไม่สรุป speedup แบบควบคุม

## ข้อสรุปและข้อจำกัด

- Primary VadCLIP พบ0/6เหตุ; เกณฑ์q95 semantic ทั้งสองsampling พบ1/6เหตุ แจ้งผิด1เหตุ เทียบ Gemma→Qwenเดิมพบ3/6เหตุ แจ้งผิด2เหตุ
- semantic-q95 พบ abnormal_scene_8_scenario_8 แต่แจ้งผิด normal_scene_8_scenario_2; binary-q95 เลือกบางส่วนของเหตุฉาก8 แต่ช่วงแตกเป็นสองส่วนจนแต่ละช่วงไม่ผ่าน temporal IoU0.3
- การใช้mean16เพิ่ม frame ranking แต่ไม่เพิ่มจำนวนเหตุที่จับคู่ได้ในเกณฑ์ที่ล็อกไว้ จึงยังไม่มีหลักฐานให้ใช้แทนระบบเดิม
- ดูทุก variant และผลรายคลิป ไม่ใช้ผลที่ดีที่สุดย้อนหลังมาแทน primary ที่ล็อกไว้
- คะแนนต่ำอาจเกี่ยวข้องทั้ง domain shift, extraction mismatch และความสามารถของตัวตรวจ ต้องทดสอบ features ทางการของ UCF ก่อนสรุปว่าโมเดลต้นฉบับทำงานไม่ได้
- ไม่มีการเปิด official test, เปลี่ยน demo/default, ฝึกโมเดล, หรือ merge main
- แหล่งโมเดล: https://github.com/nwpu-zxr/VadCLIP ; ข้อมูล: https://github.com/lilygeorgescu/UBnormal (CC-BY-NC-ND-4.0)

## ทำซ้ำ

`python -m evaluation.embeddinggemma2.benchmark_vadclip --manifest <manifest> --repo <pinned-official-repo> --checkpoint <official-model_ucf.pth> --baseline <frozen-baseline-results> --output <new-folder>`

ผลดิบมี protocol, checkpoint/source hashes, scores ทุกช่วง, features NPZ, timings และ bootstrap95CI; weights/video อยู่ scratch ไม่เข้า Git
