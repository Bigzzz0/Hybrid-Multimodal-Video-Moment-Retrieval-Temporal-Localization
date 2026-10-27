"""Validate artifacts and publish the comparison without rerunning inference."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .kamakura import WORK
from .prepare import ROOT, file_hash


def main():
    media = json.loads((WORK / "media.json").read_text(encoding="utf-8"))
    assert file_hash(Path(media["path"])) == media["media_sha256"]
    reports = {}
    bank = json.loads((ROOT / "evaluation/embeddinggemma2/kamakura_queries.json").read_text(encoding="utf-8"))
    expected = {(event["id"], query): event["positive_windows"] for event in bank["events"] for query in event["queries"]}
    intervals = None
    for profile in ("siglip", "gemma_frames", "gemma_av"):
        report = json.loads((WORK / f"{profile}.json").read_text(encoding="utf-8"))
        assert report["full_clip"] and report["window_count"] == 198
        assert report["media_sha256"] == media["media_sha256"]
        assert len(report["queries"]) == len(expected) == 20
        for row in report["queries"]:
            assert row["positive_windows"] == expected[(row["event_id"], row["query"])]
        with np.load(WORK / f"{profile}-vectors.npz", allow_pickle=False) as archive:
            assert archive["vectors"].shape == (198, 768)
            assert np.isfinite(archive["vectors"]).all()
            np.testing.assert_allclose(np.linalg.norm(archive["vectors"], axis=1), 1, atol=1e-5)
            if intervals is not None:
                np.testing.assert_array_equal(intervals, archive["intervals"])
            intervals = archive["intervals"].copy()
        assert intervals[0,0] == 0 and abs(intervals[-1,1]-media["duration_sec"]) < 1e-6
        report["accuracy_by_language"] = {lang: {f"CenterFrameHit@{k}": float(np.mean([any(hit["window_id"] in row["positive_windows"] for hit in row["top_windows"][:k]) for row in report["queries"] if row["language"] == lang]))*100 for k in (1,5)} for lang in ("th", "en")}
        reports[profile] = report
        (WORK / f"{profile}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    proof = json.loads((WORK / "input-verification.json").read_text(encoding="utf-8"))
    assert proof["passed"] and proof["parameter_count"] == 744371488
    output = ["# ผลทดสอบคลิป Kamakura: EmbeddingGemma 2 แบบ Video + Audio", "",
        "ทดสอบคลิปเต็ม Ghibli-Like Seaside Town / Kamakura One Day Train Trip near Tokyo / JAPAN TRAVEL VLOG จากไฟล์ของผู้ใช้", "",
        f"ความยาว {media['duration_sec']:.2f} วินาที · {len(media['frames'])} เฟรม · 198 ช่วง · 20 คำค้นจาก 10 เหตุการณ์ ไทย–อังกฤษ", "",
        "| โหมด | CenterFrameHit@1 | @5 | ค้นหา p50 | ทำดัชนี | Peak allocated GPU | Peak RAM ขณะทำดัชนี |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"siglip": "SigLIP2 · เฉลี่ยเฟรม", "gemma_frames": "Gemma 2 · เฉลี่ยเฟรม", "gemma_av": "Gemma 2 · Native Video + Audio"}
    for key, report in reports.items():
        output.append(f"| {names[key]} | {report['accuracy']['CenterFrameHit@1']:.1f}% | {report['accuracy']['CenterFrameHit@5']:.1f}% | {report['query_p50_ms']:.2f} ms | {report['index_resources']['wall_seconds']:.2f} s | {report['torch_peak_allocated_mib']:.1f} MiB | {report['index_resources']['peaks']['rss_mib']:.1f} MiB |")
    output += ["", "## ทรัพยากรเพิ่มเติมขณะทำดัชนี", "",
        "| โหมด | CPU core-seconds | GPU utilization เฉลี่ย | กำลังไฟ GPU เฉลี่ยทั้งเครื่อง | ช่วง/วินาที |",
        "|---|---:|---:|---:|---:|"]
    for key, report in reports.items():
        resource = report["index_resources"]
        output.append(f"| {names[key]} | {resource['cpu_core_seconds']:.2f} | {resource['means'].get('gpu_percent', 0):.1f}% | {resource['means'].get('device_power_w', 0):.1f} W | {report['index_windows_per_second']:.3f} |")
    output += ["", "## วิธีทดสอบและขอบเขต", "",
        "- เปิด text, vision และ audio encoder ครบ 744,371,488 parameters; ใช้ BF16 ตามคำแนะนำโมเดล",
        "- Native AV ป้อนเฟรมเรียงตามเวลา สูงสุด 6 ภาพต่อช่วง พร้อมเสียงจริง 16 kHz mono ไม่ป้อนชื่อบท คำตอบอ้างอิง หรือ caption ที่สร้างขึ้น",
        "- ใช้ vision budget สูงสุด 1,120 soft tokens ต่อเฟรมและ relative timestamps ปิด truncation มีการตรวจ input shapes ว่าภาพและเสียงเข้าโมเดลและลำดับไม่เกิน 8,192 tokens",
        f"- ทดลองเสียงจริงเทียบกับเสียงเงียบในภาพชุดเดิม: cosine {proof['audio_vs_silence_cosine']:.6f}, L2 difference {proof['audio_vs_silence_l2']:.6f}; ยืนยันว่าเสียงมีผลต่อเวกเตอร์ ไม่ได้ยืนยันความแม่นในการฟัง",
        "- โหมดภาพเดี่ยวทั้งสองใช้ค่าเฉลี่ยเวกเตอร์เฟรมภายในช่วงเดียวกัน จึงไม่ใช่ protocol ของ benchmark 28 คำค้นเดิม",
        "- Warmup ทั้ง media และ query ก่อนวัด แยกเวลาโหลดออกจาก indexing และแต่ละ query วัด 5 ครั้ง ทุกโมเดลทำงานแยก process และไม่รันพร้อมกัน",
        "- เวลาทำดัชนีรวมการอ่าน JPEG และ model encoding แต่ไม่รวมการถอดคลิปและเตรียม WAV ซึ่งเตรียมร่วมกันก่อนทุกโหมด",
        "- Peak allocated GPU เป็นค่าจาก PyTorch; RAM เป็น process RSS สูงสุดที่สุ่มวัดทุก 1 วินาที ส่วน GPU utilization/กำลังไฟเป็น device-wide จาก nvidia-smi ไม่ใช่พลังงานเฉพาะงาน",
        "- CenterFrameHit วัดภาพกลางช่วงที่ตรวจครบทั้ง 198 ช่วงก่อนดูผลโมเดล ไม่มีการวัด exact temporal IoU, audio recognition หรือ no-match queries",
        "- ชุดนี้มีเพียงคลิปเดียวและ 10 แนวคิดที่เขียนสองภาษา จึงไม่อ้างว่าโมเดลชนะในทุกงาน และยังไม่ได้แยกผลของความละเอียด หลายเฟรม และเสียงออกจากกัน", "",
        "## ตรวจซ้ำ", "",
        "API integration tests 4 ข้อ, input-routing tests 2 ข้อ และ frontend production build ผ่าน", "",
        "ผลดิบ เวกเตอร์ ภาพที่ตรวจ และ input-verification.json อยู่ใน scratch/embeddinggemma2/kamakura", "",
        "[คู่มือ Google](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)", ""]
    (ROOT / "evaluation/embeddinggemma2/KAMAKURA_RESULTS.md").write_text("\n".join(output), encoding="utf-8")
    (WORK / "validation.json").write_text(json.dumps({"validated_at": datetime.now(timezone.utc).isoformat(), "vectors_finite_and_normalized": True, "shared_intervals": True, "media_hash": media["media_sha256"], "annotation_sha256": file_hash(ROOT / "evaluation/embeddinggemma2/kamakura_queries.json"), "input_routing_verified": True}, indent=2), encoding="utf-8")
    print("All three corpora and reports validated; KAMAKURA_RESULTS.md written")

if __name__ == "__main__":
    main()
