import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval import search_engine as module


@pytest.mark.parametrize("ready", [True, False])
def test_visual_result_displays_ready_caption_without_lexical_hit(monkeypatch, ready):
    settings = module.settings
    vector = [1.0] + [0.0] * (settings.SIGLIP2_EMBEDDING_DIM - 1)
    tables = {
        "videos": [{"id": "v", "duration_sec": 260, "visual_index_version": settings.VISUAL_INDEX_VERSION}],
        "video_frames_v2": [{"id": "f", "timestamp": 249, "scene_id": "s", "siglip2_vector": vector}],
        "scenes_v2": [{"id": "s", "t_start": 246, "t_end": 254, "caption": ""}],
    }
    stored = [{"id": "s", "caption": "Carbonated lemonade in mason jar.", "t_start": 246, "t_end": 254}]
    monkeypatch.setattr(module.db_manager, "get_table", lambda name: name)
    monkeypatch.setattr(module.db_manager, "visual_index_compatibility", lambda _: {"compatible": True})
    monkeypatch.setattr(module.vlm_artifact_store, "caption_rows", lambda *args: stored if ready else [])
    monkeypatch.setattr(module.vlm_artifact_store, "metadata", lambda *args: {"status": "ready" if ready else "running"})
    monkeypatch.setattr(module.query_expander, "get_query_variants", lambda _: [("pancake", 1.0)])
    engine = module.HybridMomentSearchEngine()
    monkeypatch.setattr(engine, "_rows", lambda table, *args: tables.get(table, []))
    monkeypatch.setattr(engine, "_load_calibration", lambda: None)
    monkeypatch.setattr(engine, "_fusion_weights", lambda: (0.8, 0.2))
    monkeypatch.setattr(engine, "_probability", lambda rank, display: (display, False, None))
    engine.text_encoder = SimpleNamespace(encode_texts=lambda *args, **kwargs: [vector])
    engine.smoother = SimpleNamespace(smooth_timeline=lambda *args, **kwargs: (np.array([248., 249., 250.]), np.array([0.1, 0.9, 0.1])))
    engine.boundary_extractor = SimpleNamespace(extract_multiscale_proposals=lambda *args, **kwargs: [{"t_start": 248, "t_end": 250, "score": 0.9}])

    result = engine.search_moments("pancake", "v")

    assert len(result.moments) == 1
    moment = result.moments[0]
    assert moment.caption_full == (stored[0]["caption"] if ready else None)
    assert moment.caption_preview == moment.caption_full
    assert moment.modality_breakdown["caption"] == 0.0
    assert moment.score == pytest.approx(0.9)
