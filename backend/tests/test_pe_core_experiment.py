import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.schemas import SearchQueryRequest
from app.inference.contracts import PEEmbedImagesRequest, PEEmbedTextRequest
from pe_inference_worker.pe_core_service import MODEL_REVISIONS
from app.retrieval.search_engine import HybridMomentSearchEngine


def test_pe_requests_are_path_based_and_release_defaults_false():
    text = PEEmbedTextRequest(model_id="PE-Core-B16-224", texts=["a person"])
    images = PEEmbedImagesRequest(model_id="PE-Core-L14-336", frame_paths=[r"C:\frame.jpg"])
    assert text.release_after is False
    assert images.release_after is False
    assert "base64" not in images.model_dump_json().lower()


def test_search_request_keeps_siglip2_as_default():
    request = SearchQueryRequest(query="person", video_id="video-1")
    assert request.retrieval_backend == "siglip2"


@pytest.mark.parametrize(
    ("backend", "model_id", "dimension"),
    [
        ("siglip2", "google/siglip2-base-patch16-naflex", 768),
        ("pe_core_b16", "PE-Core-B16-224", 1024),
        ("pe_core_l14", "PE-Core-L14-336", 1024),
    ],
)
def test_retrieval_specs_are_isolated(backend, model_id, dimension):
    spec = HybridMomentSearchEngine._retrieval_spec(backend)
    assert spec["model_id"] == model_id
    assert spec["vector_key"] == ("siglip2_vector" if backend == "siglip2" else "pe_core_vector")
    assert dimension in {768, 1024}


def test_pe_index_metadata_id_is_model_version_specific():
    engine = HybridMomentSearchEngine()
    b16 = engine._retrieval_spec("pe_core_b16")
    l14 = engine._retrieval_spec("pe_core_l14")
    assert b16["embedding_version"] != l14["embedding_version"]
    assert b16["model_id"] != l14["model_id"]


def test_pe_checkpoint_revisions_are_pinned_for_each_allowlisted_model():
    assert set(MODEL_REVISIONS) == {"PE-Core-B16-224", "PE-Core-L14-336"}
    assert all(len(revision) == 40 for revision in MODEL_REVISIONS.values())
