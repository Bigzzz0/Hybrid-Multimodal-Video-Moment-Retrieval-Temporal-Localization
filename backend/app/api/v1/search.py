from fastapi import APIRouter, HTTPException, status
from app.db.schemas import SearchQueryRequest, SearchResponse
from app.retrieval.search_engine import search_engine
from app.core.config import settings

router = APIRouter()

@router.post("/moment", response_model=SearchResponse)
async def search_moments(req: SearchQueryRequest):
    """
    Search and localize moments in video using natural language query.
    Returns ranked moment intervals with start-end timestamps, preview metadata, and density heatmap.
    """
    if not req.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    backend = req.backend or settings.RETRIEVAL_BACKEND
    if backend == "gemma2":
        from app.gemma2.service import get_service
        from app.api.v1.events import translate
        from starlette.concurrency import run_in_threadpool
        result = await run_in_threadpool(lambda: translate(lambda: get_service().search(
            req.video_id, req.query, req.top_k, req.profile)))
        return SearchResponse(**result)
    if backend != "siglip2":
        raise HTTPException(422, "Unknown retrieval backend")

    response = search_engine.search_moments(query=req.query, video_id=req.video_id,
                                             top_k=req.top_k, profile=req.profile)
    response.backend = "siglip2"

    if "reindex_required" in response.warnings:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "reindex_required", "message": "This video uses a legacy visual index. Re-ingest it to build index v2."},
        )
    return response
