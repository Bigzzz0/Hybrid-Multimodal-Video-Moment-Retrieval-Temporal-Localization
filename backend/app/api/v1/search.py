from fastapi import APIRouter, HTTPException, status
from app.db.schemas import SearchQueryRequest, SearchResponse
from app.retrieval.search_engine import search_engine

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

    response = search_engine.search_moments(
        query=req.query,
        video_id=req.video_id,
        top_k=req.top_k,
        weight_visual=req.weight_visual,
        weight_caption=req.weight_caption,
        weight_audio=req.weight_audio,
        gaussian_sigma=req.gaussian_sigma,
        threshold_factor=req.threshold_factor
    )

    return response
