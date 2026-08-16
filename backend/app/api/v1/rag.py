from fastapi import APIRouter, HTTPException
from app.retrieval.video_rag import video_rag_engine, VideoQARequest, VideoQAResponse

router = APIRouter()

@router.post("/chat", response_model=VideoQAResponse)
async def chat_with_video(req: VideoQARequest):
    """
    Video-RAG Endpoint: Answers natural language questions based on video transcripts & frames.
    """
    try:
        response = video_rag_engine.answer_question(video_id=req.video_id, question=req.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video QA error: {str(e)}")
