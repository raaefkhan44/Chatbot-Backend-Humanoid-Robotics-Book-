from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

from ...app import book_rag_agent

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The user's message/question")
    selected_text: Optional[str] = Field(None, min_length=10, max_length=5000, description="Text selected by user for selected-text mode")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation continuity")


@router.post("/chat")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """
    Main chat endpoint that processes user messages through the RAG agent
    """
    try:
        # Process the message through the agent
        result = book_rag_agent.run(
            message=request.message,
            selected_text=request.selected_text
        )

        # Use provided session_id or generate a new one
        session_id = request.session_id or str(uuid.uuid4())

        response = {
            "answer": result["answer"],
            "sources": result["sources"],
            "session_id": session_id
        }

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


# Include the new route in the router
__all__ = ["router"]