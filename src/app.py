"""
FastAPI application for the Book RAG Chatbot
Optimized for both local development and Vercel serverless deployment
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Book RAG Chatbot API",
    description="API for Book RAG Chatbot with Floating Icon UI",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    selected_text: Optional[str] = None
    session_id: Optional[str] = None

class Source(BaseModel):
    file_path: str
    section: str
    relevance_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str

# Lazy import for agent to avoid initialization issues in serverless
_book_rag_agent = None

def get_book_rag_agent():
    """Lazy initialization of the RAG agent"""
    global _book_rag_agent
    if _book_rag_agent is None:
        from .agent import book_rag_agent
        _book_rag_agent = book_rag_agent
    return _book_rag_agent

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint that processes user messages through the RAG agent
    """
    try:
        # Validate message length
        if len(request.message.strip()) < 5:
            raise HTTPException(status_code=400, detail="Message must be at least 5 characters long")

        # Use provided session_id or generate a new one
        session_id = request.session_id or f"session_{id(request)}"

        # Get the RAG agent
        agent = get_book_rag_agent()

        # Process the message through the agent
        result = agent.run(
            message=request.message,
            selected_text=request.selected_text
        )

        # Create response
        response = ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            session_id=session_id
        )

        logger.info(f"Processed chat request for session {session_id}")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "book-rag-chatbot",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Book RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/api/health"
    }

# For local development with uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
