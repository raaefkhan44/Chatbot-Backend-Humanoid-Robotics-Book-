from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
import asyncio

from ...services.embedding_service import embedding_service

# Create a limiter for this router with a lower limit for embedding operations
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


class EmbedRequest(BaseModel):
    source_path: str = Field(..., description="Path to the book content directory")
    collection_name: str = Field("book_content", description="Name of the Qdrant collection to use")


@router.post("/embed")
@limiter.limit("10/minute")
async def embed_endpoint(request: Request, embed_request: EmbedRequest) -> Dict[str, Any]:
    """
    Regenerate embeddings from all MD files and push to vector database
    """
    try:
        # Validate source path exists
        import os
        if not os.path.exists(embed_request.source_path):
            raise HTTPException(status_code=400, detail=f"Source path does not exist: {embed_request.source_path}")

        if not os.path.isdir(embed_request.source_path):
            raise HTTPException(status_code=400, detail=f"Source path is not a directory: {embed_request.source_path}")

        # Process the embedding request
        result = embedding_service.regenerate_embeddings(
            source_path=embed_request.source_path,
            collection_name=embed_request.collection_name
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing embedding request: {str(e)}")


@router.get("/embeddings/count")
@limiter.limit("100/minute")
async def embeddings_count_endpoint(request: Request) -> Dict[str, Any]:
    """
    Get the total count of embeddings in the vector database
    """
    try:
        count = embedding_service.count_embeddings()
        return {
            "count": count,
            "collection_name": embedding_service.qdrant_service.collection_name,
            "timestamp": __import__('time').time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embeddings count: {str(e)}")