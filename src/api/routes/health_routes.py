from fastapi import APIRouter
from typing import Dict, Any
import time

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify the service is running and dependencies are accessible.
    """
    # In a real implementation, you would check actual connections to external services
    # For now, we'll return a basic health status
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "dependencies": {
            "qdrant": "not_connected",  # Would check actual connection
            "postgres": "not_connected",  # Would check actual connection
            "openai": "not_connected"  # Would check actual connection
        }
    }