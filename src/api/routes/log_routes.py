from fastapi import APIRouter, HTTPException, Query, Request
from typing import Optional, Dict, Any
from slowapi import Limiter
from slowapi.util import get_remote_address

from ...services.postgres_service import postgres_service

# Create a limiter for this router
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


@router.get("/logs")
@limiter.limit("100/minute")
async def get_logs_endpoint(
    request: Request,
    limit: int = Query(20, ge=1, le=100, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip"),
    mode: Optional[str] = Query(None, description="Filter by mode ('full' or 'selected')")
) -> Dict[str, Any]:
    """
    Retrieve paginated chat logs from the database
    """
    try:
        result = postgres_service.get_logs(
            limit=limit,
            offset=offset,
            mode=mode
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")