from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
import traceback
import time

logger = logging.getLogger(__name__)


class APIErrorHandler:
    """
    Centralized error handling for the API
    """

    @staticmethod
    async def handle_validation_error(request: Request, exc: HTTPException):
        """
        Handle validation errors and return standardized error response
        """
        error_id = f"err_{int(time.time())}_{hash(str(exc)) % 10000:04d}"

        error_response = {
            "error": exc.detail if hasattr(exc, 'detail') else "Validation error",
            "details": str(exc),
            "timestamp": time.time(),
            "request_id": error_id,
            "path": request.url.path,
            "method": request.method
        }

        logger.error(f"Validation error {error_id}: {exc.detail if hasattr(exc, 'detail') else str(exc)}")

        return JSONResponse(
            status_code=exc.status_code if hasattr(exc, 'status_code') else 422,
            content=error_response
        )

    @staticmethod
    async def handle_general_error(request: Request, exc: Exception):
        """
        Handle general errors and return standardized error response
        """
        error_id = f"err_{int(time.time())}_{hash(str(exc)) % 10000:04d}"

        error_response = {
            "error": "Internal server error",
            "details": "An unexpected error occurred",
            "timestamp": time.time(),
            "request_id": error_id,
            "path": request.url.path,
            "method": request.method
        }

        # Log the full traceback for debugging
        logger.error(f"Internal error {error_id}: {str(exc)}\n{traceback.format_exc()}")

        return JSONResponse(
            status_code=500,
            content=error_response
        )


# Create middleware to add request ID to each request
async def add_request_id_middleware(request: Request, call_next):
    """
    Middleware to add a unique request ID to each request
    """
    request_id = f"req_{int(time.time())}_{hash(str(request.url) + str(time.time())) % 10000:04d}"
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response