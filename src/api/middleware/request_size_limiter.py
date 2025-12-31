import asyncio
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

# Maximum request size in bytes (10MB)
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB


class RequestSizeLimiter:
    """
    Middleware to limit request size to prevent abuse
    """

    def __init__(self, max_size: int = MAX_REQUEST_SIZE):
        self.max_size = max_size

    async def __call__(self, request: Request, call_next):
        # Check content length header first (if available)
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    logger.warning(f"Request too large: {size} bytes from {request.client.host}")
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request too large: {size} bytes. Maximum allowed: {self.max_size} bytes"
                    )
            except ValueError:
                # If content-length header is not a valid integer, continue to read body
                pass

        # For streaming requests or when content-length is not available,
        # we'll read the body and check its size
        body = b""
        try:
            async for chunk in request.stream():
                body += chunk
                if len(body) > self.max_size:
                    logger.warning(f"Request body too large: {len(body)} bytes from {request.client.host}")
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Request body too large",
                            "details": f"Request body exceeds maximum size of {self.max_size} bytes",
                            "timestamp": __import__('time').time()
                        }
                    )

            # Set the body back to the request for further processing
            request._body = body
            request.state.body_size = len(body)
        except Exception as e:
            logger.error(f"Error reading request body: {str(e)}")
            raise HTTPException(status_code=400, detail="Error reading request body")

        response = await call_next(request)
        return response


async def get_body(request: Request):
    """
    Helper function to get the request body, since it might have been consumed by the middleware
    """
    if hasattr(request.state, 'body') and request.state.body:
        return request.state.body
    else:
        # If body was not cached by middleware, read it (but this can only be done once)
        body = await request.body()
        request.state.body = body
        return body