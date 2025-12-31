from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# Get the API key from environment variable
API_KEY = os.getenv("API_KEY", "")
API_KEY_NAME = "X-API-Key"


class APIKeySecurity:
    """
    Middleware to handle API key security for external services
    """

    def __init__(self):
        self.enabled = bool(API_KEY)  # Only enable if API_KEY is set
        if self.enabled:
            logger.info("API key security is enabled")
        else:
            logger.warning("API key security is disabled - no API_KEY environment variable set")

    async def __call__(self, request: Request, call_next):
        # For endpoints that require authentication, check the API key
        # For now, we'll just check if the API key is provided in the header
        # and log the access (in a real implementation, we'd validate it)

        if self.enabled:
            api_key_header = request.headers.get(API_KEY_NAME)
            if not api_key_header:
                logger.warning(f"Unauthorized access attempt from {request.client.host}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key is missing"
                )

            if api_key_header != API_KEY:
                logger.warning(f"Invalid API key provided by {request.client.host}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )

        response = await call_next(request)
        return response


def verify_api_key(credentials: HTTPAuthorizationCredentials = None) -> Optional[str]:
    """
    Dependency to verify API key in specific routes
    """
    if not API_KEY:
        # If no API key is configured, skip verification
        return None

    if credentials and credentials.credentials == API_KEY:
        return credentials.credentials
    else:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


# Global instance of API key security
api_key_security = APIKeySecurity()