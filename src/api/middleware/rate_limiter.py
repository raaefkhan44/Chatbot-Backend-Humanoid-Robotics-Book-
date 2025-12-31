from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)

# Initialize the rate limiter
limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiter(app: FastAPI):
    """
    Setup rate limiting for the FastAPI application
    """
    # Register the rate limit exceeded handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    logger.info("Rate limiting setup completed")


# Default rate limits
DEFAULT_LIMIT = "100/minute"
EMBED_LIMIT = "10/minute"  # Lower limit for embedding operations