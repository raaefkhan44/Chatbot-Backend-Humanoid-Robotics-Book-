from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .settings import settings
import logging

logger = logging.getLogger(__name__)

# Create the async database engine with connection pooling
engine = create_async_engine(
    settings.NEON_DATABASE_URL,  # Using NEON_DATABASE_URL as defined in settings
    pool_size=20,  # Number of connection objects to maintain
    max_overflow=30,  # Number of connections that can be created beyond pool_size
    pool_pre_ping=True,  # Verify connections are alive before using them
    pool_recycle=300,  # Recycle connections after 5 minutes
    echo=False,  # Set to True to log SQL queries for debugging
    pool_timeout=30  # Time in seconds to wait for a connection from the pool
)

# Create a configured "AsyncSession" class
AsyncSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    class_=AsyncSession  # Specify that this should create AsyncSession instances
)

# Create a Base class for declarative models
Base = declarative_base()


async def get_db():
    """
    Dependency function to get database session
    """
    async with AsyncSessionLocal() as db:
        yield db


async def init_db():
    """
    Initialize the database by creating all tables
    """
    try:
        # Import all models to ensure they're registered with Base.metadata
        from ..models.chat_models import Question, Answer, RetrievedContext
        from ..models.log_models import LogEntry
        from ..models.embedding_models import BookContentChunk, EmbeddingJob

        # Create all tables using async engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise