from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class BookContentChunk(Base):
    """
    Entity: Book Content Chunk
    Description: Processed segment of book content with metadata
    """
    __tablename__ = "book_content_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)  # The text content of the chunk
    embedding = Column(Text, nullable=True)  # Vector representation of the content (stored as JSON string)
    file_path = Column(String, nullable=False)  # Path to the source markdown file
    section = Column(String, nullable=True)  # Section title where this chunk appears
    chapter = Column(String, nullable=True)  # Chapter name/number
    chunk_index = Column(Integer, nullable=False)  # Position of this chunk within the document
    metadata_json = Column(Text, nullable=True)  # Additional metadata (stored as JSON string)

    def __repr__(self):
        return f"<BookContentChunk(id={self.id}, file_path='{self.file_path}', chunk_index={self.chunk_index})>"


class EmbeddingJob(Base):
    """
    Entity: Embedding Job
    Description: Record of embedding generation jobs
    """
    __tablename__ = "embedding_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(20), nullable=False)  # Current status of the job ('pending', 'processing', 'completed', 'failed')
    total_files = Column(Integer, nullable=True)  # Number of files to process
    processed_files = Column(Integer, nullable=True)  # Number of files processed
    total_embeddings = Column(Integer, nullable=True)  # Total number of embeddings generated
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)  # When the job started
    end_time = Column(DateTime, nullable=True)  # When the job completed
    error_message = Column(Text, nullable=True)  # Error details if job failed

    def __repr__(self):
        return f"<EmbeddingJob(id={self.id}, status={self.status})>"