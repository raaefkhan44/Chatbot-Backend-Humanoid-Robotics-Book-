from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

from ..models.chat_models import Question, Answer, RetrievedContext
from ..models.log_models import LogEntry
from ..models.embedding_models import EmbeddingJob
from ..config.database import AsyncSessionLocal


class PostgresService:
    """
    Service class to handle all PostgreSQL database operations
    """

    def __init__(self):
        pass

    def get_db_session(self) -> AsyncSession:
        """
        Get an async database session
        """
        return AsyncSessionLocal()

    async def log_interaction(self,
                              question_content: str,
                              answer_content: str,
                              mode: str,
                              session_id: Optional[str] = None,
                              user_feedback: Optional[str] = None,
                              source_chunks: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Log a user interaction to PostgreSQL
        """
        async with AsyncSessionLocal() as db:
            try:
                # Create Question
                question = Question(
                    content=question_content,
                    session_id=session_id,
                    source_mode=mode
                )
                db.add(question)
                await db.flush()  # Get the ID without committing

                # Create Answer
                answer = Answer(
                    question_id=question.id,
                    content=answer_content,
                    session_id=session_id,
                    source_chunks=json.dumps(source_chunks) if source_chunks else None
                )
                db.add(answer)
                await db.flush()  # Get the ID without committing

                # Create Log Entry
                log_entry = LogEntry(
                    question_id=question.id,
                    answer_id=answer.id,
                    user_session=session_id,
                    mode=mode,
                    user_feedback=user_feedback
                )
                db.add(log_entry)

                await db.commit()
                return str(question.id)
            except Exception as e:
                await db.rollback()
                raise e

    async def get_logs(self,
                       limit: int = 20,
                       offset: int = 0,
                       mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve paginated chat logs from the database
        """
        async with AsyncSessionLocal() as db:
            try:
                # Build the query using select
                query = select(LogEntry).join(Question).join(Answer)

                if mode:
                    query = query.where(LogEntry.mode == mode)

                # Get total count for pagination
                count_query = select(func.count()).select_from(query.subquery())
                result = await db.execute(count_query)
                total = result.scalar()

                # Apply pagination to the main query
                query = query.offset(offset).limit(limit)
                result = await db.execute(query)
                logs = result.scalars().all()

                # Format results
                formatted_logs = []
                for log in logs:
                    formatted_logs.append({
                        "id": log.id,
                        "question": log.question.content,
                        "answer": log.answer.content,
                        "mode": log.mode,
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None
                    })

                return {
                    "logs": formatted_logs,
                    "total": total,
                    "limit": limit,
                    "offset": offset
                }
            except Exception as e:
                raise e

    async def create_embedding_job(self,
                                   total_files: int,
                                   status: str = "pending") -> str:
        """
        Create a new embedding job record
        """
        async with AsyncSessionLocal() as db:
            try:
                job = EmbeddingJob(
                    status=status,
                    total_files=total_files
                )
                db.add(job)
                await db.commit()
                await db.refresh(job)  # Refresh to get the generated ID
                return str(job.id)
            except Exception as e:
                await db.rollback()
                raise e

    async def update_embedding_job(self,
                                   job_id: str,
                                   status: Optional[str] = None,
                                   processed_files: Optional[int] = None,
                                   total_embeddings: Optional[int] = None,
                                   error_message: Optional[str] = None,
                                   end_time: Optional[datetime] = None) -> bool:
        """
        Update an existing embedding job
        """
        async with AsyncSessionLocal() as db:
            try:
                # Use select to get the job
                result = await db.execute(
                    select(EmbeddingJob).where(EmbeddingJob.id == job_id)
                )
                job = result.scalar_one_or_none()

                if not job:
                    return False

                if status is not None:
                    job.status = status
                if processed_files is not None:
                    job.processed_files = processed_files
                if total_embeddings is not None:
                    job.total_embeddings = total_embeddings
                if error_message is not None:
                    job.error_message = error_message
                if end_time is not None:
                    job.end_time = end_time

                await db.commit()
                return True
            except Exception as e:
                await db.rollback()
                raise e

    async def get_embedding_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details of a specific embedding job
        """
        async with AsyncSessionLocal() as db:
            try:
                result = await db.execute(
                    select(EmbeddingJob).where(EmbeddingJob.id == job_id)
                )
                job = result.scalar_one_or_none()

                if not job:
                    return None

                return {
                    "id": str(job.id),
                    "status": job.status,
                    "total_files": job.total_files,
                    "processed_files": job.processed_files,
                    "total_embeddings": job.total_embeddings,
                    "start_time": job.start_time.isoformat() if job.start_time else None,
                    "end_time": job.end_time.isoformat() if job.end_time else None,
                    "error_message": job.error_message
                }
            except Exception as e:
                raise e


# Global instance of PostgresService
postgres_service = PostgresService()