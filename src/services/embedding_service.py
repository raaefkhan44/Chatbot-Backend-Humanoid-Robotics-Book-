from typing import List, Dict, Any, Optional
from ..config.settings import settings
from ..services.qdrant_service import qdrant_service
from ..services.postgres_service import postgres_service
from ..utils.document_parser import document_parser
from ..utils.text_chunker import text_chunker
import cohere
import logging
import os
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service class to handle document parsing, chunking, and embedding generation
    """

    def __init__(self):
        self.cohere_client = settings.cohere_client
        self.qdrant_service = qdrant_service
        self.postgres_service = postgres_service

    def generate_embeddings_for_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Generate embeddings for a list of documents and store them in Qdrant
        """
        try:
            # Create the collection if it doesn't exist
            self.qdrant_service.create_collection()

            for i, doc in enumerate(documents):
                logger.info(f"Processing document {i+1}/{len(documents)}: {doc['file_path']}")

                # Chunk the document content
                chunks = text_chunker.chunk_markdown(doc["content"], {
                    "file_path": doc["file_path"],
                    "title": doc["title"],
                    "sections": doc["sections"],
                    "chapter": doc["chapter"]
                })

                # Process chunks in batches for efficiency
                batch_size = 96  # Cohere's recommended batch size
                for j in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[j:j + batch_size]

                    # Extract content from batch chunks
                    batch_contents = [chunk_data["content"] for chunk_data in batch_chunks]

                    # Validate chunks before sending to Cohere
                    valid_contents = []
                    valid_indices = []

                    for idx, chunk_data in enumerate(batch_chunks):
                        content = chunk_data["content"]
                        is_valid, error_msg = text_chunker.validate_chunk(content)
                        if is_valid:
                            valid_contents.append(content)
                            valid_indices.append(idx)
                        else:
                            logger.warning(f"Skipping invalid chunk: {error_msg}")

                    if not valid_contents:
                        continue

                    # Generate embeddings using Cohere
                    try:
                        response = self.cohere_client.embed(
                            texts=valid_contents,
                            model=settings.EMBEDDING_MODEL,  # e.g., "embed-multilingual-v3.0"
                            input_type="search_document"
                        )

                        # Store embeddings in Qdrant
                        for idx, embedding in enumerate(response.embeddings):
                            original_idx = valid_indices[idx]
                            chunk_data = batch_chunks[original_idx]
                            content = chunk_data["content"]

                            # Store in Qdrant - use a proper ID format (Qdrant expects UUIDs or integers)
                            import uuid
                            chunk_id = str(uuid.uuid4())  # Generate a unique UUID for each chunk
                            metadata = {
                                "file_path": doc["file_path"],
                                "section": chunk_data["metadata"].get("sections", [""])[0] if chunk_data["metadata"].get("sections") else "",
                                "chapter": chunk_data["metadata"].get("chapter", ""),
                                "chunk_index": j + original_idx,
                                "title": chunk_data["metadata"].get("title", ""),
                            }

                            success = self.qdrant_service.store_embedding(
                                chunk_id=chunk_id,
                                content=content,
                                embedding=embedding,
                                metadata=metadata
                            )

                            if not success:
                                logger.error(f"Failed to store embedding for chunk {chunk_id}")
                    except Exception as e:
                        logger.error(f"Error generating embeddings for batch: {str(e)}")
                        continue

            logger.info("Embedding generation completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            return False

    def process_directory(self, directory_path: str, job_id: Optional[str] = None) -> bool:
        """
        Process all markdown files in a directory and generate embeddings
        """
        try:
            # Parse all markdown files in the directory
            documents = document_parser.parse_directory(directory_path)

            if not documents:
                logger.warning(f"No markdown documents found in {directory_path}")
                if job_id:
                    self.postgres_service.update_embedding_job(
                        job_id,
                        status="failed",
                        error_message="No markdown documents found in directory"
                    )
                return False

            logger.info(f"Found {len(documents)} documents to process")

            # Create embedding job record if not provided
            if not job_id:
                job_id = self.postgres_service.create_embedding_job(len(documents), "processing")
                logger.info(f"Created embedding job: {job_id}")

            # Update job status as processing
            self.postgres_service.update_embedding_job(job_id, status="processing")

            try:
                # Generate embeddings for all documents
                success = self.generate_embeddings_for_documents(documents)

                # Update job status based on result
                final_status = "completed" if success else "failed"
                self.postgres_service.update_embedding_job(
                    job_id,
                    status=final_status,
                    processed_files=len(documents),
                    total_embeddings=self.qdrant_service.get_embedding_count(),
                    end_time=datetime.utcnow()
                )

                return success
            except Exception as e:
                # Update job status to failed
                self.postgres_service.update_embedding_job(
                    job_id,
                    status="failed",
                    error_message=str(e),
                    end_time=datetime.utcnow()
                )
                raise e
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            if job_id:
                self.postgres_service.update_embedding_job(
                    job_id,
                    status="failed",
                    error_message=str(e),
                    end_time=datetime.utcnow()
                )
            return False

    def process_directory_async(self, directory_path: str) -> str:
        """
        Process directory asynchronously and return job ID
        """
        # Create embedding job record
        job_id = self.postgres_service.create_embedding_job(
            total_files=len(document_parser.parse_directory(directory_path)),
            status="pending"
        )

        # Update to processing and start async processing
        self.postgres_service.update_embedding_job(job_id, status="processing")

        # In a real implementation, we would use a task queue like Celery
        # For now, we'll simulate async processing by calling the sync method in a thread
        import threading
        thread = threading.Thread(
            target=self.process_directory,
            args=(directory_path, job_id)
        )
        thread.start()

        return job_id

    def count_embeddings(self) -> int:
        """
        Get the total count of embeddings in the vector database
        """
        return self.qdrant_service.get_embedding_count()

    def regenerate_embeddings(self, source_path: str, collection_name: str = "book_content") -> Dict[str, Any]:
        """
        Regenerate embeddings from all MD files and push to vector database
        """
        try:
            # Update Qdrant service collection name if needed
            self.qdrant_service.collection_name = collection_name

            # Process the directory
            success = self.process_directory(source_path)

            result = {
                "status": "completed" if success else "failed",
                "job_id": f"job_{int(datetime.utcnow().timestamp())}",
                "total_files": len(document_parser.parse_directory(source_path)) if os.path.exists(source_path) else 0,
                "message": "Embedding regeneration completed successfully" if success else "Embedding regeneration failed"
            }

            return result
        except Exception as e:
            logger.error(f"Error regenerating embeddings: {str(e)}")
            return {
                "status": "failed",
                "job_id": f"job_{int(datetime.utcnow().timestamp())}",
                "total_files": 0,
                "message": f"Embedding regeneration failed: {str(e)}"
            }


# Global instance of EmbeddingService
embedding_service = EmbeddingService()