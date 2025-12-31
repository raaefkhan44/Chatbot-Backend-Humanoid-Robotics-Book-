from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Service class to handle all Qdrant vector database operations
    """

    def __init__(self):
        """
        Initialize the Qdrant client with configuration from settings
        """
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            prefer_grpc=False  # Set to True in production for better performance
        )
        self.collection_name = "book_content"
        self.vector_size = 1024  # For Cohere's embed-multilingual-v3.0 model

    def create_collection(self) -> bool:
        """
        Create a collection for storing book content chunks with embeddings
        """
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection with appropriate vector size
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    ),
                    # Add payload indexes for efficient filtering
                    optimizers_config=models.OptimizersConfigDiff(
                        memmap_threshold=20000,
                        indexing_threshold=20000,
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
                return True
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
                return True
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {str(e)}")
            return False

    def store_embedding(self,
                       chunk_id: str,
                       content: str,
                       embedding: List[float],
                       metadata: Dict[str, Any]) -> bool:
        """
        Store a single content chunk with its embedding in Qdrant
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=chunk_id,
                        vector=embedding,
                        payload={
                            "content": content,
                            "file_path": metadata.get("file_path", ""),
                            "section": metadata.get("section", ""),
                            "chapter": metadata.get("chapter", ""),
                            "chunk_index": metadata.get("chunk_index", 0),
                            "metadata": metadata
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False

    def search_similar(self,
                      query_embedding: List[float],
                      top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content chunks based on the query embedding
        """
        try:
            # Use the newer Qdrant client API - don't specify 'using' since we have an unnamed vector
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            ).points

            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "content": result.payload.get("content", ""),
                    "file_path": result.payload.get("file_path", ""),
                    "section": result.payload.get("section", ""),
                    "chapter": result.payload.get("chapter", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "relevance_score": getattr(result, 'score', 0),  # Handle different result formats
                    "metadata": result.payload.get("metadata", {})
                })

            return results
        except Exception as e:
            logger.error(f"Error searching similar content: {str(e)}")
            # Fallback to try the older API format if available
            try:
                # Try using the legacy search method without specifying vector name
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False
                )

                results = []
                for result in search_results:
                    results.append({
                        "id": result.id,
                        "content": result.payload.get("content", ""),
                        "file_path": result.payload.get("file_path", ""),
                        "section": result.payload.get("section", ""),
                        "chapter": result.payload.get("chapter", ""),
                        "chunk_index": result.payload.get("chunk_index", 0),
                        "relevance_score": result.score,
                        "metadata": result.payload.get("metadata", {})
                    })

                return results
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {str(fallback_error)}")

                # Try the legacy method without the search method if it doesn't exist
                try:
                    # Use scroll to get points if search is not available
                    records, next_page = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=top_k,
                        with_payload=True,
                        with_vectors=False
                    )

                    results = []
                    for record in records:
                        results.append({
                            "id": record.id,
                            "content": record.payload.get("content", ""),
                            "file_path": record.payload.get("file_path", ""),
                            "section": record.payload.get("section", ""),
                            "chapter": record.payload.get("chapter", ""),
                            "chunk_index": record.payload.get("chunk_index", 0),
                            "relevance_score": 0,  # No similarity score with scroll
                            "metadata": record.payload.get("metadata", {})
                        })

                    return results
                except Exception as scroll_error:
                    logger.error(f"Scroll method also failed: {str(scroll_error)}")
                    return []

    def get_embedding_count(self) -> int:
        """
        Get the total count of embeddings in the collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting embedding count: {str(e)}")
            return 0

    def delete_collection(self) -> bool:
        """
        Delete the entire collection (use with caution!)
        """
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False


# Global instance of QdrantService
qdrant_service = QdrantService()