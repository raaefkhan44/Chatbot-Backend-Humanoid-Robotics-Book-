"""
Connection configuration for the Book RAG Chatbot
Handles connections to Gemini 2.5 Flash external provider, embedding client, and Qdrant
"""
import os
from dotenv import load_dotenv

# Load environment variables before any settings are accessed
load_dotenv()

from typing import Optional, List, Dict, Any
import google.generativeai as genai
from google.generativeai import configure as configure_genai
from .qdrant_rest import QdrantRestClient
from cohere import Client as CohereClient
import cohere
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Configure Gemini client
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        configure_genai(api_key=self.gemini_api_key)
        self.gemini_client = genai
        self.chat_model = genai.GenerativeModel('gemini-2.5-flash')

        # Configure Cohere client for embeddings
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is required")

        self.cohere_client = CohereClient(api_key=self.cohere_api_key)

        # Configure Qdrant client
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

        self.qdrant_client = QdrantRestClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )

        # Collection name for book content
        self.collection_name = "book_content"

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the provided texts using Cohere
        """
        try:
            response = self.cohere_client.embed(
                texts=texts,
                model="embed-multilingual-v3.0",  # Using Cohere's multilingual embedding model
                input_type="search_document"
            )
            return [embedding for embedding in response.embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def qdrant_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar content in Qdrant
        """
        try:
            # Use the newer Qdrant client API
            search_result = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            ).points

            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'content': hit.payload.get('content', ''),
                    'file_path': hit.payload.get('file_path', ''),
                    'section': hit.payload.get('section', ''),
                    'relevance_score': getattr(hit, 'score', 0),  # Handle different result formats
                    'metadata': hit.payload
                }
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error searching Qdrant: {str(e)}")
            # Fallback to the older API format if available
            try:
                search_result = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k,
                    with_payload=True
                )

                results = []
                for hit in search_result:
                    result = {
                        'id': hit.id,
                        'content': hit.payload.get('content', ''),
                        'file_path': hit.payload.get('file_path', ''),
                        'section': hit.payload.get('section', ''),
                        'relevance_score': hit.score,
                        'metadata': hit.payload
                    }
                    results.append(result)

                return results
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {str(fallback_error)}")
                # Try using scroll method if search is not available
                try:
                    records, _ = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=top_k,
                        with_payload=True
                    )

                    results = []
                    for record in records:
                        result = {
                            'id': record.id,
                            'content': record.payload.get('content', ''),
                            'file_path': record.payload.get('file_path', ''),
                            'section': record.payload.get('section', ''),
                            'relevance_score': 0,  # No similarity score with scroll
                            'metadata': record.payload
                        }
                        results.append(result)

                    return results
                except Exception as scroll_error:
                    logger.error(f"Scroll method also failed: {str(scroll_error)}")
                    raise e  # Re-raise the original error

    def selected_text_search(self, selected_text: str) -> List[Dict[str, Any]]:
        """
        Create a result from selected text (for selected-text QA mode)
        """
        try:
            # Generate embedding for the selected text
            embeddings = self.embed([selected_text])

            # Return the selected text as a single result
            result = {
                'id': 'selected_text',
                'content': selected_text,
                'file_path': 'selected_text',
                'section': 'selected',
                'relevance_score': 1.0,
                'metadata': {}
            }

            return [result]
        except Exception as e:
            logger.error(f"Error processing selected text: {str(e)}")
            raise

# Global instance with lazy initialization
_connection_manager = None


def get_connection_manager():
    """
    Get the connection manager instance, creating it if it doesn't exist
    This allows for lazy initialization after environment variables are loaded
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


# For backward compatibility, expose as a property that initializes on first access
def __getattr__(name):
    if name == "connection_manager":
        return get_connection_manager()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")