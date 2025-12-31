from typing import List, Dict, Any, Optional
from ..config.settings import settings
from ..services.qdrant_service import qdrant_service
from ..services.postgres_service import postgres_service
from ..services.mcp_client import mcp_client
from ..utils.text_chunker import text_chunker
import cohere
import logging
import json
import time

logger = logging.getLogger(__name__)


class RAGService:
    """
    Service class to handle the Retrieval-Augmented Generation pipeline
    """

    def __init__(self):
        self.cohere_client = settings.cohere_client
        self.qdrant_service = qdrant_service
        self.postgres_service = postgres_service
        self.mcp_client = mcp_client

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context chunks from the vector database based on the query
        """
        start_time = time.time()
        logger.info(f"Starting context retrieval for query: {query[:50]}...")

        try:
            # Generate embedding for the query using Cohere
            response = self.cohere_client.embed(
                texts=[query],
                model=settings.EMBEDDING_MODEL,  # e.g., "embed-multilingual-v3.0"
                input_type="search_query"
            )
            query_embedding = response.embeddings[0]

            # Search for similar content in Qdrant
            similar_chunks = self.qdrant_service.search_similar(
                query_embedding=query_embedding,
                top_k=top_k
            )

            duration = time.time() - start_time
            logger.info(f"Context retrieval completed in {duration:.2f}s, found {len(similar_chunks)} chunks")
            return similar_chunks
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error retrieving context after {duration:.2f}s: {str(e)}")
            return []

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], use_restricted_context: bool = False) -> str:
        """
        Generate an answer based on the query and retrieved context
        """
        start_time = time.time()
        logger.info(f"Starting answer generation for query: {query[:50]}...")

        try:
            # Format the context for the LLM
            context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])

            # Placeholder for agent service - will be replaced with OpenAI Agents SDK implementation
            # For now, return a message indicating the new agent service needs to be implemented
            answer = "New agent service with OpenAI Agents SDK needs to be implemented."

            duration = time.time() - start_time
            logger.info(f"Answer generation completed in {duration:.2f}s")
            return answer
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error generating answer after {duration:.2f}s: {str(e)}")
            return "Sorry, I encountered an error while generating the answer."

    async def query_full_book(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a full-book RAG query
        """
        try:
            # Retrieve relevant context
            context_chunks = self.retrieve_context(question)

            if not context_chunks:
                logger.warning("No context chunks found for query")
                return {
                    "question": question,
                    "answer": "I couldn't find relevant information in the book to answer your question.",
                    "sources": [],
                    "session_id": session_id,
                    "timestamp": __import__('time').time()
                }

            # Generate answer based on context
            answer = self.generate_answer(question, context_chunks, use_restricted_context=False)

            # Format sources for response
            sources = []
            for chunk in context_chunks:
                sources.append({
                    "file_path": chunk.get("file_path", ""),
                    "section": chunk.get("section", ""),
                    "relevance_score": chunk.get("relevance_score", 0.0)
                })

            # Log the interaction
            interaction_id = await self.postgres_service.log_interaction(
                question_content=question,
                answer_content=answer,
                mode="full",
                session_id=session_id,
                source_chunks=sources
            )

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "session_id": session_id,
                "timestamp": __import__('time').time()
            }
        except Exception as e:
            logger.error(f"Error in full-book query: {str(e)}")
            return {
                "question": question,
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": [],
                "session_id": session_id,
                "timestamp": __import__('time').time()
            }

    async def query_selected_text(self, question: str, selected_text: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a selected-text QA query
        """
        try:
            # Create a single context chunk from the selected text
            context_chunks = [{
                "id": "selected_text",
                "content": selected_text,
                "file_path": "selected_text",
                "section": "selected",
                "chapter": "selected",
                "chunk_index": 0,
                "relevance_score": 1.0,
                "metadata": {}
            }]

            # Generate answer based only on the selected text
            answer = self.generate_answer(question, context_chunks, use_restricted_context=True)

            # Log the interaction
            interaction_id = await self.postgres_service.log_interaction(
                question_content=question,
                answer_content=answer,
                mode="selected",
                session_id=session_id,
                source_chunks=[{
                    "file_path": "selected_text",
                    "section": "selected",
                    "relevance_score": 1.0
                }]
            )

            return {
                "question": question,
                "answer": answer,
                "sources": [],  # No sources since we're using user-provided text
                "session_id": session_id,
                "timestamp": __import__('time').time()
            }
        except Exception as e:
            logger.error(f"Error in selected-text query: {str(e)}")
            return {
                "question": question,
                "answer": "Sorry, I encountered an error while processing your question.",
                "sources": [],
                "session_id": session_id,
                "timestamp": __import__('time').time()
            }


# Global instance of RAGService
rag_service = RAGService()