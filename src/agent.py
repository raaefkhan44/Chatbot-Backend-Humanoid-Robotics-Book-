"""
Agent implementation for the Book RAG Chatbot
Uses OpenAI Agents SDK with Gemini 2.5 Flash via OpenAI-style external provider
"""
import os
import json
from typing import Dict, Any, List
import google.generativeai as genai
from .connection import connection_manager
import logging

logger = logging.getLogger(__name__)

class RAGQueryTool:
    """
    Tool for content retrieval from book database
    """
    def __init__(self):
        self.connection = connection_manager

    def __call__(self, query: str, mode: str, top_k: int) -> Dict[str, Any]:
        """
        Execute content retrieval based on mode
        """
        try:
            # Generate embedding for the query
            query_embeddings = self.connection.embed([query])
            query_embedding = query_embeddings[0]

            if mode == "rag":
                # Full-book RAG mode - search Qdrant
                results = self.connection.qdrant_search(query_embedding, top_k)
            elif mode == "selected":
                # Selected-text mode - use provided text
                # For this mode, we'll return the query itself as context
                # since the actual selected text would be passed separately
                results = self.connection.selected_text_search(query)
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'rag' or 'selected'")

            return {
                "results": results,
                "query": query,
                "mode": mode
            }
        except Exception as e:
            logger.error(f"Error in rag_query tool: {str(e)}")
            raise

# Initialize the tool
rag_query_tool = RAGQueryTool()

class BookRAGAgent:
    """
    Agent implementation for book Q&A using Gemini 2.5 Flash
    """
    def __init__(self):
        self.connection = connection_manager
        self.rag_query_tool = rag_query_tool
        self.model = self.connection.chat_model

        # System instructions - designed for clean, natural responses
        self.system_instructions = """You are a friendly AI assistant helping students learn about Physical AI and Humanoid Robotics.

CORE BEHAVIOR:
- Be conversational, clear, and professional
- Provide helpful explanations based on the textbook content
- Write in natural paragraphs (not bullet points unless listing specific items)
- Never mention "RAG", "chunks", "tools", "embeddings", or internal processes
- Never show similarity scores, debug info, or technical artifacts

RESPONSE FORMAT:
- Start directly with your answer (no meta-commentary)
- Use 2-3 clear paragraphs to explain concepts
- Sound like a knowledgeable tutor, not a robot
- Keep explanations accessible but accurate"""

    def _classify_query(self, message: str) -> str:
        """Classify the user's message type"""
        message_lower = message.lower().strip()
        words = message_lower.split()

        # Greetings
        greeting_words = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy']
        if any(message_lower == word or (len(words) <= 2 and word in words) for word in greeting_words):
            return 'greeting'

        # Thanks
        thanks_words = ['thanks', 'thank you', 'thx', 'ty', 'appreciate']
        if len(words) <= 3 and any(word in message_lower for word in thanks_words):
            return 'thanks'

        # Very short ambiguous queries (1-3 chars like "ros", "ai")
        if len(message.strip()) <= 3 and message.strip().isalpha():
            return 'short_query'

        # Single word queries that need context
        if len(words) == 1 and len(message.strip()) > 3:
            return 'short_query'

        return 'knowledge'

    def run(self, message: str, selected_text: str = None) -> Dict[str, Any]:
        """
        Execute the agent with the given message
        """
        try:
            logger.info(f"Agent run started - Message: '{message[:50]}...', Selected text: {bool(selected_text)}")

            # Classify the query type
            query_type = self._classify_query(message)
            logger.info(f"Query classified as: {query_type}")

            # Handle greetings
            if query_type == 'greeting':
                return {
                    "answer": "Hello! 👋 I'm here to help you learn about Physical AI and Humanoid Robotics. What would you like to know?",
                    "sources": [],
                    "context_used": False
                }

            # Handle thanks
            if query_type == 'thanks':
                return {
                    "answer": "You're welcome! Feel free to ask if you have any other questions.",
                    "sources": [],
                    "context_used": False
                }

            # Handle short/ambiguous queries - always ask for clarification
            if query_type == 'short_query':
                message_lower = message.lower().strip()

                # Provide helpful clarification prompts
                if message_lower in ['ros', 'ros2', 'ros 2']:
                    return {
                        "answer": "I'd be happy to explain ROS (Robot Operating System)! Would you like to know:\n- What ROS 2 is and its key features?\n- How it works as a communication framework?\n- Its role in humanoid robotics?",
                        "sources": [],
                        "context_used": False
                    }
                elif message_lower in ['ai', 'ml']:
                    return {
                        "answer": "I can explain AI and machine learning in robotics! Are you interested in:\n- AI architectures for humanoid robots?\n- Machine learning applications?\n- Specific AI tools like Isaac Sim?",
                        "sources": [],
                        "context_used": False
                    }
                else:
                    return {
                        "answer": f"I'd be happy to explain '{message}'! Could you provide a bit more detail? For example:\n- Are you asking about its definition?\n- How it works?\n- Its applications in robotics?",
                        "sources": [],
                        "context_used": False
                    }

            # Determine mode based on whether selected text is provided
            mode = "selected" if selected_text else "rag"

            # Call the rag_query tool to retrieve relevant context
            if mode == "selected":
                context_result = self.rag_query_tool(selected_text, mode, top_k=5)
            else:
                context_result = self.rag_query_tool(message, mode, top_k=5)

            # Extract context from results
            context_chunks = context_result["results"]

            # Limit content size and create a more structured summary to avoid recitation triggers
            # Take only the top 3 most relevant chunks and truncate each to 300 chars
            summarized_chunks = []
            for i, chunk in enumerate(context_chunks[:3]):
                if chunk["content"]:
                    content = chunk["content"]
                    # Truncate long content
                    if len(content) > 300:
                        content = content[:300] + "..."
                    summarized_chunks.append(f"[Source {i+1} - {chunk.get('section', 'Unknown')}]:\n{content}")

            context_text = "\n\n".join(summarized_chunks)

            # Create a clean, natural prompt
            full_prompt = f"""You are a helpful tutor. A student asks: "{message}"

Here's relevant information from the textbook:
{context_text}

Provide a clear, natural explanation in 2-3 paragraphs. Write conversationally - no bullet points, no meta-commentary, no mention of "sources" or "context". Just explain the concept clearly."""

            # Generate response using Gemini with safety settings to reduce false positives
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Try with progressively shorter context if recitation is detected
            max_retries = 3
            answer = None

            for attempt in range(max_retries):
                try:
                    # On retry, progressively reduce context
                    prompt_to_use = full_prompt
                    if attempt == 1:
                        # Second attempt: Shorter context
                        logger.info(f"Retry {attempt + 1}: Using reduced context")
                        if context_chunks:
                            short_content = context_chunks[0]["content"][:150] + "..."
                            prompt_to_use = f"""A student asks: "{message}"

Reference: {short_content}

Explain this naturally in 2 paragraphs."""

                    elif attempt == 2:
                        # Third attempt: General knowledge
                        logger.info(f"Retry {attempt + 1}: Using general knowledge")
                        prompt_to_use = f"""Explain "{message}" in the context of robotics and humanoid systems. Write 2 clear, helpful paragraphs."""

                    response = self.model.generate_content(
                        prompt_to_use,
                        generation_config={
                            "temperature": 0.7 + (attempt * 0.15),  # Increase temperature on retry
                            "max_output_tokens": 1500,
                            "top_p": 0.95,
                            "top_k": 40,
                        },
                        safety_settings=safety_settings
                    )

                    # Check if response was blocked (FinishReason 12)
                    was_blocked = False
                    if hasattr(response, 'candidates') and response.candidates:
                        for candidate in response.candidates:
                            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 12:
                                was_blocked = True
                                logger.warning(f"Attempt {attempt + 1}: Response blocked (FinishReason 12)")
                                break

                    if was_blocked and attempt < max_retries - 1:
                        # Try again with even less context
                        continue

                    # Extract text from response
                    answer = self._extract_answer_from_response(response)

                    # If still blocked after all retries, provide helpful message
                    if was_blocked and attempt == max_retries - 1:
                        logger.info("All retry attempts exhausted, providing fallback message")
                        answer = "I understand you're asking about this topic. While I'm having trouble generating a detailed response, I can tell you that the relevant information can be found in the sections listed below. Please refer to those sections for a complete explanation."

                    break

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} error: {str(e)}")
                    if attempt == max_retries - 1:
                        answer = "I'm having trouble answering right now. Please try again or rephrase your question."
                        break

            # Format and deduplicate sources aggressively
            sources = []
            seen_sections = set()

            for chunk in context_chunks:
                section = chunk.get("section", "Unknown")

                # Skip if we've already seen this section
                if section in seen_sections:
                    continue

                seen_sections.add(section)
                sources.append({
                    "file_path": chunk.get("file_path", ""),
                    "section": section,
                    "relevance_score": chunk.get("relevance_score", 0.0)
                })

                # Stop after 3 unique sections
                if len(sources) >= 3:
                    break

            logger.info(f"Agent run completed successfully - Answer length: {len(answer)}, Sources: {len(sources)}")
            return {
                "answer": answer,
                "sources": sources,
                "context_used": bool(context_chunks)
            }
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}", exc_info=True)
            # Return more detailed error message for debugging
            error_detail = f"Agent error: {type(e).__name__}: {str(e)}"
            logger.error(error_detail)
            return {
                "answer": f"I'm having trouble answering right now. Please try again. (Error: {type(e).__name__})",
                "sources": [],
                "context_used": False
            }

    def _extract_answer_from_response(self, response) -> str:
        """
        Safely extract text from Gemini response, handling various edge cases.
        """
        try:
            if not response:
                logger.error("Empty response object from Gemini")
                return "I'm having trouble answering right now. Please try again."

            # First try the direct text access (most reliable)
            try:
                if hasattr(response, 'text') and response.text:
                    if isinstance(response.text, str) and len(response.text.strip()) > 0:
                        return response.text
            except ValueError as ve:
                # This happens when response.text raises an exception (e.g., due to safety filters)
                logger.warning(f"Could not access response.text: {str(ve)}")

            # Check if response has candidates (when the model doesn't produce content)
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    # Try to get text from the parts first
                    if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text and len(part.text.strip()) > 0:
                                return part.text

                    # Check finish reason after trying to get content
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        if finish_reason == 12:  # RECITATION - content blocked due to potential copyright
                            logger.warning(f"Response blocked due to recitation/copyright concerns (reason: {finish_reason})")
                            return "I apologize, but I cannot provide that answer due to content policy restrictions. Please try rephrasing your question."
                        elif finish_reason not in [1, None]:  # 1 is normal stop, None is ok
                            logger.warning(f"Response finished with reason: {finish_reason}")
                            return "I'm having trouble generating a response. Please try rephrasing your question."

            # Handle the case where response.text is None or empty string
            logger.error(f"No valid text in response")
            return "I'm having trouble answering right now. Please try again."

        except Exception as e:
            logger.error(f"Error extracting answer from response: {str(e)}")
            return "I'm having trouble answering right now. Please try again."

# Global instance of the agent
book_rag_agent = BookRAGAgent()