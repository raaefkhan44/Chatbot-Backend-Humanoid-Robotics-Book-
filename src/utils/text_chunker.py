import re
from typing import List, Tuple, Dict, Any


class TextChunker:
    """
    Utility class for chunking text content into smaller pieces for embedding
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        """
        Initialize with default chunk size and overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks using simple character-based splitting with overlap
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the end, try to split at a natural break point
            if end < len(text):
                # Look for natural break points (newlines, periods, etc.)
                search_text = text[start:end]
                last_newline = search_text.rfind('\n')
                last_period = search_text.rfind('. ')
                last_space = search_text.rfind(' ')

                # Choose the best break point
                break_point = last_newline if last_newline > last_period else last_period
                if break_point < self.chunk_size - 50:  # Only use if not too far from end
                    break_point = last_space

                if break_point > start + 100:  # Ensure chunk is not too small
                    end = start + break_point + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position, accounting for overlap
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk the input text and return list of chunks with metadata
        """
        if metadata is None:
            metadata = {}

        chunks = self.split_text(text)

        chunked_data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "content": chunk,
                "chunk_index": i,
                "metadata": metadata.copy(),
                "length": len(chunk)
            }
            chunked_data.append(chunk_data)

        return chunked_data

    def chunk_markdown(self, markdown_text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Specialized method for chunking markdown content while preserving structure
        """
        if metadata is None:
            metadata = {}

        chunks = self.chunk_text(markdown_text, metadata)

        # Add markdown-specific processing if needed
        for chunk in chunks:
            content = chunk["content"]
            code_block_count = content.count("```")
            if code_block_count % 2 == 1:
                pass

        return chunks

    def validate_chunk(self, chunk: str) -> Tuple[bool, str]:
        """
        Validate that a chunk meets the requirements
        """
        if len(chunk) < 50:
            return False, "Chunk too short (less than 50 characters)"

        if len(chunk) > 2000:
            return False, "Chunk too long (more than 2000 characters)"

        return True, "Valid chunk"


# Global instance of TextChunker
text_chunker = TextChunker()
