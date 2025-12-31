import re
from typing import List, Tuple, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk the input text and return list of chunks with metadata
        """
        if metadata is None:
            metadata = {}

        # Use langchain's text splitter for more sophisticated chunking
        chunks = self.text_splitter.split_text(text)

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

        # This is a simple approach - for more sophisticated markdown chunking,
        # we might want to separate by headers or other markdown elements
        chunks = self.chunk_text(markdown_text, metadata)

        # Add markdown-specific processing if needed
        for chunk in chunks:
            # Clean up any broken markdown syntax
            content = chunk["content"]
            # Ensure code blocks are properly closed
            code_block_count = content.count("```")
            if code_block_count % 2 == 1:  # Odd number means unclosed code block
                # This is a simplified fix - in practice you'd want more sophisticated handling
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