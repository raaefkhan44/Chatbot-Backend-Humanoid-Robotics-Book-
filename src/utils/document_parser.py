import os
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path


class DocumentParser:
    """
    Utility class for parsing book content from markdown files
    """

    def __init__(self):
        pass

    def parse_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a single markdown file and extract content with metadata
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Extract basic metadata from file path
        path_obj = Path(file_path)
        relative_path = str(path_obj.relative_to(path_obj.anchor))

        # Extract title from first H1 if available
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else path_obj.stem

        # Extract sections (H2 headers)
        sections = []
        section_pattern = r'^##\s+(.+)$'
        for match in re.finditer(section_pattern, content, re.MULTILINE):
            sections.append(match.group(1))

        return {
            "file_path": relative_path,
            "title": title,
            "content": content,
            "sections": sections,
            "chapter": path_obj.parent.name  # Use directory name as chapter
        }

    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Parse all markdown files in a directory
        """
        documents = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        doc = self.parse_markdown_file(file_path)
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error parsing file {file_path}: {str(e)}")
        return documents

    def extract_content_chunks(self, content: str, max_chunk_size: int = 1024, overlap: int = 100) -> List[str]:
        """
        Extract content into chunks with specified size and overlap
        """
        chunks = []

        # Split content into sentences to avoid breaking sentences
        sentences = re.split(r'[.!?]+\s+', content)

        current_chunk = ""
        for sentence in sentences:
            # If adding the sentence would exceed the chunk size
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk.strip():
                    # Add the current chunk to the list
                    chunks.append(current_chunk.strip())

                # Start a new chunk with some overlap from the previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Handle case where a single sentence is longer than max_chunk_size
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                # Split long chunks by character count
                for i in range(0, len(chunk), max_chunk_size - overlap):
                    sub_chunk = chunk[i:i + max_chunk_size - overlap]
                    final_chunks.append(sub_chunk)

        return final_chunks


# Global instance of DocumentParser
document_parser = DocumentParser()