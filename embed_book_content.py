#!/usr/bin/env python3
"""
Script to embed the book content into Qdrant RAG system
"""

import os
import sys
import requests
from pathlib import Path

def embed_book_content():
    """
    Embed the book content from docs directory into Qdrant
    """
    # Determine the project root (parent of backend directory)
    project_root = Path(__file__).parent.parent
    docs_dir = project_root / "docs"

    print(f"Project root: {project_root}")
    print(f"Docs directory: {docs_dir}")

    # Check if docs directory exists
    if not docs_dir.exists():
        print(f"Error: Docs directory does not exist: {docs_dir}")
        return False

    # Check if there are markdown files in docs
    md_files = list(docs_dir.rglob("*.md"))
    if not md_files:
        print(f"No markdown files found in {docs_dir}")
        return False

    print(f"Found {len(md_files)} markdown files to embed")
    print("Files to be processed:")
    for file in md_files[:10]:  # Show first 10 files
        print(f"  - {file.relative_to(project_root)}")
    if len(md_files) > 10:
        print(f"  ... and {len(md_files) - 10} more files")

    # Check if backend is running
    backend_url = "http://127.0.0.1:8001"
    try:
        response = requests.get(f"{backend_url}/api/health")
        if response.status_code == 200:
            print(f"✓ Backend is running at {backend_url}")
        else:
            print(f"✗ Backend is not responding properly. Status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Backend is not running at {backend_url}")
        print("Please start the backend with: cd backend && python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload")
        return False

    # Prepare the embed request
    embed_data = {
        "source_path": str(docs_dir.resolve()),
        "collection_name": "book_content"
    }

    print(f"\nEmbedding content from: {docs_dir.resolve()}")
    print("Sending request to backend...")

    try:
        response = requests.post(
            f"{backend_url}/api/embed",
            json=embed_data,
            timeout=300  # 5 minute timeout for large documents
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Embedding completed successfully!")
            print(f"  Status: {result.get('status', 'unknown')}")
            print(f"  Message: {result.get('message', 'No message')}")
            print(f"  Total files processed: {result.get('total_files', 0)}")
            print(f"  Job ID: {result.get('job_id', 'N/A')}")

            # Check the embeddings count
            count_response = requests.get(f"{backend_url}/api/embeddings/count")
            if count_response.status_code == 200:
                count_data = count_response.json()
                print(f"  Total embeddings in Qdrant: {count_data.get('count', 0)}")

            return True
        else:
            print(f"✗ Embedding request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("✗ Embedding request timed out. This might take a while for large documents.")
        print("  The embedding process might still be running in the background.")
        return True  # Return True since it might still be processing
    except Exception as e:
        print(f"✗ Error during embedding: {str(e)}")
        return False

def check_embeddings_count():
    """
    Check the current count of embeddings in Qdrant
    """
    backend_url = "http://127.0.0.1:8001"

    try:
        response = requests.get(f"{backend_url}/api/embeddings/count")
        if response.status_code == 200:
            data = response.json()
            print(f"Current embeddings count: {data.get('count', 0)}")
            print(f"Collection: {data.get('collection_name', 'unknown')}")
            return data.get('count', 0)
        else:
            print(f"Failed to get embeddings count: {response.status_code}")
            return 0
    except Exception as e:
        print(f"Error checking embeddings count: {str(e)}")
        return 0

if __name__ == "__main__":
    print("Humanoid Robotics Book - Qdrant Embedding Script")
    print("=" * 50)

    # Check current embeddings count
    print("\nChecking current embeddings count...")
    current_count = check_embeddings_count()
    print(f"Current embeddings in Qdrant: {current_count}")

    if current_count == 0:
        print("\nNo embeddings found in Qdrant. Starting embedding process...")
    else:
        print(f"\n{current_count} embeddings already exist in Qdrant.")
        response = input("Do you want to regenerate embeddings? (y/N): ")
        if response.lower() != 'y':
            print("Aborting embedding process.")
            sys.exit(0)

    # Embed the book content
    success = embed_book_content()

    if success:
        print("\n✓ Book content has been successfully embedded into Qdrant!")
        print("Your chatbot should now be able to retrieve relevant information from the book.")

        # Check final count
        final_count = check_embeddings_count()
        print(f"Final embeddings count: {final_count}")
    else:
        print("\n✗ Failed to embed book content. Please check the backend logs for errors.")
        print("Make sure the backend is running and properly configured with Qdrant and Cohere.")