"""
Verify Qdrant collection and data
"""
import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

from qdrant_client import QdrantClient
from src.connection import get_connection_manager

def verify_qdrant():
    """Verify Qdrant collection and data"""
    print("=" * 60)
    print("QDRANT VERIFICATION")
    print("=" * 60)

    try:
        # Get connection manager
        conn = get_connection_manager()
        client = conn.qdrant_client
        collection_name = conn.collection_name

        print(f"\n✓ Connected to Qdrant")
        print(f"  URL: {os.getenv('QDRANT_URL')}")
        print(f"  Collection: {collection_name}")

        # Check if collection exists
        print(f"\n[1] Checking if collection '{collection_name}' exists...")
        try:
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if collection_name in collection_names:
                print(f"  ✓ Collection '{collection_name}' EXISTS")
            else:
                print(f"  ✗ Collection '{collection_name}' NOT FOUND")
                print(f"  Available collections: {collection_names}")
                print(f"\n  ⚠ You need to run the embedding script:")
                print(f"     python run_embedding.py")
                return False
        except Exception as e:
            print(f"  ✗ Error checking collections: {e}")
            return False

        # Get collection info
        print(f"\n[2] Getting collection info...")
        try:
            collection_info = client.get_collection(collection_name)
            vector_count = collection_info.points_count
            vector_size = collection_info.config.params.vectors.size

            print(f"  ✓ Collection info retrieved")
            print(f"    Vector count: {vector_count}")
            print(f"    Vector dimension: {vector_size}")

            if vector_count == 0:
                print(f"\n  ✗ Collection is EMPTY (no vectors)")
                print(f"     You need to run the embedding script:")
                print(f"     python run_embedding.py")
                return False
            else:
                print(f"  ✓ Collection has data ({vector_count} vectors)")
        except Exception as e:
            print(f"  ✗ Error getting collection info: {e}")
            return False

        # Test embedding
        print(f"\n[3] Testing embedding generation...")
        try:
            test_text = "What is ROS 2?"
            embeddings = conn.embed([test_text])
            embedding_dim = len(embeddings[0])

            print(f"  ✓ Embedding generated successfully")
            print(f"    Input: '{test_text}'")
            print(f"    Embedding dimension: {embedding_dim}")

            if embedding_dim != vector_size:
                print(f"  ✗ DIMENSION MISMATCH!")
                print(f"     Cohere embedding: {embedding_dim}")
                print(f"     Qdrant collection: {vector_size}")
                print(f"     This will cause search failures!")
                return False
        except Exception as e:
            print(f"  ✗ Error generating embedding: {e}")
            return False

        # Test Qdrant search
        print(f"\n[4] Testing Qdrant similarity search...")
        try:
            test_queries = [
                "What is ROS 2?",
                "Explain NVIDIA Isaac Sim",
                "What are VLAs in robotics?"
            ]

            for query in test_queries:
                print(f"\n  Query: '{query}'")
                query_embedding = conn.embed([query])[0]
                results = conn.qdrant_search(query_embedding, top_k=3)

                if len(results) == 0:
                    print(f"    ✗ No results found")
                else:
                    print(f"    ✓ Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"      {i}. Score: {result['relevance_score']:.3f}")
                        print(f"         Section: {result.get('section', 'N/A')}")
                        print(f"         File: {result.get('file_path', 'N/A')}")
                        content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                        print(f"         Content: {content_preview}")
        except Exception as e:
            print(f"  ✗ Error testing search: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test RAG agent
        print(f"\n[5] Testing RAG agent end-to-end...")
        try:
            from src.app import book_rag_agent

            test_message = "What are the key features of ROS 2?"
            print(f"  Query: '{test_message}'")

            result = book_rag_agent.run(test_message)

            print(f"  ✓ Agent response received")
            print(f"    Answer length: {len(result['answer'])} characters")
            print(f"    Sources: {len(result.get('sources', []))}")
            print(f"    Context used: {result.get('context_used', False)}")
            print(f"\n  Answer preview:")
            print(f"    {result['answer'][:200]}...")

            if result.get('sources'):
                print(f"\n  Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"    {i}. {source.get('section', 'N/A')} (score: {source.get('relevance_score', 0):.3f})")
        except Exception as e:
            print(f"  ✗ Error testing RAG agent: {e}")
            import traceback
            traceback.print_exc()
            return False

        print("\n" + "=" * 60)
        print("✓ ALL CHECKS PASSED - Qdrant is working correctly")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_qdrant()
    sys.exit(0 if success else 1)
