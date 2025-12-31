# Test script to validate frontend-backend integration

import requests
import time
import json

# Base URL for the backend API
BASE_URL = "http://localhost:8000/api"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Health check passed")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_query():
    """Test the query endpoint"""
    try:
        payload = {
            "question": "What is this book about?"
        }
        response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "answer" in data:
                print("✓ Query endpoint test passed")
                return True
            else:
                print(f"✗ Query endpoint missing answer: {data}")
                return False
        else:
            print(f"✗ Query endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Query endpoint error: {e}")
        return False

def test_select_query():
    """Test the select-query endpoint"""
    try:
        payload = {
            "question": "What does this text mean?",
            "selected_text": "This is a sample text selection for testing purposes."
        }
        response = requests.post(f"{BASE_URL}/select-query", json=payload)
        if response.status_code == 200:
            data = response.json()
            if "answer" in data:
                print("✓ Select-query endpoint test passed")
                return True
            else:
                print(f"✗ Select-query endpoint missing answer: {data}")
                return False
        else:
            print(f"✗ Select-query endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Select-query endpoint error: {e}")
        return False

def test_embedding_endpoints():
    """Test the embedding endpoints"""
    try:
        # Test embeddings count
        response = requests.get(f"{BASE_URL}/embeddings/count")
        if response.status_code == 200:
            data = response.json()
            if "count" in data:
                print("✓ Embeddings count endpoint test passed")
                print(f"  Current embedding count: {data['count']}")
            else:
                print(f"✗ Embeddings count endpoint missing count: {data}")
        else:
            print(f"✗ Embeddings count endpoint failed: {response.status_code} - {response.text}")

        # Test embed endpoint (POST request to trigger embedding)
        response = requests.post(f"{BASE_URL}/embed")
        if response.status_code in [200, 202]:
            print("✓ Embed endpoint test passed")
            return True
        else:
            print(f"✗ Embed endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Embedding endpoints error: {e}")
        return False

def test_logs_endpoint():
    """Test the logs endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/logs?limit=5")
        if response.status_code == 200:
            data = response.json()
            if "logs" in data:
                print("✓ Logs endpoint test passed")
                return True
            else:
                print(f"✗ Logs endpoint missing logs: {data}")
                return False
        else:
            print(f"✗ Logs endpoint failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"✗ Logs endpoint error: {e}")
        return False

def main():
    print("Testing RAG Chatbot Backend Integration...")
    print("="*50)

    # Wait a bit to ensure the server is running
    time.sleep(2)

    all_tests_passed = True

    print("\n1. Testing health endpoint...")
    all_tests_passed &= test_health()

    print("\n2. Testing query endpoint...")
    all_tests_passed &= test_query()

    print("\n3. Testing select-query endpoint...")
    all_tests_passed &= test_select_query()

    print("\n4. Testing embedding endpoints...")
    all_tests_passed &= test_embedding_endpoints()

    print("\n5. Testing logs endpoint...")
    all_tests_passed &= test_logs_endpoint()

    print("\n" + "="*50)
    if all_tests_passed:
        print("✓ All integration tests passed!")
    else:
        print("✗ Some tests failed")

    return all_tests_passed

if __name__ == "__main__":
    main()