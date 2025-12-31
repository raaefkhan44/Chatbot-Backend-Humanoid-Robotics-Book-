"""
Lightweight Qdrant REST API client to avoid heavy grpcio and numpy dependencies
"""
import httpx
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class QdrantRestClient:
    """Lightweight Qdrant client using REST API only"""

    def __init__(self, url: str, api_key: Optional[str] = None):
        self.url = url.rstrip('/')
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["api-key"] = api_key

    def query_points(
        self,
        collection_name: str,
        query: List[float],
        limit: int = 5,
        with_payload: bool = True
    ) -> Dict[str, Any]:
        """Search for similar vectors using REST API"""
        endpoint = f"{self.url}/collections/{collection_name}/points/query"

        payload = {
            "query": query,
            "limit": limit,
            "with_payload": with_payload
        }

        try:
            response = httpx.post(endpoint, json=payload, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            result = response.json()

            # Format response to match qdrant-client structure
            points = []
            for point in result.get("result", {}).get("points", []):
                points.append({
                    "id": point.get("id"),
                    "score": point.get("score", 0),
                    "payload": point.get("payload", {})
                })

            return {"points": points}
        except Exception as e:
            logger.error(f"Qdrant query error: {e}")
            raise

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        with_payload: bool = True
    ) -> List[Dict[str, Any]]:
        """Legacy search method for compatibility"""
        endpoint = f"{self.url}/collections/{collection_name}/points/search"

        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": with_payload
        }

        try:
            response = httpx.post(endpoint, json=payload, headers=self.headers, timeout=30.0)
            response.raise_for_status()
            result = response.json()

            # Format response
            points = []
            for point in result.get("result", []):
                class SearchResult:
                    def __init__(self, data):
                        self.id = data.get("id")
                        self.score = data.get("score", 0)
                        self.payload = data.get("payload", {})

                points.append(SearchResult(point))

            return points
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            raise
