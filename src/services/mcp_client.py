import requests
from typing import Optional, Dict, Any
from ..config.settings import settings
import logging

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for interacting with Context7 MCP Server to fetch OpenAI Agent SDK documentation
    """

    def __init__(self):
        if not settings.CONTEXT7_MCP_SERVER_URL:
            logger.warning("Context7 MCP Server URL not configured. Documentation features may be limited.")

    def fetch_openai_agent_docs(self, topic: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Fetch OpenAI Agent SDK documentation from Context7 MCP Server
        """
        if not settings.context7_mcp_server_url:
            logger.error("Context7 MCP Server URL not configured")
            return None

        try:
            # This is a placeholder implementation - actual MCP protocol implementation
            # would require more specific knowledge of the Context7 server interface
            url = f"{settings.context7_mcp_server_url}/docs/openai-agent"

            params = {}
            if topic:
                params["topic"] = topic

            response = requests.get(url, params=params)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            logger.error(f"Error fetching OpenAI Agent docs: {str(e)}")
            return None

    def get_agent_api_reference(self) -> Optional[Dict[str, Any]]:
        """
        Get OpenAI Agent API reference documentation
        """
        return self.fetch_openai_agent_docs(topic="api-reference")

    def get_agent_code_patterns(self) -> Optional[Dict[str, Any]]:
        """
        Get recommended code patterns for OpenAI Agent implementation
        """
        return self.fetch_openai_agent_docs(topic="code-patterns")

    def get_agent_tool_definitions(self) -> Optional[Dict[str, Any]]:
        """
        Get information about tool definitions for OpenAI Agents
        """
        return self.fetch_openai_agent_docs(topic="tools")

    def get_agent_best_practices(self) -> Optional[Dict[str, Any]]:
        """
        Get recommended best practices for OpenAI Agent implementation
        """
        return self.fetch_openai_agent_docs(topic="best-practices")


# Global instance of MCPClient
mcp_client = MCPClient()