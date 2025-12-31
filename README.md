# RAG Chatbot Backend

This is the backend service for the RAG (Retrieval-Augmented Generation) Chatbot that integrates with the Docusaurus-based book project.

## Features

- Full-book RAG queries: Ask questions about the entire book content
- Selected-text QA: Ask questions about specific text selections
- Embedding management: Regenerate embeddings when book content changes
- Logging: Track all user interactions for analytics
- Rate limiting: Prevent API abuse
- Request size limits: Prevent large payload attacks

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL (Neon Serverless) for logs and metadata
- **Vector Database**: Qdrant Cloud for embeddings
- **AI**: OpenAI API for embeddings and completions
- **Language**: Python 3.11

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with the following environment variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   NEON_DATABASE_URL=postgresql+asyncpg://username:password@ep-xxx.us-east-1.aws.neon.tech/dbname?sslmode=require
   CONTEXT7_MCP_SERVER_URL=your_context7_mcp_server_url
   API_KEY=optional_api_key_for_authentication
   ```

3. Run the application:
   ```bash
   python -m src.api.main
   ```

## API Endpoints

See [API Documentation](docs/api.md) for detailed endpoint information.

## Architecture

The system is organized into the following modules:

- **Models**: SQLAlchemy database models
- **Services**: Business logic for RAG, embedding, and agent operations
- **Utils**: Helper functions for document parsing, text chunking, and validation
- **Routes**: FastAPI endpoints
- **Config**: Application settings and database configuration

## Running Tests

```bash
pytest tests/
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `NEON_DATABASE_URL`: Connection string for Neon PostgreSQL database
- `CONTEXT7_MCP_SERVER_URL`: URL for Context7 MCP Server (optional)
- `API_KEY`: Optional API key for authentication (if enabled) "# Chatbot-Backend-Humanoid-Robotics-Book-" 
