"""
Vercel serverless adapter for FastAPI application
"""
from src.app import app
from mangum import Mangum

handler = Mangum(app, lifespan="off")
