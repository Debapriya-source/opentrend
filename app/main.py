"""Main FastAPI application."""

from collections.abc import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from app.core.config import settings
from app.core.logging import setup_logging
from app.database.connection import create_db_and_tables, test_connections
from app.api.v1.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting OpenTrend AI application...")
    
    # Setup logging
    setup_logging()
    
    # Test database connections (optional for development)
    try:
        if test_connections():
            logger.info("All database connections successful")
        else:
            logger.warning("Some database connections failed, but continuing for development")
    except Exception as e:
        logger.warning(f"Database connection test failed, but continuing for development: {e}")
    
    # Create database tables
    try:
        create_db_and_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.warning(f"Failed to create database tables, but continuing for development: {e}")
    
    logger.info("OpenTrend AI application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenTrend AI application...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered platform to automatically identify, analyze, and predict market trends using various data sources",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Configure appropriately for production
)

# Include API routes
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "message": "Welcome to OpenTrend AI",
        "version": settings.app_version,
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    try:
        # Test database connections
        if test_connections():
            return {
                "status": "healthy",
                "database": "connected",
                "timestamp": "2024-01-01T00:00:00Z",
            }
        else:
            raise HTTPException(status_code=503, detail="Database connection failed")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # For development, return a more informative error
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z",
        }


@app.get("/info")
async def info() -> dict:
    """Application information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "AI-powered market trend analysis platform",
        "features": [
            "Automatic trend discovery",
            "Market data analysis",
            "Sentiment analysis",
            "Price predictions",
            "Real-time insights",
        ],
    }

