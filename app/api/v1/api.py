"""Main API router for v1 endpoints."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, data, analysis

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])

