"""Main API router for v1 endpoints."""

from fastapi import APIRouter

from app.api.v1.endpoints import auth, data, analysis, timeseries, llm_insights, forecasting, visualization

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(timeseries.router, prefix="/timeseries", tags=["timeseries"])
api_router.include_router(llm_insights.router, prefix="/llm", tags=["llm-insights"])
api_router.include_router(forecasting.router, prefix="/forecasting", tags=["forecasting"])
api_router.include_router(visualization.router, prefix="/visualization", tags=["visualization"])
