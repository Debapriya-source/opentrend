"""Visualization endpoints for creating charts and plots."""

from typing import Any, List, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.deps import get_current_active_user
from app.services.visualization_service import visualization_service
from app.database.models import User

router = APIRouter()


@router.get("/price-chart/{symbol}")
def create_price_chart(
    symbol: str,
    days_back: int = Query(default=90, ge=7, le=365, description="Days of historical data"),
    chart_type: str = Query(default="candlestick", regex="^(candlestick|line|ohlc)$"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create price chart for a symbol."""
    try:
        symbol = symbol.upper()

        result = visualization_service.create_price_chart(symbol, days_back, chart_type)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

        return {"message": f"{chart_type.title()} chart created successfully", "visualization": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating price chart: {str(e)}"
        )


@router.get("/technical-analysis/{symbol}")
def create_technical_analysis_chart(
    symbol: str,
    days_back: int = Query(default=90, ge=30, le=365, description="Days of historical data"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create comprehensive technical analysis chart."""
    try:
        symbol = symbol.upper()

        result = visualization_service.create_technical_analysis_chart(symbol, days_back)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

        return {"message": "Technical analysis chart created successfully", "visualization": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating technical analysis chart: {str(e)}",
        )


@router.post("/forecast-chart")
def create_forecast_visualization(
    forecast_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create visualization for forecast results."""
    try:
        result = visualization_service.create_forecast_visualization(forecast_data)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

        return {"message": "Forecast visualization created successfully", "visualization": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating forecast visualization: {str(e)}"
        )


@router.post("/sentiment-chart")
def create_sentiment_analysis_chart(
    sentiment_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create visualization for sentiment analysis results."""
    try:
        result = visualization_service.create_sentiment_analysis_chart(sentiment_data)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

        return {"message": "Sentiment analysis chart created successfully", "visualization": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating sentiment chart: {str(e)}"
        )


@router.get("/correlation-matrix")
def create_correlation_matrix(
    symbols: List[str] = Query(..., description="List of symbols for correlation analysis"),
    days_back: int = Query(default=90, ge=30, le=365, description="Days of historical data"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create correlation matrix for multiple symbols."""
    try:
        symbols = [symbol.upper() for symbol in symbols[:20]]  # Limit to 20 symbols

        result = visualization_service.create_correlation_matrix(symbols, days_back)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

        return {"message": "Correlation matrix created successfully", "visualization": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating correlation matrix: {str(e)}"
        )


@router.post("/portfolio-performance")
def create_portfolio_performance_chart(
    portfolio_data: Dict[str, float],
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create portfolio performance visualization.

    portfolio_data should be a dict with symbol as key and weight as value.
    Example: {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3}
    """
    try:
        # Validate portfolio weights sum to 1.0 (approximately)
        total_weight = sum(portfolio_data.values())
        if abs(total_weight - 1.0) > 0.01:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Portfolio weights must sum to 1.0, got {total_weight}"
            )

        result = visualization_service.create_portfolio_performance_chart(portfolio_data)

        if "error" in result:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=result["error"])

        return {"message": "Portfolio performance chart created successfully", "visualization": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating portfolio chart: {str(e)}"
        )


@router.get("/dashboard/{symbol}")
def create_symbol_dashboard(
    symbol: str,
    days_back: int = Query(default=90, ge=30, le=365, description="Days of historical data"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create comprehensive dashboard for a symbol with multiple visualizations."""
    try:
        symbol = symbol.upper()

        dashboard_components = {}

        # Price chart
        try:
            price_chart = visualization_service.create_price_chart(symbol, days_back, "candlestick")
            if "error" not in price_chart:
                dashboard_components["price_chart"] = price_chart
        except Exception as e:
            dashboard_components["price_chart"] = {"error": str(e)}

        # Technical analysis
        try:
            tech_chart = visualization_service.create_technical_analysis_chart(symbol, days_back)
            if "error" not in tech_chart:
                dashboard_components["technical_analysis"] = tech_chart
        except Exception as e:
            dashboard_components["technical_analysis"] = {"error": str(e)}

        # Check if any component was successful
        successful_components = [name for name, component in dashboard_components.items() if "error" not in component]

        if not successful_components:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create any dashboard components"
            )

        return {
            "message": f"Dashboard created successfully for {symbol}",
            "symbol": symbol,
            "dashboard_components": dashboard_components,
            "successful_components": successful_components,
            "failed_components": len(dashboard_components) - len(successful_components),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error creating dashboard: {str(e)}"
        )


@router.get("/chart-types")
def get_available_chart_types(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get information about available chart types and their capabilities."""
    try:
        chart_types = {
            "price_charts": {
                "candlestick": {
                    "description": "OHLC candlestick chart showing price action",
                    "best_for": "Detailed price analysis, pattern recognition",
                    "data_required": "OHLC data",
                },
                "line": {
                    "description": "Simple line chart of closing prices",
                    "best_for": "Trend visualization, clean presentation",
                    "data_required": "Close prices",
                },
                "ohlc": {
                    "description": "Open-High-Low-Close bar chart",
                    "best_for": "Professional trading analysis",
                    "data_required": "OHLC data",
                },
            },
            "analysis_charts": {
                "technical_analysis": {
                    "description": "Comprehensive chart with technical indicators",
                    "includes": ["RSI", "MACD", "Moving Averages", "Volume"],
                    "best_for": "Technical analysis, trading decisions",
                },
                "correlation_matrix": {
                    "description": "Heat map showing correlation between symbols",
                    "best_for": "Portfolio diversification, risk analysis",
                },
                "sentiment_analysis": {
                    "description": "Pie chart of news sentiment distribution",
                    "best_for": "Understanding market sentiment",
                },
            },
            "forecast_charts": {
                "forecast_visualization": {
                    "description": "Historical prices with forecast projections",
                    "includes": ["Confidence intervals", "Multiple models"],
                    "best_for": "Future price predictions",
                }
            },
            "portfolio_charts": {
                "portfolio_performance": {
                    "description": "Portfolio composition and performance analysis",
                    "includes": ["Asset allocation", "Performance metrics"],
                    "best_for": "Portfolio management",
                }
            },
        }

        return {
            "available_chart_types": chart_types,
            "visualization_libraries": {
                "plotly": "Interactive web-based charts",
                "matplotlib": "Static publication-quality charts",
                "seaborn": "Statistical data visualization",
                "altair": "Grammar of graphics charts",
            },
            "supported_formats": ["JSON (Plotly)", "PNG", "SVG", "HTML"],
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error getting chart types: {str(e)}"
        )
