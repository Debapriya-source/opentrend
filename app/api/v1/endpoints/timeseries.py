"""Time-series data endpoints using InfluxDB."""

from typing import Any, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.deps import get_current_active_user
from app.services.influxdb_client import influxdb_service
from app.database.models import User

router = APIRouter()


@router.get("/market-data/{symbol}")
def get_market_data_timeseries(
    symbol: str,
    start_time: datetime = Query(..., description="Start time for data range"),
    end_time: datetime = Query(..., description="End time for data range"),
    aggregation_window: str = Query(default="1h", description="Aggregation window (e.g., '1m', '5m', '1h', '1d')"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get time-series market data for a specific symbol from InfluxDB."""
    try:
        if not influxdb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InfluxDB service is not available",
            )

        data_points = influxdb_service.query_market_data(
            symbol=symbol.upper(), start_time=start_time, end_time=end_time, aggregation_window=aggregation_window
        )

        return {
            "symbol": symbol.upper(),
            "start_time": start_time,
            "end_time": end_time,
            "aggregation_window": aggregation_window,
            "data_points": len(data_points),
            "data": data_points,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market data: {str(e)}",
        )


@router.get("/market-data/multiple")
def get_multiple_symbols_timeseries(
    symbols: List[str] = Query(..., description="List of symbols to retrieve"),
    start_time: datetime = Query(..., description="Start time for data range"),
    end_time: datetime = Query(..., description="End time for data range"),
    aggregation_window: str = Query(default="1h", description="Aggregation window (e.g., '1m', '5m', '1h', '1d')"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get time-series market data for multiple symbols from InfluxDB."""
    try:
        if not influxdb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InfluxDB service is not available",
            )

        symbols_upper = [symbol.upper() for symbol in symbols]
        symbol_data = influxdb_service.query_multiple_symbols(
            symbols=symbols_upper, start_time=start_time, end_time=end_time, aggregation_window=aggregation_window
        )

        return {
            "symbols": symbols_upper,
            "start_time": start_time,
            "end_time": end_time,
            "aggregation_window": aggregation_window,
            "data": symbol_data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving market data for multiple symbols: {str(e)}",
        )


@router.get("/market-data/{symbol}/latest")
def get_latest_market_data(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the latest market data point for a symbol from InfluxDB."""
    try:
        if not influxdb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InfluxDB service is not available",
            )

        latest_data = influxdb_service.get_latest_data_point(symbol.upper())

        if not latest_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No data found for symbol {symbol.upper()}",
            )

        return {
            "symbol": symbol.upper(),
            "latest_data": latest_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving latest market data: {str(e)}",
        )


@router.get("/market-data/{symbol}/ohlcv")
def get_ohlcv_data(
    symbol: str,
    start_time: datetime = Query(..., description="Start time for data range"),
    end_time: datetime = Query(..., description="End time for data range"),
    aggregation_window: str = Query(default="1d", description="Aggregation window for OHLCV data"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get OHLCV (Open, High, Low, Close, Volume) data optimized for charting."""
    try:
        if not influxdb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InfluxDB service is not available",
            )

        data_points = influxdb_service.query_market_data(
            symbol=symbol.upper(), start_time=start_time, end_time=end_time, aggregation_window=aggregation_window
        )

        # Format data for charting libraries
        ohlcv_data = []
        for point in data_points:
            if all(point.get(field) is not None for field in ["open", "high", "low", "close", "volume"]):
                ohlcv_data.append({
                    "timestamp": point["timestamp"],
                    "open": float(point["open"]),
                    "high": float(point["high"]),
                    "low": float(point["low"]),
                    "close": float(point["close"]),
                    "volume": int(point["volume"]),
                })

        return {
            "symbol": symbol.upper(),
            "start_time": start_time,
            "end_time": end_time,
            "aggregation_window": aggregation_window,
            "data_points": len(ohlcv_data),
            "ohlcv": ohlcv_data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving OHLCV data: {str(e)}",
        )


@router.get("/market-data/compare")
def compare_symbols(
    symbols: List[str] = Query(..., description="List of symbols to compare"),
    start_time: datetime = Query(..., description="Start time for data range"),
    end_time: datetime = Query(..., description="End time for data range"),
    field: str = Query(default="close", description="Field to compare (open, high, low, close)"),
    aggregation_window: str = Query(default="1d", description="Aggregation window"),
    normalize: bool = Query(default=False, description="Normalize values to percentage change"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Compare multiple symbols on a specific field (e.g., close price)."""
    try:
        if not influxdb_service.is_connected():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="InfluxDB service is not available",
            )

        if field not in ["open", "high", "low", "close"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field must be one of: open, high, low, close",
            )

        symbols_upper = [symbol.upper() for symbol in symbols]
        symbol_data = influxdb_service.query_multiple_symbols(
            symbols=symbols_upper, start_time=start_time, end_time=end_time, aggregation_window=aggregation_window
        )

        # Extract comparison data
        comparison_data = {}
        for symbol, data_points in symbol_data.items():
            values = []
            for point in data_points:
                if point.get(field) is not None:
                    value = float(point[field])
                    values.append({
                        "timestamp": point["timestamp"],
                        "value": value,
                    })

            # Normalize to percentage change if requested
            if normalize and values:
                base_value = values[0]["value"]
                for item in values:
                    item["value"] = ((item["value"] - base_value) / base_value) * 100

            comparison_data[symbol] = values

        return {
            "symbols": symbols_upper,
            "field": field,
            "start_time": start_time,
            "end_time": end_time,
            "aggregation_window": aggregation_window,
            "normalized": normalize,
            "comparison_data": comparison_data,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing symbols: {str(e)}",
        )


@router.get("/health")
def influxdb_health_check(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Check InfluxDB connection health."""
    try:
        is_connected = influxdb_service.is_connected()

        return {
            "influxdb_connected": is_connected,
            "status": "healthy" if is_connected else "unhealthy",
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        return {
            "influxdb_connected": False,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow(),
        }
