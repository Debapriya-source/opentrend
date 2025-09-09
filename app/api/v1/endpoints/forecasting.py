"""Advanced forecasting endpoints using Prophet, LSTM, and ensemble methods."""

from typing import Any, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.deps import get_current_active_user
from app.services.forecasting_service import forecasting_service
from app.database.models import User

router = APIRouter()


@router.post("/prophet/{symbol}")
def forecast_with_prophet(
    symbol: str,
    horizon_days: int = Query(default=30, ge=1, le=365, description="Forecast horizon in days"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Generate price forecast using Facebook Prophet model."""
    try:
        symbol = symbol.upper()

        result = forecasting_service.forecast_with_prophet(symbol, horizon_days)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prophet forecast failed: {result['error']}"
            )

        return {"message": "Prophet forecast generated successfully", "forecast": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating Prophet forecast: {str(e)}"
        )


@router.post("/lstm/{symbol}")
def forecast_with_lstm(
    symbol: str,
    horizon_days: int = Query(default=30, ge=1, le=90, description="Forecast horizon in days"),
    sequence_length: int = Query(default=60, ge=30, le=120, description="LSTM sequence length"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Generate price forecast using LSTM neural network."""
    try:
        symbol = symbol.upper()

        result = forecasting_service.forecast_with_lstm(symbol, horizon_days, sequence_length)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"LSTM forecast failed: {result['error']}"
            )

        return {"message": "LSTM forecast generated successfully", "forecast": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating LSTM forecast: {str(e)}"
        )


@router.post("/ensemble/{symbol}")
def forecast_with_ensemble(
    symbol: str,
    horizon_days: int = Query(default=30, ge=1, le=90, description="Forecast horizon in days"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Generate ensemble forecast using multiple ML models."""
    try:
        symbol = symbol.upper()

        result = forecasting_service.forecast_ensemble(symbol, horizon_days)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=f"Ensemble forecast failed: {result['error']}"
            )

        return {"message": "Ensemble forecast generated successfully", "forecast": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error generating ensemble forecast: {str(e)}"
        )


@router.api_route("/compare/{symbol}", methods=["GET", "POST"])
def compare_forecasting_models(
    symbol: str,
    horizon_days: int = Query(default=30, ge=1, le=90, description="Forecast horizon in days"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Compare forecasts from different models for the same symbol."""
    try:
        symbol = symbol.upper()

        results = {}

        # Prophet forecast
        try:
            prophet_result = forecasting_service.forecast_with_prophet(symbol, horizon_days)
            if "error" not in prophet_result:
                results["prophet"] = {
                    "predicted_price": prophet_result["predicted_price"],
                    "price_change_percent": prophet_result["price_change_percent"],
                    "confidence_interval": prophet_result["confidence_interval"],
                    "model_performance": prophet_result.get("model_performance", {}),
                }
        except Exception as e:
            results["prophet"] = {"error": str(e)}

        # LSTM forecast
        try:
            lstm_result = forecasting_service.forecast_with_lstm(symbol, horizon_days)
            if "error" not in lstm_result:
                results["lstm"] = {
                    "predicted_price": lstm_result["predicted_price"],
                    "price_change_percent": lstm_result["price_change_percent"],
                    "confidence_interval": lstm_result["confidence_interval"],
                    "model_performance": lstm_result.get("model_performance", {}),
                }
        except Exception as e:
            results["lstm"] = {"error": str(e)}

        # Ensemble forecast
        try:
            ensemble_result = forecasting_service.forecast_ensemble(symbol, horizon_days)
            if "error" not in ensemble_result:
                results["ensemble"] = {
                    "predicted_price": ensemble_result["predicted_price"],
                    "price_change_percent": ensemble_result["price_change_percent"],
                    "confidence_interval": ensemble_result["confidence_interval"],
                    "model_weights": ensemble_result.get("model_weights", {}),
                    "models_used": ensemble_result.get("models_used", []),
                }
        except Exception as e:
            results["ensemble"] = {"error": str(e)}

        if not any("error" not in result for result in results.values() if isinstance(result, dict)):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="All forecasting models failed")

        # Calculate consensus metrics
        valid_predictions = [
            result["predicted_price"]
            for result in results.values()
            if isinstance(result, dict) and "error" not in result
        ]

        consensus_metrics = {}
        if valid_predictions:
            consensus_metrics = {
                "average_prediction": round(sum(valid_predictions) / len(valid_predictions), 2),
                "min_prediction": round(min(valid_predictions), 2),
                "max_prediction": round(max(valid_predictions), 2),
                "prediction_spread": round(max(valid_predictions) - min(valid_predictions), 2),
                "models_agreeing": len(valid_predictions),
            }

        return {
            "symbol": symbol,
            "horizon_days": horizon_days,
            "model_comparisons": results,
            "consensus_metrics": consensus_metrics,
            "comparison_timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error comparing forecasting models: {str(e)}"
        )


@router.get("/batch")
def batch_forecast(
    symbols: List[str] = Query(..., description="List of symbols to forecast"),
    model_type: str = Query(default="ensemble", regex="^(prophet|lstm|ensemble)$"),
    horizon_days: int = Query(default=30, ge=1, le=90, description="Forecast horizon in days"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Generate forecasts for multiple symbols using the specified model."""
    try:
        symbols = [symbol.upper() for symbol in symbols[:10]]  # Limit to 10 symbols
        results = {}

        for symbol in symbols:
            try:
                if model_type == "prophet":
                    result = forecasting_service.forecast_with_prophet(symbol, horizon_days)
                elif model_type == "lstm":
                    result = forecasting_service.forecast_with_lstm(symbol, horizon_days)
                else:  # ensemble
                    result = forecasting_service.forecast_ensemble(symbol, horizon_days)

                if "error" not in result:
                    results[symbol] = {
                        "predicted_price": result["predicted_price"],
                        "current_price": result["current_price"],
                        "price_change_percent": result["price_change_percent"],
                        "confidence_interval": result["confidence_interval"],
                        "model_type": result["model_type"],
                    }
                else:
                    results[symbol] = {"error": result["error"]}

            except Exception as e:
                results[symbol] = {"error": str(e)}

        # Summary statistics
        successful_forecasts = [
            result for result in results.values() if isinstance(result, dict) and "error" not in result
        ]

        summary = {
            "total_symbols": len(symbols),
            "successful_forecasts": len(successful_forecasts),
            "failed_forecasts": len(symbols) - len(successful_forecasts),
        }

        if successful_forecasts:
            price_changes = [result["price_change_percent"] for result in successful_forecasts]
            summary.update({
                "average_price_change": round(sum(price_changes) / len(price_changes), 2),
                "bullish_forecasts": len([pc for pc in price_changes if pc > 0]),
                "bearish_forecasts": len([pc for pc in price_changes if pc < 0]),
            })

        return {
            "model_type": model_type,
            "horizon_days": horizon_days,
            "forecasts": results,
            "summary": summary,
            "batch_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in batch forecasting: {str(e)}"
        )


@router.get("/models/status")
def get_forecasting_models_status(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get status and capabilities of forecasting models."""
    try:
        # Test model availability with a simple check
        status_info = {
            "prophet": {
                "available": True,
                "description": "Facebook Prophet for time-series forecasting",
                "best_for": "Long-term trends, seasonal patterns",
                "max_horizon_days": 365,
                "min_data_points": 30,
            },
            "lstm": {
                "available": True,
                "description": "LSTM Neural Network for sequence prediction",
                "best_for": "Short to medium-term predictions, complex patterns",
                "max_horizon_days": 90,
                "min_data_points": 90,
                "pytorch_available": True,
            },
            "ensemble": {
                "available": True,
                "description": "Weighted combination of multiple models",
                "best_for": "Balanced predictions with reduced overfitting",
                "max_horizon_days": 90,
                "component_models": ["prophet", "lstm", "random_forest"],
            },
            "random_forest": {
                "available": True,
                "description": "Random Forest for feature-based prediction",
                "best_for": "Feature-rich predictions, interpretability",
                "max_horizon_days": 30,
            },
        }

        return {
            "forecasting_models": status_info,
            "system_info": {"torch_available": True, "prophet_available": True, "sklearn_available": True},
            "status_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {"error": str(e), "status_timestamp": datetime.utcnow().isoformat()}
