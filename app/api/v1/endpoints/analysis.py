"""Analysis endpoints for trend analysis and predictions."""

from typing import Any, Optional
from datetime import datetime, timedelta
import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, desc, asc

from app.database.connection import get_session
from app.database.models import TrendAnalysis, Prediction, User, MarketData
from app.api.deps import get_current_active_user
from app.services.analysis_service import AnalysisService
from app.services.data_collector import DataCollectorService

router = APIRouter()


@router.post("/trends/analyze")
def analyze_trends(
    symbol: str,
    timeframe: str = "medium",  # "short", "medium", "long"
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Analyze trends for a specific symbol."""
    try:
        analysis_service = AnalysisService()
        analysis_result = analysis_service.analyze_trends(symbol, timeframe)

        # Store analysis result (convert numpy types to Python types and serialize indicators)
        indicators_dict = {k: float(v) if hasattr(v, "item") else v for k, v in analysis_result["indicators"].items()}
        trend_analysis = TrendAnalysis(
            symbol=analysis_result["symbol"],
            trend_type=analysis_result["trend_type"],
            confidence_score=float(analysis_result["confidence_score"]),
            timeframe=timeframe,
            analysis_date=datetime.utcnow(),
            indicators=json.dumps(indicators_dict),
            description=analysis_result["description"],
        )
        session.add(trend_analysis)
        session.commit()

        return {
            "message": "Trend analysis completed successfully",
            "analysis": analysis_result,
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trends: {str(e)}",
        )


def ingest_data_background(symbol: str, session: Session):
    """Background task to ingest market data for a symbol."""
    try:
        data_service = DataCollectorService()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=365)  # Get 1 year of data

        data_points = data_service.collect_market_data(symbol, start_date, end_date)

        # Store in database
        for data_point in data_points:
            market_data = MarketData(
                symbol=data_point["symbol"],
                timestamp=data_point["timestamp"],
                open_price=data_point["open"],
                high_price=data_point["high"],
                low_price=data_point["low"],
                close_price=data_point["close"],
                volume=data_point["volume"],
                source=data_point["source"],
            )
            session.add(market_data)

        session.commit()

    except Exception as e:
        session.rollback()
        raise e


@router.post("/predictions/generate")
def generate_prediction(
    symbol: str,
    prediction_type: str = "price",  # "price", "trend", "volatility"
    horizon_days: int = 30,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Generate predictions for a specific symbol."""
    try:
        from loguru import logger

        logger.info(f"Starting prediction generation for symbol: {symbol}")

        # Check if we have market data for this symbol
        start_date = datetime.utcnow() - timedelta(days=365)
        query = select(MarketData).where(MarketData.symbol == symbol, MarketData.timestamp >= start_date).limit(1)
        existing_data = session.exec(query).first()

        logger.info(f"Existing data check for {symbol}: {'Found' if existing_data else 'Not found'}")

        # If no data exists, trigger background data ingestion
        if not existing_data:
            logger.info(f"No existing data found for {symbol}, starting data ingestion...")
            try:
                # Try to ingest data immediately for better user experience
                data_service = DataCollectorService()
                end_date = datetime.utcnow()
                start_date = end_date - timedelta(days=365)

                logger.info(f"Collecting market data for {symbol} from {start_date} to {end_date}")
                data_points = data_service.collect_market_data(symbol, start_date, end_date)
                logger.info(f"Collected {len(data_points) if data_points else 0} data points for {symbol}")

                if not data_points:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No market data available for symbol {symbol}. Please check if the symbol is valid.",
                    )

                # Store in database
                for data_point in data_points:
                    market_data = MarketData(
                        symbol=data_point["symbol"],
                        timestamp=data_point["timestamp"],
                        open_price=data_point["open"],
                        high_price=data_point["high"],
                        low_price=data_point["low"],
                        close_price=data_point["close"],
                        volume=data_point["volume"],
                        source=data_point["source"],
                    )
                    session.add(market_data)

                session.commit()
                logger.info(f"Successfully stored {len(data_points)} data points for {symbol}")

            except Exception as data_error:
                logger.error(f"Error during data ingestion for {symbol}: {str(data_error)}")
                session.rollback()
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error ingesting data for {symbol}: {str(data_error)}",
                )

        # Now generate the prediction
        logger.info(f"Starting prediction generation with AnalysisService for {symbol}")
        analysis_service = AnalysisService()
        prediction_result = analysis_service.generate_prediction(symbol, prediction_type, horizon_days)
        logger.info(f"Prediction generation completed for {symbol}")

        # Store prediction result (convert numpy types to Python types and serialize features)
        features_dict = {
            k: float(v) if hasattr(v, "item") else v for k, v in prediction_result["features_used"].items()
        }
        prediction = Prediction(
            symbol=prediction_result["symbol"],
            prediction_type=prediction_type,
            predicted_value=float(prediction_result["predicted_value"]),
            confidence_interval_lower=float(prediction_result["confidence_interval_lower"]),
            confidence_interval_upper=float(prediction_result["confidence_interval_upper"]),
            prediction_date=datetime.utcnow(),
            model_used=prediction_result["model_used"],
            features_used=json.dumps(features_dict),
        )
        session.add(prediction)
        session.commit()

        return {
            "message": "Prediction generated successfully",
            "prediction": prediction_result,
            "data_ingested": not existing_data,  # Indicate if data was just ingested
        }
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating prediction: {str(e)}",
        )


@router.get("/trends")
def get_trend_analyses(
    symbol: Optional[str] = None,
    trend_type: Optional[str] = None,
    timeframe: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get trend analyses with optional filtering."""
    query = select(TrendAnalysis)

    if symbol:
        query = query.where(TrendAnalysis.symbol == symbol)
    if trend_type:
        query = query.where(TrendAnalysis.trend_type == trend_type)
    if timeframe:
        query = query.where(TrendAnalysis.timeframe == timeframe)

    query = query.order_by(desc(TrendAnalysis.analysis_date)).limit(limit)

    analyses = session.exec(query).all()

    return {
        "analyses": len(analyses),
        "data": [analysis.dict() for analysis in analyses],
    }


@router.get("/predictions")
def get_predictions(
    symbol: Optional[str] = None,
    prediction_type: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get predictions with optional filtering."""
    query = select(Prediction)

    if symbol:
        query = query.where(Prediction.symbol == symbol)
    if prediction_type:
        query = query.where(Prediction.prediction_type == prediction_type)

    query = query.order_by(desc(Prediction.prediction_date)).limit(limit)

    predictions = session.exec(query).all()

    return {
        "predictions": len(predictions),
        "data": [prediction.dict() for prediction in predictions],
    }


@router.get("/market-data/{symbol}")
def get_market_data(
    symbol: str,
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get market data for a specific symbol."""
    try:
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Query market data
        query = (
            select(MarketData)
            .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date, MarketData.timestamp <= end_date)
            .order_by(asc(MarketData.timestamp))
        )

        market_data = session.exec(query).all()

        if not market_data:
            return {
                "symbol": symbol,
                "data_points": 0,
                "data": [],
                "message": f"No market data found for {symbol} in the last {days} days",
            }

        # Convert to list of dictionaries
        data_list = []
        for data_point in market_data:
            data_list.append({
                "date": data_point.timestamp.strftime("%Y-%m-%d"),
                "open": float(data_point.open_price),
                "high": float(data_point.high_price),
                "low": float(data_point.low_price),
                "close": float(data_point.close_price),
                "volume": int(data_point.volume),
                "source": data_point.source,
            })

        return {
            "symbol": symbol,
            "data_points": len(data_list),
            "data": data_list,
            "date_range": {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")},
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching market data: {str(e)}",
        )


@router.get("/sentiment/current")
def get_current_sentiment(
    symbol: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get current market sentiment."""
    try:
        analysis_service = AnalysisService()
        sentiment_result = analysis_service.get_current_sentiment(symbol)

        return {
            "sentiment": sentiment_result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting sentiment: {str(e)}",
        )


@router.post("/models/train")
def train_models(
    model_type: str = "all",  # "sentiment", "forecasting", "all"
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Trigger model retraining."""
    try:
        analysis_service = AnalysisService()
        training_result = analysis_service.train_models(model_type)

        return {
            "message": "Model training completed successfully",
            "training_result": training_result,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training models: {str(e)}",
        )
