"""Analysis endpoints for trend analysis and predictions."""

from typing import Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.database.connection import get_session
from app.database.models import TrendAnalysis, Prediction, User
from app.api.deps import get_current_active_user
from app.services.analysis_service import AnalysisService

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
        
        # Store analysis result
        trend_analysis = TrendAnalysis(
            symbol=analysis_result["symbol"],
            trend_type=analysis_result["trend_type"],
            confidence_score=analysis_result["confidence_score"],
            timeframe=timeframe,
            analysis_date=datetime.utcnow(),
            indicators=analysis_result["indicators"],
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
        analysis_service = AnalysisService()
        prediction_result = analysis_service.generate_prediction(
            symbol, prediction_type, horizon_days
        )
        
        # Store prediction result
        prediction = Prediction(
            symbol=prediction_result["symbol"],
            prediction_type=prediction_type,
            predicted_value=prediction_result["predicted_value"],
            confidence_interval_lower=prediction_result["confidence_interval_lower"],
            confidence_interval_upper=prediction_result["confidence_interval_upper"],
            prediction_date=datetime.utcnow(),
            model_used=prediction_result["model_used"],
            features_used=prediction_result["features_used"],
        )
        session.add(prediction)
        session.commit()
        
        return {
            "message": "Prediction generated successfully",
            "prediction": prediction_result,
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating prediction: {str(e)}",
        )


@router.get("/trends")
def get_trend_analyses(
    symbol: str = None,
    trend_type: str = None,
    timeframe: str = None,
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
    
    query = query.order_by(TrendAnalysis.analysis_date.desc()).limit(limit)
    
    analyses = session.exec(query).all()
    
    return {
        "analyses": len(analyses),
        "data": [analysis.dict() for analysis in analyses],
    }


@router.get("/predictions")
def get_predictions(
    symbol: str = None,
    prediction_type: str = None,
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
    
    query = query.order_by(Prediction.prediction_date.desc()).limit(limit)
    
    predictions = session.exec(query).all()
    
    return {
        "predictions": len(predictions),
        "data": [prediction.dict() for prediction in predictions],
    }


@router.get("/sentiment/current")
def get_current_sentiment(
    symbol: str = None,
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

