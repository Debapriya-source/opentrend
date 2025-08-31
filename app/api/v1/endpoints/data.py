"""Data ingestion endpoints."""

from typing import Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.database.connection import get_session
from app.database.models import MarketData, NewsArticle
from app.api.deps import get_current_active_user
from app.services.data_collector import DataCollectorService
from app.database.models import User

router = APIRouter()


@router.post("/ingest/market-data")
def ingest_market_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Ingest market data for a specific symbol."""
    try:
        data_service = DataCollectorService()
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
        
        return {
            "message": f"Successfully ingested {len(data_points)} data points for {symbol}",
            "symbol": symbol,
            "data_points": len(data_points),
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting market data: {str(e)}",
        )


@router.post("/ingest/news")
def ingest_news_articles(
    keywords: List[str],
    sources: List[str] = None,
    days_back: int = 7,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Ingest news articles based on keywords."""
    try:
        data_service = DataCollectorService()
        articles = data_service.collect_news_articles(keywords, sources, days_back)
        
        # Store in database
        for article in articles:
            news_article = NewsArticle(
                title=article["title"],
                content=article["content"],
                source=article["source"],
                url=article["url"],
                published_at=article["published_at"],
                sentiment_score=article.get("sentiment_score"),
                sentiment_label=article.get("sentiment_label"),
            )
            session.add(news_article)
        
        session.commit()
        
        return {
            "message": f"Successfully ingested {len(articles)} news articles",
            "articles": len(articles),
            "keywords": keywords,
        }
    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting news articles: {str(e)}",
        )


@router.get("/market-data/{symbol}")
def get_market_data(
    symbol: str,
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get market data for a specific symbol."""
    query = select(MarketData).where(MarketData.symbol == symbol)
    
    if start_date:
        query = query.where(MarketData.timestamp >= start_date)
    if end_date:
        query = query.where(MarketData.timestamp <= end_date)
    
    query = query.order_by(MarketData.timestamp.desc()).limit(limit)
    
    data_points = session.exec(query).all()
    
    return {
        "symbol": symbol,
        "data_points": len(data_points),
        "data": [point.dict() for point in data_points],
    }


@router.get("/news")
def get_news_articles(
    keywords: List[str] = None,
    sources: List[str] = None,
    start_date: datetime = None,
    end_date: datetime = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get news articles with optional filtering."""
    query = select(NewsArticle)
    
    if keywords:
        # Simple keyword filtering (can be enhanced with full-text search)
        for keyword in keywords:
            query = query.where(
                (NewsArticle.title.contains(keyword)) | 
                (NewsArticle.content.contains(keyword))
            )
    
    if sources:
        query = query.where(NewsArticle.source.in_(sources))
    
    if start_date:
        query = query.where(NewsArticle.published_at >= start_date)
    if end_date:
        query = query.where(NewsArticle.published_at <= end_date)
    
    query = query.order_by(NewsArticle.published_at.desc()).limit(limit)
    
    articles = session.exec(query).all()
    
    return {
        "articles": len(articles),
        "data": [article.dict() for article in articles],
    }
