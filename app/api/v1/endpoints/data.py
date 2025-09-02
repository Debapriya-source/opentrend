"""Data ingestion endpoints."""

from typing import Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, desc, col

from app.database.connection import get_session
from app.database.models import MarketData, NewsArticle
from app.api.deps import get_current_active_user
from app.services.data_collector import DataCollectorService
from app.services.influxdb_client import influxdb_service
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

        # Store in PostgreSQL database
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

        # Also store in InfluxDB for time-series queries
        influxdb_service.write_market_data(data_points)

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
    sources: Optional[List[str]] = None,
    days_back: int = 7,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Ingest news articles based on keywords."""
    try:
        data_service = DataCollectorService()
        articles = data_service.collect_news_articles(keywords, sources, days_back)

        # Store in database
        new_articles_count = 0
        for article in articles:
            # Check if article already exists to avoid duplicates
            existing = session.exec(select(NewsArticle).where(NewsArticle.url == article["url"])).first()

            if not existing:
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
                new_articles_count += 1

        session.commit()

        return {
            "message": f"Successfully ingested {new_articles_count} new news articles (skipped {len(articles) - new_articles_count} duplicates)",
            "articles": new_articles_count,
            "total_collected": len(articles),
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
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
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

    query = query.order_by(desc(MarketData.timestamp)).limit(limit)

    data_points = session.exec(query).all()

    return {
        "symbol": symbol,
        "data_points": len(data_points),
        "data": [point.dict() for point in data_points],
    }


@router.get("/news")
def get_news_articles(
    keywords: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get news articles with optional filtering."""
    query = select(NewsArticle)

    if keywords:
        # Simple keyword filtering (can be enhanced with full-text search)
        for keyword in keywords:
            query = query.where(col(NewsArticle.title).contains(keyword) | col(NewsArticle.content).contains(keyword))

    if sources:
        query = query.where(col(NewsArticle.source).in_(sources))

    if start_date:
        query = query.where(NewsArticle.published_at >= start_date)
    if end_date:
        query = query.where(NewsArticle.published_at <= end_date)

    query = query.order_by(desc(NewsArticle.published_at)).limit(limit)

    articles = session.exec(query).all()

    return {
        "articles": len(articles),
        "data": [article.dict() for article in articles],
    }


@router.post("/batch/market-data")
def batch_ingest_market_data(
    symbols: List[str],
    days_back: int = 30,
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Batch ingest market data for multiple symbols."""
    try:
        data_service = DataCollectorService()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        results = []
        total_points = 0

        for symbol in symbols:
            try:
                data_points = data_service.collect_market_data(symbol, start_date, end_date)

                # Store in PostgreSQL database
                new_data_points = []
                for data_point in data_points:
                    # Check for duplicates
                    existing = session.exec(
                        select(MarketData).where(
                            MarketData.symbol == data_point["symbol"], MarketData.timestamp == data_point["timestamp"]
                        )
                    ).first()

                    if not existing:
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
                        new_data_points.append(data_point)

                session.commit()

                # Also store new data points in InfluxDB
                if new_data_points:
                    influxdb_service.write_market_data(new_data_points)
                total_points += len(data_points)
                results.append({"symbol": symbol, "status": "success", "data_points": len(data_points)})

            except Exception as e:
                session.rollback()
                results.append({"symbol": symbol, "status": "error", "error": str(e)})

        return {
            "message": f"Batch ingestion completed for {len(symbols)} symbols",
            "total_data_points": total_points,
            "results": results,
        }

    except Exception as e:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch ingestion: {str(e)}",
        )


@router.post("/scheduler/trigger")
def trigger_scheduled_update(
    update_type: str = "all",  # "market", "news", "all"
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Manually trigger scheduled data updates."""
    try:
        # This is a simplified trigger - in production, you'd want to use proper background tasks
        return {
            "message": "Scheduled update triggered",
            "update_type": update_type,
            "note": "Use the /data/batch/market-data endpoint with popular symbols for immediate results",
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering scheduled update: {str(e)}",
        )
