"""LLM-powered insights and analysis endpoints."""

from typing import Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.api.deps import get_current_active_user
from app.services.llm_service import llm_service
from app.services.analysis_service import AnalysisService
from app.database.models import User
from app.database.connection import get_session
from sqlmodel import Session, select, desc, col
from app.database.models import NewsArticle, MarketData

router = APIRouter()


@router.post("/analyze/sentiment")
def analyze_news_sentiment(
    keywords: List[str],
    days_back: int = Query(default=7, ge=1, le=30, description="Days to look back for news"),
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Analyze sentiment of recent news articles using advanced LLM."""
    try:
        # Get recent news articles
        start_date = datetime.utcnow() - timedelta(days=days_back)
        query = (
            select(NewsArticle)
            .where(NewsArticle.published_at >= start_date)
            .order_by(desc(NewsArticle.published_at))
            .limit(50)  # Limit for performance
        )

        # Filter by keywords if provided
        if keywords:
            for keyword in keywords:
                query = query.where(
                    col(NewsArticle.title).contains(keyword) | col(NewsArticle.content).contains(keyword)
                )

        articles = session.exec(query).all()

        if not articles:
            return {
                "message": "No articles found matching criteria",
                "keywords": keywords,
                "days_back": days_back,
                "sentiment_analysis": None,
            }

        # Convert to format expected by LLM service
        articles_data = [
            {
                "title": article.title,
                "content": article.content[:1000],  # Limit content for performance
                "url": article.url,
                "published_at": article.published_at.isoformat(),
                "source": article.source,
            }
            for article in articles
        ]

        # Analyze sentiment using LLM
        sentiment_analysis = llm_service.analyze_news_sentiment(articles_data)

        return {
            "keywords": keywords,
            "days_back": days_back,
            "articles_analyzed": len(articles_data),
            "sentiment_analysis": sentiment_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}",
        )


@router.get("/insights/{symbol}")
def get_market_insights(
    symbol: str,
    days_back: int = Query(default=30, ge=7, le=90, description="Days of market data to analyze"),
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get comprehensive LLM-powered market insights for a symbol."""
    try:
        symbol = symbol.upper()

        # Get market data
        start_date = datetime.utcnow() - timedelta(days=days_back)
        market_query = (
            select(MarketData)
            .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date)
            .order_by(desc(MarketData.timestamp))
            .limit(100)
        )

        market_data = session.exec(market_query).all()

        if not market_data:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No market data found for {symbol}")

        # Get related news articles
        news_query = (
            select(NewsArticle)
            .where(NewsArticle.published_at >= start_date)
            .where(
                col(NewsArticle.title).contains(symbol)
                | col(NewsArticle.content).contains(symbol)
                | col(NewsArticle.title).contains(symbol.replace(".", ""))
            )
            .order_by(desc(NewsArticle.published_at))
            .limit(20)
        )

        news_articles = session.exec(news_query).all()

        # Convert to format for LLM service
        market_data_dict = [
            {
                "close": float(data.close_price),
                "open": float(data.open_price),
                "high": float(data.high_price),
                "low": float(data.low_price),
                "volume": int(data.volume),
                "timestamp": data.timestamp.isoformat(),
            }
            for data in market_data
        ]

        news_data_dict = [
            {
                "title": article.title,
                "content": article.content[:500],
                "url": article.url,
                "published_at": article.published_at.isoformat(),
                "source": article.source,
            }
            for article in news_articles
        ]

        # Analyze news sentiment
        news_sentiment = llm_service.analyze_news_sentiment(news_data_dict)

        # Generate comprehensive insights
        insights = llm_service.generate_market_insights(symbol, market_data_dict, news_sentiment)

        return {
            "symbol": symbol,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": datetime.utcnow().isoformat(),
                "days_analyzed": days_back,
            },
            "data_summary": {
                "market_data_points": len(market_data_dict),
                "news_articles": len(news_data_dict),
                "current_price": market_data_dict[0]["close"] if market_data_dict else None,
            },
            "insights": insights,
            "news_sentiment": news_sentiment,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating insights: {str(e)}",
        )


@router.post("/analyze/enhanced-trend")
def get_enhanced_trend_analysis(
    symbol: str,
    timeframe: str = Query(default="medium", regex="^(short|medium|long)$"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get enhanced trend analysis with LLM insights."""
    try:
        symbol = symbol.upper()

        # Use the enhanced analysis service
        analysis_service = AnalysisService()
        enhanced_analysis = analysis_service.analyze_trends(symbol, timeframe)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis": enhanced_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing enhanced trend analysis: {str(e)}",
        )


@router.get("/insights/market-summary")
def get_market_summary(
    symbols: List[str] = Query(..., description="List of symbols to analyze"),
    current_user: User = Depends(get_current_active_user),
    session: Session = Depends(get_session),
) -> Any:
    """Get AI-powered market summary for multiple symbols."""
    try:
        symbols = [symbol.upper() for symbol in symbols]
        summaries = {}

        for symbol in symbols[:10]:  # Limit to 10 symbols for performance
            try:
                # Get recent market data (last 7 days)
                start_date = datetime.utcnow() - timedelta(days=7)
                market_query = (
                    select(MarketData)
                    .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date)
                    .order_by(desc(MarketData.timestamp))
                    .limit(10)
                )

                market_data = session.exec(market_query).all()

                if market_data:
                    # Get basic metrics
                    latest_price = float(market_data[0].close_price)
                    oldest_price = float(market_data[-1].close_price)
                    price_change = ((latest_price - oldest_price) / oldest_price) * 100

                    # Get recent news
                    news_query = (
                        select(NewsArticle)
                        .where(
                            NewsArticle.published_at >= start_date,
                            col(NewsArticle.title).contains(symbol) | col(NewsArticle.content).contains(symbol),
                        )
                        .limit(5)
                    )
                    news_articles = session.exec(news_query).all()

                    # Analyze sentiment if news available
                    sentiment_summary = "NEUTRAL"
                    if news_articles:
                        news_data = [
                            {"title": article.title, "content": article.content[:300], "url": article.url}
                            for article in news_articles
                        ]
                        sentiment_result = llm_service.analyze_news_sentiment(news_data)
                        sentiment_summary = sentiment_result.get("overall_sentiment", "NEUTRAL")

                    summaries[symbol] = {
                        "current_price": latest_price,
                        "price_change_7d": round(price_change, 2),
                        "trend": "up" if price_change > 0 else "down" if price_change < 0 else "flat",
                        "sentiment": sentiment_summary,
                        "news_articles_count": len(news_articles),
                        "last_updated": market_data[0].timestamp.isoformat(),
                    }
                else:
                    summaries[symbol] = {"error": "No recent market data available"}

            except Exception as e:
                summaries[symbol] = {"error": f"Analysis failed: {str(e)}"}

        return {
            "market_summary": summaries,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "symbols_analyzed": len([s for s in summaries.values() if "error" not in s]),
            "symbols_failed": len([s for s in summaries.values() if "error" in s]),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating market summary: {str(e)}",
        )


@router.get("/model-status")
def get_llm_model_status(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get status of LLM models."""
    try:
        return {
            "sentiment_analyzer": llm_service.sentiment_analyzer is not None,
            "text_generator": llm_service.text_generator is not None,
            "cuda_available": llm_service.text_generator.device.type == "cuda" if llm_service.text_generator else False,
            "model_info": {
                "sentiment_model": llm_service.settings.sentiment_model_name,
                "device": "GPU"
                if llm_service.text_generator and llm_service.text_generator.device.type == "cuda"
                else "CPU",
            },
            "status": "operational" if llm_service.sentiment_analyzer else "limited",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        return {"status": "error", "error": str(e), "timestamp": datetime.utcnow().isoformat()}
