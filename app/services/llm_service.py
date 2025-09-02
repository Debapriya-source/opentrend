"""LLM service for intelligent market analysis and insights."""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
from transformers import pipeline
from sqlmodel import Session, select, desc

from app.core.config import get_settings
from app.database.connection import engine
from app.database.models import MarketData, NewsArticle


class LLMService:
    """Service for LLM-powered market analysis and insights."""

    def __init__(self):
        self.settings = get_settings()
        self.sentiment_analyzer = None
        self.text_generator = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize LLM models."""
        try:
            # Initialize sentiment analysis model (lightweight approach)
            logger.info("Loading sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.settings.sentiment_model_name,
                return_all_scores=True,
                device=-1,  # Use CPU for better compatibility
                framework="pt",  # Use PyTorch explicitly
            )

            logger.info("Sentiment analysis model initialized successfully")

            # Skip text generation for now to avoid complexity
            self.text_generator = None
            logger.info("Text generation disabled for performance")

        except Exception as e:
            logger.error(f"Error initializing LLM models: {e}")
            # Fallback to basic models if advanced ones fail
            self._initialize_fallback_models()

    def _initialize_fallback_models(self):
        """Initialize fallback models if main models fail."""
        try:
            logger.info("Initializing basic sentiment model...")
            # Use a very simple model as fallback
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1,  # Use CPU
                framework="pt",
            )
            logger.info("Basic sentiment model initialized")
        except Exception as e:
            logger.error(f"Error initializing fallback models: {e}")
            # Use VaderSentiment as ultimate fallback
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.sentiment_analyzer = None
            logger.info("Using VADER sentiment as ultimate fallback")

    def analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles using advanced NLP."""
        if not self.sentiment_analyzer and not hasattr(self, "vader_analyzer"):
            logger.warning("No sentiment analyzer available")
            return {"error": "Sentiment analyzer not initialized"}

        try:
            sentiments = []
            detailed_analysis = []

            for article in articles:
                text = f"{article['title']} {article.get('content', '')[:500]}"

                if self.sentiment_analyzer:
                    # Use transformer model
                    result = self.sentiment_analyzer(text)

                    if isinstance(result, list) and len(result) > 0:
                        scores = result[0] if isinstance(result[0], list) else result

                        # Convert to standardized format
                        sentiment_data = {
                            "article_url": article.get("url", ""),
                            "title": article.get("title", ""),
                            "scores": scores,
                            "dominant_sentiment": max(scores, key=lambda x: x["score"])["label"],
                            "confidence": max(scores, key=lambda x: x["score"])["score"],
                        }

                        detailed_analysis.append(sentiment_data)
                        sentiments.append(sentiment_data["dominant_sentiment"])

                elif hasattr(self, "vader_analyzer"):
                    # Use VADER as fallback
                    vader_scores = self.vader_analyzer.polarity_scores(text)

                    # Convert VADER scores to standard format
                    compound = vader_scores["compound"]
                    if compound >= 0.05:
                        dominant_sentiment = "POSITIVE"
                    elif compound <= -0.05:
                        dominant_sentiment = "NEGATIVE"
                    else:
                        dominant_sentiment = "NEUTRAL"

                    sentiment_data = {
                        "article_url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "scores": [{"label": dominant_sentiment, "score": abs(compound)}],
                        "dominant_sentiment": dominant_sentiment,
                        "confidence": abs(compound),
                    }

                    detailed_analysis.append(sentiment_data)
                    sentiments.append(dominant_sentiment)

            # Aggregate sentiment
            if sentiments:
                positive_count = sentiments.count("POSITIVE")
                negative_count = sentiments.count("NEGATIVE")
                neutral_count = len(sentiments) - positive_count - negative_count

                overall_sentiment = "NEUTRAL"
                if positive_count > negative_count and positive_count > neutral_count:
                    overall_sentiment = "POSITIVE"
                elif negative_count > positive_count and negative_count > neutral_count:
                    overall_sentiment = "NEGATIVE"

                return {
                    "overall_sentiment": overall_sentiment,
                    "sentiment_distribution": {
                        "positive": positive_count,
                        "negative": negative_count,
                        "neutral": neutral_count,
                    },
                    "total_articles": len(articles),
                    "detailed_analysis": detailed_analysis[:10],  # Limit for performance
                }

            return {"error": "No valid sentiment analysis results"}

        except Exception as e:
            logger.error(f"Error in news sentiment analysis: {e}")
            return {"error": str(e)}

    def generate_market_insights(
        self, symbol: str, market_data: List[Dict[str, Any]], news_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate LLM-powered market insights."""
        try:
            # Prepare context for insight generation
            recent_prices = [float(data["close"]) for data in market_data[-5:]]
            price_trend = "rising" if recent_prices[-1] > recent_prices[0] else "falling"
            price_change = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100

            sentiment_context = news_sentiment.get("overall_sentiment", "NEUTRAL").lower()

            # Create analysis context for structured insights
            # (context used internally by insight generation methods)

            insights = {
                "symbol": symbol,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "key_metrics": {
                    "price_trend": price_trend,
                    "price_change_pct": round(price_change, 2),
                    "current_price": round(recent_prices[-1], 2),
                    "sentiment": sentiment_context,
                },
                "insights": self._generate_structured_insights(symbol, market_data, news_sentiment),
                "risk_assessment": self._assess_risk(market_data, news_sentiment),
                "recommendations": self._generate_recommendations(symbol, market_data, news_sentiment),
            }

            return insights

        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return {"error": str(e)}

    def _generate_structured_insights(
        self, symbol: str, market_data: List[Dict[str, Any]], news_sentiment: Dict[str, Any]
    ) -> List[str]:
        """Generate structured insights based on data analysis."""
        insights = []

        try:
            # Price analysis
            prices = [float(data["close"]) for data in market_data[-10:]]
            if len(prices) >= 2:
                recent_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100
                if abs(recent_change) > 5:
                    insights.append(
                        f"{symbol} experienced significant price movement of {recent_change:.1f}% in the latest session"
                    )

            # Volume analysis
            volumes = [data.get("volume", 0) for data in market_data[-5:]]
            if volumes:
                avg_volume = sum(volumes) / len(volumes)
                latest_volume = volumes[-1]
                if latest_volume > avg_volume * 1.5:
                    insights.append(
                        f"Trading volume is {(latest_volume / avg_volume):.1f}x above average, indicating increased interest"
                    )

            # Sentiment analysis
            sentiment = news_sentiment.get("overall_sentiment", "NEUTRAL")
            if sentiment != "NEUTRAL":
                distribution = news_sentiment.get("sentiment_distribution", {})
                total = sum(distribution.values())
                if total > 0:
                    sentiment_strength = max(distribution.values()) / total
                    if sentiment_strength > 0.6:
                        insights.append(
                            f"News sentiment is strongly {sentiment.lower()} with {sentiment_strength:.1%} of articles showing this bias"
                        )

            # Technical patterns
            if len(prices) >= 5:
                if all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1)):
                    insights.append(f"{symbol} shows a consistent upward trend over the recent period")
                elif all(prices[i] >= prices[i + 1] for i in range(len(prices) - 1)):
                    insights.append(f"{symbol} shows a consistent downward trend over the recent period")

            return insights[:5]  # Limit to top 5 insights

        except Exception as e:
            logger.error(f"Error generating structured insights: {e}")
            return [f"Analysis for {symbol} completed with limited insights due to data constraints"]

    def _assess_risk(self, market_data: List[Dict[str, Any]], news_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk based on market data and sentiment."""
        try:
            prices = [float(data["close"]) for data in market_data[-20:]]

            # Calculate volatility
            if len(prices) > 1:
                price_changes = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
                volatility = sum(abs(change) for change in price_changes) / len(price_changes)
            else:
                volatility = 0

            # Risk factors
            risk_factors = []
            risk_score = 0

            if volatility > 0.05:  # 5% average daily change
                risk_factors.append("High price volatility detected")
                risk_score += 3

            sentiment = news_sentiment.get("overall_sentiment", "NEUTRAL")
            if sentiment == "NEGATIVE":
                risk_factors.append("Negative news sentiment")
                risk_score += 2
            elif sentiment == "POSITIVE":
                risk_score -= 1  # Positive sentiment reduces risk

            # Volume risk
            volumes = [data.get("volume", 0) for data in market_data[-5:]]
            if volumes and volumes[-1] < sum(volumes[:-1]) / len(volumes[:-1]) * 0.5:
                risk_factors.append("Low trading volume may indicate liquidity risk")
                risk_score += 1

            # Risk level
            if risk_score <= 1:
                risk_level = "LOW"
            elif risk_score <= 3:
                risk_level = "MODERATE"
            else:
                risk_level = "HIGH"

            return {
                "risk_level": risk_level,
                "risk_score": min(risk_score, 10),  # Cap at 10
                "volatility": round(volatility * 100, 2),
                "risk_factors": risk_factors,
            }

        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {"risk_level": "UNKNOWN", "error": str(e)}

    def _generate_recommendations(
        self, symbol: str, market_data: List[Dict[str, Any]], news_sentiment: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        try:
            prices = [float(data["close"]) for data in market_data[-10:]]
            sentiment = news_sentiment.get("overall_sentiment", "NEUTRAL")

            # Price-based recommendations
            if len(prices) >= 2:
                recent_change = ((prices[-1] - prices[-2]) / prices[-2]) * 100

                if recent_change > 5 and sentiment == "POSITIVE":
                    recommendations.append(
                        "Consider taking profits if holding long positions due to strong recent gains"
                    )
                elif recent_change < -5 and sentiment != "NEGATIVE":
                    recommendations.append("Potential buying opportunity on the dip if fundamentals remain strong")

            # Sentiment-based recommendations
            if sentiment == "NEGATIVE":
                recommendations.append("Exercise caution due to negative news sentiment - wait for stabilization")
            elif sentiment == "POSITIVE":
                recommendations.append("Positive news flow supports current price levels")

            # Volume-based recommendations
            volumes = [data.get("volume", 0) for data in market_data[-3:]]
            if volumes and volumes[-1] > sum(volumes[:-1]) / len(volumes[:-1]) * 2:
                recommendations.append("High volume suggests strong conviction in current price movement")

            # Default recommendation if no specific signals
            if not recommendations:
                recommendations.append(
                    "Monitor closely for clearer directional signals before making investment decisions"
                )

            return recommendations[:3]  # Limit to top 3 recommendations

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [f"Unable to generate specific recommendations for {symbol} at this time"]

    def enhance_trend_analysis(self, symbol: str, technical_indicators: Dict[str, float]) -> Dict[str, Any]:
        """Enhance trend analysis with LLM insights."""
        try:
            # Get recent market data and news
            with Session(engine) as session:
                # Get recent market data
                market_query = (
                    select(MarketData).where(MarketData.symbol == symbol).order_by(desc(MarketData.timestamp)).limit(30)
                )
                market_data = session.exec(market_query).all()

                # Get recent news
                news_query = (
                    select(NewsArticle)
                    .where(NewsArticle.published_at >= datetime.utcnow() - timedelta(days=7))
                    .order_by(desc(NewsArticle.published_at))
                    .limit(20)
                )
                news_articles = session.exec(news_query).all()

            if not market_data:
                return {"error": "No market data available for analysis"}

            # Convert to dictionaries
            market_data_dict = [
                {"close": float(data.close_price), "volume": int(data.volume), "timestamp": data.timestamp.isoformat()}
                for data in market_data
            ]

            news_data_dict = [
                {
                    "title": article.title,
                    "content": article.content[:500],  # Limit content length
                    "url": article.url,
                    "published_at": article.published_at.isoformat(),
                }
                for article in news_articles
            ]

            # Analyze news sentiment
            news_sentiment = self.analyze_news_sentiment(news_data_dict)

            # Generate comprehensive insights
            insights = self.generate_market_insights(symbol, market_data_dict, news_sentiment)

            # Combine with technical indicators
            enhanced_analysis = {
                "symbol": symbol,
                "technical_indicators": technical_indicators,
                "llm_insights": insights,
                "news_sentiment": news_sentiment,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "data_points_analyzed": {
                    "market_data_points": len(market_data_dict),
                    "news_articles": len(news_data_dict),
                },
            }

            return enhanced_analysis

        except Exception as e:
            logger.error(f"Error enhancing trend analysis: {e}")
            return {"error": str(e)}


# Global LLM service instance
llm_service = LLMService()
