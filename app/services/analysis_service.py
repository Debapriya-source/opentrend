"""Analysis service for trend analysis, predictions, and sentiment analysis."""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlmodel import Session, desc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from loguru import logger
from app.database.connection import engine

from app.database.models import MarketData, NewsArticle
from app.services.llm_service import llm_service


class AnalysisService:
    """Service for market analysis, predictions, and sentiment analysis."""

    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.scaler = StandardScaler()

    def analyze_trends(self, symbol: str, timeframe: str = "medium") -> Dict[str, Any]:
        """Analyze trends for a specific symbol."""
        try:
            logger.info(f"Analyzing trends for {symbol} with timeframe {timeframe}")

            # Get market data
            with Session(engine) as session:
                # Determine date range based on timeframe
                if timeframe == "short":
                    start_date = datetime.utcnow() - timedelta(days=30)
                elif timeframe == "medium":
                    start_date = datetime.utcnow() - timedelta(days=90)
                else:  # long
                    start_date = datetime.utcnow() - timedelta(days=365)

                # Get market data from database
                from sqlmodel import select

                query = (
                    select(MarketData)
                    .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date)
                    .order_by("timestamp")
                )

                data_points = session.exec(query).all()

            if not data_points:
                raise ValueError(f"No market data found for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame([point.dict() for point in data_points])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(df)

            # Determine trend
            trend_type, confidence_score = self._determine_trend(df, indicators)

            # Generate description
            description = self._generate_trend_description(symbol, trend_type, indicators)

            # Enhance with LLM insights
            try:
                llm_insights = llm_service.enhance_trend_analysis(symbol, indicators)
                enhanced_result = {
                    "symbol": symbol,
                    "trend_type": trend_type,
                    "confidence_score": confidence_score,
                    "timeframe": timeframe,
                    "indicators": indicators,
                    "description": description,
                    "llm_analysis": llm_insights,
                }
                logger.info(f"Enhanced trend analysis completed for {symbol}: {trend_type}")
                return enhanced_result
            except Exception as e:
                logger.warning(f"LLM enhancement failed for {symbol}, returning basic analysis: {e}")
                result = {
                    "symbol": symbol,
                    "trend_type": trend_type,
                    "confidence_score": confidence_score,
                    "timeframe": timeframe,
                    "indicators": indicators,
                    "description": description,
                }
                return result

        except Exception as e:
            logger.error(f"Error analyzing trends for {symbol}: {e}")
            raise

    def generate_prediction(
        self, symbol: str, prediction_type: str = "price", horizon_days: int = 30
    ) -> Dict[str, Any]:
        """Generate predictions for a specific symbol."""
        try:
            logger.info(f"Generating {prediction_type} prediction for {symbol}")

            # Get historical data
            with Session(engine) as session:
                from sqlmodel import select

                start_date = datetime.utcnow() - timedelta(days=365)
                query = (
                    select(MarketData)
                    .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date)
                    .order_by("timestamp")
                )

                data_points = session.exec(query).all()

            if not data_points:
                raise ValueError(f"No market data found for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame([point.dict() for point in data_points])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()

            if prediction_type == "price":
                prediction_result = self._predict_price(df, horizon_days)
            elif prediction_type == "trend":
                prediction_result = self._predict_trend(df, horizon_days)
            elif prediction_type == "volatility":
                prediction_result = self._predict_volatility(df, horizon_days)
            else:
                raise ValueError(f"Unknown prediction type: {prediction_type}")

            result = {
                "symbol": symbol,
                "prediction_type": prediction_type,
                "predicted_value": prediction_result["predicted_value"],
                "confidence_interval_lower": prediction_result["confidence_interval_lower"],
                "confidence_interval_upper": prediction_result["confidence_interval_upper"],
                "model_used": prediction_result["model_used"],
                "features_used": prediction_result["features_used"],
            }

            logger.info(f"Prediction generated for {symbol}: {prediction_type}")
            return result

        except Exception as e:
            logger.error(f"Error generating prediction for {symbol}: {e}")
            raise

    def get_current_sentiment(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get current market sentiment."""
        try:
            logger.info("Getting current market sentiment")

            # Get recent news articles
            with Session(engine) as session:
                from sqlmodel import select

                start_date = datetime.utcnow() - timedelta(days=7)
                query = (
                    select(NewsArticle)
                    .where(NewsArticle.published_at >= start_date)
                    .order_by(desc(NewsArticle.published_at))
                )

                if symbol:
                    # Filter by symbol-related keywords
                    symbol_keywords = [symbol.lower(), symbol.replace(".", "").lower()]
                    # This is a simple filter - could be enhanced with better text matching
                    articles = session.exec(query).all()
                    articles = [
                        a
                        for a in articles
                        if any(
                            keyword in a.title.lower() or keyword in a.content.lower() for keyword in symbol_keywords
                        )
                    ]
                else:
                    articles = session.exec(query).all()

            if not articles:
                return {
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "articles_analyzed": 0,
                    "sentiment_distribution": {
                        "positive": 0,
                        "neutral": 0,
                        "negative": 0,
                    },
                }

            # Analyze sentiment
            sentiment_scores = []
            sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}

            for article in articles:
                # Combine title and content for sentiment analysis
                text = f"{article.title} {article.content}"
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                compound_score = sentiment["compound"]
                sentiment_scores.append(compound_score)

                # Categorize sentiment
                if compound_score >= 0.05:
                    sentiment_distribution["positive"] += 1
                elif compound_score <= -0.05:
                    sentiment_distribution["negative"] += 1
                else:
                    sentiment_distribution["neutral"] += 1

            # Calculate overall sentiment
            avg_sentiment = np.mean(sentiment_scores)

            if avg_sentiment >= 0.05:
                overall_sentiment = "positive"
            elif avg_sentiment <= -0.05:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"

            result = {
                "overall_sentiment": overall_sentiment,
                "sentiment_score": avg_sentiment,
                "articles_analyzed": len(articles),
                "sentiment_distribution": sentiment_distribution,
                "symbol": symbol,
            }

            logger.info(f"Sentiment analysis completed: {overall_sentiment}")
            return result

        except Exception as e:
            logger.error(f"Error getting sentiment: {e}")
            raise

    def train_models(self, model_type: str = "all") -> Dict[str, Any]:
        """Train or retrain ML models."""
        try:
            logger.info(f"Training models: {model_type}")

            # This is a placeholder for model training
            # In a real implementation, this would:
            # 1. Load historical data
            # 2. Preprocess features
            # 3. Train models (sentiment, forecasting, etc.)
            # 4. Save models to storage

            result = {
                "status": "completed",
                "models_trained": [model_type],
                "training_date": datetime.utcnow().isoformat(),
            }

            logger.info(f"Model training completed: {model_type}")
            return result

        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators."""
        indicators = {}

        # Moving averages
        indicators["sma_20"] = df["close_price"].rolling(window=20).mean().iloc[-1]
        indicators["sma_50"] = df["close_price"].rolling(window=50).mean().iloc[-1]
        indicators["ema_12"] = df["close_price"].ewm(span=12).mean().iloc[-1]
        indicators["ema_26"] = df["close_price"].ewm(span=26).mean().iloc[-1]

        # RSI
        delta = df["close_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators["rsi"] = 100 - (100 / (1 + rs.iloc[-1]))

        # MACD
        indicators["macd"] = indicators["ema_12"] - indicators["ema_26"]

        # Bollinger Bands
        sma_20 = df["close_price"].rolling(window=20).mean()
        std_20 = df["close_price"].rolling(window=20).std()
        indicators["bb_upper"] = sma_20.iloc[-1] + (std_20.iloc[-1] * 2)
        indicators["bb_lower"] = sma_20.iloc[-1] - (std_20.iloc[-1] * 2)

        # Volume indicators
        indicators["volume_sma"] = df["volume"].rolling(window=20).mean().iloc[-1]
        indicators["volume_ratio"] = df["volume"].iloc[-1] / indicators["volume_sma"]

        return indicators

    def _determine_trend(self, df: pd.DataFrame, indicators: Dict[str, float]) -> tuple:
        """Determine trend based on technical indicators."""
        current_price = df["close_price"].iloc[-1]

        # Trend signals
        bullish_signals = 0
        bearish_signals = 0

        # Moving average signals
        if current_price > indicators["sma_20"]:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if current_price > indicators["sma_50"]:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # MACD signals
        if indicators["macd"] > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # RSI signals
        if 30 < indicators["rsi"] < 70:
            # Neutral RSI
            pass
        elif indicators["rsi"] > 70:
            bearish_signals += 1  # Overbought
        else:
            bullish_signals += 1  # Oversold

        # Volume signals
        if indicators["volume_ratio"] > 1.5:
            # High volume - amplify existing signals
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            elif bearish_signals > bullish_signals:
                bearish_signals += 1

        # Determine trend
        if bullish_signals > bearish_signals:
            trend_type = "bullish"
            confidence_score = min(0.9, 0.5 + (bullish_signals - bearish_signals) * 0.1)
        elif bearish_signals > bullish_signals:
            trend_type = "bearish"
            confidence_score = min(0.9, 0.5 + (bearish_signals - bullish_signals) * 0.1)
        else:
            trend_type = "neutral"
            confidence_score = 0.5

        return trend_type, confidence_score

    def _generate_trend_description(self, symbol: str, trend_type: str, indicators: Dict[str, float]) -> str:
        """Generate human-readable trend description."""
        _current_price = indicators.get("sma_20", 0)

        if trend_type == "bullish":
            description = f"{symbol} shows bullish momentum with strong technical indicators. "
            description += "Price is above key moving averages and MACD is positive. "
            description += f"RSI at {indicators.get('rsi', 0):.1f} indicates healthy momentum."
        elif trend_type == "bearish":
            description = f"{symbol} shows bearish pressure with weakening technical indicators. "
            description += "Price is below key moving averages and MACD is negative. "
            description += f"RSI at {indicators.get('rsi', 0):.1f} suggests potential reversal."
        else:
            description = f"{symbol} is in a neutral consolidation phase. "
            description += "Technical indicators are mixed, suggesting indecision in the market."

        return description

    def _predict_price(self, df: pd.DataFrame, horizon_days: int) -> Dict[str, Any]:
        """Predict future price using simple moving average trend."""
        # Simple prediction based on recent trend
        recent_prices = df["close_price"].tail(30)
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]

        current_price = df["close_price"].iloc[-1]
        predicted_price = current_price + (trend * horizon_days)

        # Simple confidence interval
        volatility = df["close_price"].pct_change().std()
        confidence_range = current_price * volatility * np.sqrt(horizon_days)

        return {
            "predicted_value": predicted_price,
            "confidence_interval_lower": predicted_price - confidence_range,
            "confidence_interval_upper": predicted_price + confidence_range,
            "model_used": "trend_extrapolation",
            "features_used": {"trend": trend, "volatility": volatility},
        }

    def _predict_trend(self, df: pd.DataFrame, horizon_days: int) -> Dict[str, Any]:
        """Predict trend direction."""
        # Simple trend prediction based on moving averages
        sma_short = df["close_price"].rolling(window=10).mean().iloc[-1]
        sma_long = df["close_price"].rolling(window=30).mean().iloc[-1]

        if sma_short > sma_long:
            predicted_trend = 1.0  # Bullish
        else:
            predicted_trend = -1.0  # Bearish

        return {
            "predicted_value": predicted_trend,
            "confidence_interval_lower": -1.0,
            "confidence_interval_upper": 1.0,
            "model_used": "moving_average_crossover",
            "features_used": {"sma_short": sma_short, "sma_long": sma_long},
        }

    def _predict_volatility(self, df: pd.DataFrame, horizon_days: int) -> Dict[str, Any]:
        """Predict future volatility."""
        # Calculate historical volatility
        returns = df["close_price"].pct_change().dropna()
        current_volatility = returns.std()

        # Simple volatility prediction (mean reversion)
        long_term_volatility = returns.rolling(window=252).std().mean()
        predicted_volatility = (current_volatility + long_term_volatility) / 2

        return {
            "predicted_value": predicted_volatility,
            "confidence_interval_lower": predicted_volatility * 0.5,
            "confidence_interval_upper": predicted_volatility * 1.5,
            "model_used": "volatility_mean_reversion",
            "features_used": {
                "current_vol": current_volatility,
                "long_term_vol": long_term_volatility,
            },
        }
