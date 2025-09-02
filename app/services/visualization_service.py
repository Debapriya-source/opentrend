"""Visualization service for creating charts and plots."""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

from sqlmodel import Session, select
from app.database.connection import engine
from app.database.models import MarketData


class VisualizationService:
    """Service for creating various types of financial visualizations."""

    def __init__(self):
        # Set style preferences
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # Configure Altair
        alt.data_transformers.enable("json")

    def create_price_chart(self, symbol: str, days_back: int = 90, chart_type: str = "candlestick") -> Dict[str, Any]:
        """Create price charts (candlestick, line, OHLC)."""
        try:
            # Get market data
            df = self._get_market_data(symbol, days_back)

            if df.empty:
                return {"error": f"No data available for {symbol}"}

            if chart_type == "candlestick":
                return self._create_candlestick_chart(df, symbol)
            elif chart_type == "line":
                return self._create_line_chart(df, symbol)
            elif chart_type == "ohlc":
                return self._create_ohlc_chart(df, symbol)
            else:
                return {"error": f"Unknown chart type: {chart_type}"}

        except Exception as e:
            logger.error(f"Error creating price chart for {symbol}: {e}")
            return {"error": str(e)}

    def create_technical_analysis_chart(self, symbol: str, days_back: int = 90) -> Dict[str, Any]:
        """Create comprehensive technical analysis chart with indicators."""
        try:
            df = self._get_market_data(symbol, days_back)

            if df.empty:
                return {"error": f"No data available for {symbol}"}

            # Calculate technical indicators
            df = self._add_technical_indicators(df)

            # Create subplots
            fig = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=(f"{symbol} Price & Moving Averages", "Volume", "RSI", "MACD"),
                vertical_spacing=0.08,
                row_heights=[0.5, 0.15, 0.15, 0.2],
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open_price"],
                    high=df["high_price"],
                    low=df["low_price"],
                    close=df["close_price"],
                    name="Price",
                ),
                row=1,
                col=1,
            )

            # Moving averages
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["sma_20"], name="SMA 20", line=dict(color="blue")), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["sma_50"], name="SMA 50", line=dict(color="red")), row=1, col=1
            )

            # Volume
            fig.add_trace(
                go.Bar(x=df["timestamp"], y=df["volume"], name="Volume", marker_color="lightblue"), row=2, col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["rsi"], name="RSI", line=dict(color="purple")), row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

            # MACD
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["macd"], name="MACD", line=dict(color="blue")), row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df["timestamp"], y=df["macd_signal"], name="Signal", line=dict(color="red")), row=4, col=1
            )
            fig.add_trace(
                go.Bar(x=df["timestamp"], y=df["macd_histogram"], name="Histogram", marker_color="gray"), row=4, col=1
            )

            # Update layout
            fig.update_layout(
                title=f"{symbol} Technical Analysis", xaxis_rangeslider_visible=False, height=800, showlegend=True
            )

            # Convert to JSON
            chart_json = fig.to_json()

            return {
                "chart_type": "technical_analysis",
                "symbol": symbol,
                "plotly_json": chart_json,
                "data_points": len(df),
                "date_range": {"start": df["timestamp"].min().isoformat(), "end": df["timestamp"].max().isoformat()},
            }

        except Exception as e:
            logger.error(f"Error creating technical analysis chart for {symbol}: {e}")
            return {"error": str(e)}

    def create_forecast_visualization(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization for forecast results."""
        try:
            symbol = forecast_data.get("symbol", "Unknown")
            forecast_points = forecast_data.get("forecast_points", [])

            if not forecast_points:
                return {"error": "No forecast points available"}

            # Get historical data for context
            df = self._get_market_data(symbol, days_back=90)

            # Create forecast DataFrame
            forecast_df = pd.DataFrame(forecast_points)
            forecast_df["date"] = pd.to_datetime(forecast_df["date"])

            # Create plot
            fig = go.Figure()

            # Historical prices
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df["close_price"],
                    mode="lines",
                    name="Historical Price",
                    line=dict(color="blue"),
                )
            )

            # Forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["predicted_price"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color="red", dash="dash"),
                )
            )

            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["upper_bound"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["lower_bound"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    name="Confidence Interval",
                    fillcolor="rgba(255,0,0,0.2)",
                )
            )

            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Forecast ({forecast_data.get('model_type', 'Unknown')} Model)",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified",
                height=600,
            )

            return {
                "chart_type": "forecast",
                "symbol": symbol,
                "model_type": forecast_data.get("model_type", "unknown"),
                "plotly_json": fig.to_json(),
            }

        except Exception as e:
            logger.error(f"Error creating forecast visualization: {e}")
            return {"error": str(e)}

    def create_sentiment_analysis_chart(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization for sentiment analysis results."""
        try:
            distribution = sentiment_data.get("sentiment_distribution", {})

            if not distribution:
                return {"error": "No sentiment distribution data available"}

            # Create pie chart
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(distribution.keys()),
                        values=list(distribution.values()),
                        hole=0.3,
                        marker_colors=["green", "red", "gray"],
                    )
                ]
            )

            fig.update_layout(
                title="News Sentiment Distribution",
                annotations=[dict(text="Sentiment", x=0.5, y=0.5, font_size=20, showarrow=False)],
            )

            return {
                "chart_type": "sentiment_pie",
                "plotly_json": fig.to_json(),
                "total_articles": sentiment_data.get("total_articles", 0),
            }

        except Exception as e:
            logger.error(f"Error creating sentiment chart: {e}")
            return {"error": str(e)}

    def create_correlation_matrix(self, symbols: List[str], days_back: int = 90) -> Dict[str, Any]:
        """Create correlation matrix for multiple symbols."""
        try:
            # Get data for all symbols
            price_data = {}
            for symbol in symbols:
                df = self._get_market_data(symbol, days_back)
                if not df.empty:
                    price_data[symbol] = df.set_index("timestamp")["close_price"]

            if len(price_data) < 2:
                return {"error": "Need at least 2 symbols with data"}

            # Create correlation matrix
            correlation_df = pd.DataFrame(price_data).corr()

            # Create heatmap using Plotly
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_df.values,
                    x=correlation_df.columns,
                    y=correlation_df.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=np.round(correlation_df.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                )
            )

            fig.update_layout(
                title="Stock Price Correlation Matrix", xaxis_title="Symbols", yaxis_title="Symbols", height=600
            )

            return {
                "chart_type": "correlation_matrix",
                "symbols": list(correlation_df.columns),
                "plotly_json": fig.to_json(),
                "correlation_data": correlation_df.to_dict(),
            }

        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return {"error": str(e)}

    def create_portfolio_performance_chart(self, portfolio_data: Dict[str, float]) -> Dict[str, Any]:
        """Create portfolio performance visualization."""
        try:
            symbols = list(portfolio_data.keys())

            # Get performance data for each symbol
            performance_data = []
            for symbol in symbols:
                df = self._get_market_data(symbol, days_back=30)
                if not df.empty:
                    start_price = df["close_price"].iloc[0]
                    end_price = df["close_price"].iloc[-1]
                    return_pct = ((end_price - start_price) / start_price) * 100
                    performance_data.append({
                        "symbol": symbol,
                        "weight": portfolio_data[symbol],
                        "return_30d": return_pct,
                        "contribution": return_pct * portfolio_data[symbol],
                    })

            if not performance_data:
                return {"error": "No performance data available"}

            perf_df = pd.DataFrame(performance_data)

            # Create subplot with portfolio composition and performance
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("Portfolio Composition", "30-Day Performance"),
                specs=[[{"type": "domain"}, {"type": "xy"}]],
            )

            # Portfolio composition pie chart
            fig.add_trace(go.Pie(labels=perf_df["symbol"], values=perf_df["weight"], name="Composition"), row=1, col=1)

            # Performance bar chart
            colors = ["green" if x > 0 else "red" for x in perf_df["return_30d"]]
            fig.add_trace(
                go.Bar(x=perf_df["symbol"], y=perf_df["return_30d"], name="30D Return %", marker_color=colors),
                row=1,
                col=2,
            )

            fig.update_layout(title="Portfolio Analysis", height=500)

            # Calculate portfolio metrics
            portfolio_return = sum(perf_df["contribution"])

            return {
                "chart_type": "portfolio_performance",
                "plotly_json": fig.to_json(),
                "portfolio_metrics": {
                    "total_return_30d": round(portfolio_return, 2),
                    "best_performer": perf_df.loc[perf_df["return_30d"].idxmax()]["symbol"],
                    "worst_performer": perf_df.loc[perf_df["return_30d"].idxmin()]["symbol"],
                    "symbols_count": len(symbols),
                },
            }

        except Exception as e:
            logger.error(f"Error creating portfolio chart: {e}")
            return {"error": str(e)}

    def _get_market_data(self, symbol: str, days_back: int = 90) -> pd.DataFrame:
        """Get market data for visualization."""
        try:
            with Session(engine) as session:
                start_date = datetime.utcnow() - timedelta(days=days_back)
                query = (
                    select(MarketData)
                    .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date)
                    .order_by("timestamp")
                )

                data_points = session.exec(query).all()

                if not data_points:
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        "timestamp": point.timestamp,
                        "open_price": float(point.open_price),
                        "high_price": float(point.high_price),
                        "low_price": float(point.low_price),
                        "close_price": float(point.close_price),
                        "volume": int(point.volume),
                    }
                    for point in data_points
                ])

                return df

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to DataFrame."""
        # Moving averages
        df["sma_20"] = df["close_price"].rolling(window=20).mean()
        df["sma_50"] = df["close_price"].rolling(window=50).mean()
        df["ema_12"] = df["close_price"].ewm(span=12).mean()
        df["ema_26"] = df["close_price"].ewm(span=26).mean()

        # RSI
        delta = df["close_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        return df

    def _create_candlestick_chart(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create candlestick chart."""
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["timestamp"],
                    open=df["open_price"],
                    high=df["high_price"],
                    low=df["low_price"],
                    close=df["close_price"],
                    name=symbol,
                )
            ]
        )

        fig.update_layout(
            title=f"{symbol} Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=600,
        )

        return {"chart_type": "candlestick", "symbol": symbol, "plotly_json": fig.to_json()}

    def _create_line_chart(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create line chart."""
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["timestamp"],
                    y=df["close_price"],
                    mode="lines",
                    name=f"{symbol} Price",
                    line=dict(color="blue"),
                )
            ]
        )

        fig.update_layout(title=f"{symbol} Price Chart", xaxis_title="Date", yaxis_title="Price ($)", height=600)

        return {"chart_type": "line", "symbol": symbol, "plotly_json": fig.to_json()}

    def _create_ohlc_chart(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Create OHLC chart."""
        fig = go.Figure(
            data=[
                go.Ohlc(
                    x=df["timestamp"],
                    open=df["open_price"],
                    high=df["high_price"],
                    low=df["low_price"],
                    close=df["close_price"],
                    name=symbol,
                )
            ]
        )

        fig.update_layout(title=f"{symbol} OHLC Chart", xaxis_title="Date", yaxis_title="Price ($)", height=600)

        return {"chart_type": "ohlc", "symbol": symbol, "plotly_json": fig.to_json()}


# Global visualization service instance
visualization_service = VisualizationService()
