"""Advanced forecasting service using Prophet and other ML models."""

from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlmodel import Session, select

from app.core.config import get_settings
from app.database.connection import engine
from app.database.models import MarketData

warnings.filterwarnings("ignore", category=FutureWarning)


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class ForecastingService:
    """Advanced forecasting service with multiple ML models."""

    def __init__(self):
        self.settings = get_settings()
        self.models = {}
        self.scalers = {}

    def forecast_with_prophet(self, symbol: str, horizon_days: int = 30) -> Dict[str, Any]:
        """Generate forecasts using Facebook Prophet."""
        try:
            logger.info(f"Starting Prophet forecast for {symbol} with {horizon_days} days horizon")

            # Get historical data
            df = self._get_historical_data(symbol, days_back=730)  # 2 years of data

            if len(df) < 30:
                raise ValueError(f"Insufficient data for {symbol}. Need at least 30 days, got {len(df)}")

            # Prepare data for Prophet
            prophet_df = df[["timestamp", "close_price"]].copy()
            prophet_df.columns = ["ds", "y"]
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            # Initialize and configure Prophet model
            model = Prophet(
                changepoint_prior_scale=0.05,  # Flexibility in trend changes
                seasonality_prior_scale=10.0,  # Seasonality strength
                holidays_prior_scale=10.0,  # Holiday effects
                seasonality_mode="multiplicative",  # Better for financial data
                interval_width=0.8,  # 80% confidence intervals
                daily_seasonality=False,  # Disable daily seasonality for stock data
                weekly_seasonality=True,  # Keep weekly patterns
                yearly_seasonality=True,  # Keep yearly patterns
            )

            # Add custom seasonalities for financial markets
            model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)

            # Fit the model
            logger.info(f"Training Prophet model with {len(prophet_df)} data points")
            model.fit(prophet_df)

            # Create future dates
            future = model.make_future_dataframe(periods=horizon_days)

            # Generate forecast
            forecast = model.predict(future)

            # Extract forecast results
            forecast_data = forecast.tail(horizon_days)

            # Calculate model performance on historical data
            historical_forecast = forecast[:-horizon_days]
            mae = mean_absolute_error(prophet_df["y"], historical_forecast["yhat"])
            rmse = np.sqrt(mean_squared_error(prophet_df["y"], historical_forecast["yhat"]))

            # Prepare results
            forecast_points = []
            for _, row in forecast_data.iterrows():
                forecast_points.append({
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "predicted_price": round(row["yhat"], 2),
                    "lower_bound": round(row["yhat_lower"], 2),
                    "upper_bound": round(row["yhat_upper"], 2),
                    "trend": round(row["trend"], 2),
                })

            current_price = df["close_price"].iloc[-1]
            predicted_price = forecast_data["yhat"].iloc[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100

            result = {
                "symbol": symbol,
                "model_type": "prophet",
                "forecast_horizon_days": horizon_days,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "price_change_percent": round(price_change, 2),
                "confidence_interval": {
                    "lower": round(forecast_data["yhat_lower"].iloc[-1], 2),
                    "upper": round(forecast_data["yhat_upper"].iloc[-1], 2),
                },
                "model_performance": {
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "training_data_points": len(prophet_df),
                },
                "forecast_points": forecast_points,
                "trend_components": {
                    "trend": round(forecast_data["trend"].iloc[-1], 2),
                    "weekly": round(forecast_data.get("weekly", pd.Series([0])).iloc[-1], 2),
                    "yearly": round(forecast_data.get("yearly", pd.Series([0])).iloc[-1], 2),
                },
                "forecast_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(f"Prophet forecast completed for {symbol}: {price_change:.2f}% change predicted")
            return result

        except Exception as e:
            logger.error(f"Error in Prophet forecast for {symbol}: {e}")
            return {"error": str(e), "model_type": "prophet"}

    def forecast_with_lstm(self, symbol: str, horizon_days: int = 30, sequence_length: int = 60) -> Dict[str, Any]:
        """Generate forecasts using LSTM neural network."""
        try:
            logger.info(f"Starting LSTM forecast for {symbol}")

            # Get historical data
            df = self._get_historical_data(symbol, days_back=365)

            if len(df) < sequence_length + 30:
                raise ValueError(f"Insufficient data for LSTM. Need at least {sequence_length + 30} days")

            # Prepare data
            prices = df["close_price"].values.reshape(-1, 1)

            # Scale the data
            scaler = StandardScaler()
            scaled_prices = scaler.fit_transform(prices)

            # Create sequences
            X, y = self._create_sequences(scaled_prices, sequence_length)

            # Split data (80% train, 20% test)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)

            # Initialize model
            model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Train model
            model.train()
            epochs = 100
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 20 == 0:
                    logger.debug(f"LSTM Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test_tensor)
                test_loss = criterion(test_predictions, y_test_tensor)

            # Generate forecast
            model.eval()
            with torch.no_grad():
                # Use last sequence for prediction
                last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
                last_sequence_tensor = torch.FloatTensor(last_sequence)

                forecast_points = []
                current_sequence = last_sequence_tensor.clone()

                for i in range(horizon_days):
                    prediction = model(current_sequence)
                    forecast_points.append(prediction.item())

                    # Update sequence for next prediction
                    new_sequence = torch.cat([current_sequence[:, 1:, :], prediction.unsqueeze(0).unsqueeze(2)], dim=1)
                    current_sequence = new_sequence

            # Inverse transform predictions
            forecast_prices = scaler.inverse_transform(np.array(forecast_points).reshape(-1, 1)).flatten()

            # Calculate confidence intervals (simple approach using historical volatility)
            returns = df["close_price"].pct_change().dropna()
            volatility = returns.std()

            current_price = df["close_price"].iloc[-1]
            predicted_price = forecast_prices[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100

            # Generate forecast points with dates
            forecast_dates = pd.date_range(
                start=df["timestamp"].iloc[-1] + timedelta(days=1), periods=horizon_days, freq="D"
            )

            detailed_forecast = []
            for i, (date, price) in enumerate(zip(forecast_dates, forecast_prices)):
                # Simple confidence interval based on volatility
                confidence_range = price * volatility * np.sqrt(i + 1)
                detailed_forecast.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": round(price, 2),
                    "lower_bound": round(price - confidence_range, 2),
                    "upper_bound": round(price + confidence_range, 2),
                })

            result = {
                "symbol": symbol,
                "model_type": "lstm",
                "forecast_horizon_days": horizon_days,
                "current_price": round(current_price, 2),
                "predicted_price": round(predicted_price, 2),
                "price_change_percent": round(price_change, 2),
                "confidence_interval": {
                    "lower": round(predicted_price - (predicted_price * volatility * np.sqrt(horizon_days)), 2),
                    "upper": round(predicted_price + (predicted_price * volatility * np.sqrt(horizon_days)), 2),
                },
                "model_performance": {
                    "test_loss": round(test_loss.item(), 6),
                    "training_epochs": epochs,
                    "sequence_length": sequence_length,
                },
                "forecast_points": detailed_forecast,
                "forecast_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(f"LSTM forecast completed for {symbol}: {price_change:.2f}% change predicted")
            return result

        except Exception as e:
            logger.error(f"Error in LSTM forecast for {symbol}: {e}")
            return {"error": str(e), "model_type": "lstm"}

    def forecast_ensemble(self, symbol: str, horizon_days: int = 30) -> Dict[str, Any]:
        """Generate ensemble forecast using multiple models."""
        try:
            logger.info(f"Starting ensemble forecast for {symbol}")

            forecasts = {}
            weights = {}

            # Prophet forecast
            try:
                prophet_result = self.forecast_with_prophet(symbol, horizon_days)
                if "error" not in prophet_result:
                    forecasts["prophet"] = prophet_result
                    weights["prophet"] = 0.4  # Higher weight for Prophet
            except Exception as e:
                logger.warning(f"Prophet forecast failed: {e}")

            # LSTM forecast
            try:
                lstm_result = self.forecast_with_lstm(symbol, horizon_days)
                if "error" not in lstm_result:
                    forecasts["lstm"] = lstm_result
                    weights["lstm"] = 0.3
            except Exception as e:
                logger.warning(f"LSTM forecast failed: {e}")

            # Random Forest forecast (simple implementation)
            try:
                rf_result = self._forecast_with_random_forest(symbol, horizon_days)
                if "error" not in rf_result:
                    forecasts["random_forest"] = rf_result
                    weights["random_forest"] = 0.3
            except Exception as e:
                logger.warning(f"Random Forest forecast failed: {e}")

            if not forecasts:
                return {"error": "All forecasting models failed", "model_type": "ensemble"}

            # Normalize weights
            total_weight = sum(weights.values())
            for model in weights:
                weights[model] = weights[model] / total_weight

            # Calculate ensemble prediction
            ensemble_price = 0
            ensemble_lower = 0
            ensemble_upper = 0

            for model_name, forecast in forecasts.items():
                weight = weights[model_name]
                ensemble_price += forecast["predicted_price"] * weight
                ensemble_lower += forecast["confidence_interval"]["lower"] * weight
                ensemble_upper += forecast["confidence_interval"]["upper"] * weight

            current_price = list(forecasts.values())[0]["current_price"]
            price_change = ((ensemble_price - current_price) / current_price) * 100

            result = {
                "symbol": symbol,
                "model_type": "ensemble",
                "forecast_horizon_days": horizon_days,
                "current_price": round(current_price, 2),
                "predicted_price": round(ensemble_price, 2),
                "price_change_percent": round(price_change, 2),
                "confidence_interval": {"lower": round(ensemble_lower, 2), "upper": round(ensemble_upper, 2)},
                "model_weights": weights,
                "individual_forecasts": {
                    model: {
                        "predicted_price": forecast["predicted_price"],
                        "price_change_percent": forecast["price_change_percent"],
                    }
                    for model, forecast in forecasts.items()
                },
                "models_used": list(forecasts.keys()),
                "forecast_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(f"Ensemble forecast completed for {symbol}: {price_change:.2f}% change predicted")
            return result

        except Exception as e:
            logger.error(f"Error in ensemble forecast for {symbol}: {e}")
            return {"error": str(e), "model_type": "ensemble"}

    def _get_historical_data(self, symbol: str, days_back: int = 365) -> pd.DataFrame:
        """Get historical market data for a symbol."""
        with Session(engine) as session:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            query = (
                select(MarketData)
                .where(MarketData.symbol == symbol, MarketData.timestamp >= start_date)
                .order_by("timestamp")
            )

            data_points = session.exec(query).all()

            if not data_points:
                raise ValueError(f"No historical data found for {symbol}")

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

    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i : (i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    def _forecast_with_random_forest(self, symbol: str, horizon_days: int = 30) -> Dict[str, Any]:
        """Simple Random Forest forecast implementation."""
        try:
            df = self._get_historical_data(symbol, days_back=365)

            # Create features
            df["returns"] = df["close_price"].pct_change()
            df["sma_5"] = df["close_price"].rolling(5).mean()
            df["sma_20"] = df["close_price"].rolling(20).mean()
            df["volume_sma"] = df["volume"].rolling(20).mean()
            df["price_to_sma"] = df["close_price"] / df["sma_20"]
            df["volume_ratio"] = df["volume"] / df["volume_sma"]

            # Drop NaN values
            df = df.dropna()

            if len(df) < 50:
                raise ValueError("Insufficient data for Random Forest")

            # Prepare features and target
            feature_cols = ["sma_5", "sma_20", "price_to_sma", "volume_ratio"]
            X = df[feature_cols].values
            y = df["close_price"].values

            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X[:-10], y[10:])  # Predict next day based on current features

            # Simple forecast (using last known features)
            last_features = X[-1].reshape(1, -1)
            predicted_price = model.predict(last_features)[0]

            current_price = df["close_price"].iloc[-1]
            price_change = ((predicted_price - current_price) / current_price) * 100

            # Simple confidence interval
            returns_std = df["returns"].std()
            confidence_range = current_price * returns_std * np.sqrt(horizon_days)

            return {
                "symbol": symbol,
                "model_type": "random_forest",
                "predicted_price": round(predicted_price, 2),
                "price_change_percent": round(price_change, 2),
                "confidence_interval": {
                    "lower": round(predicted_price - confidence_range, 2),
                    "upper": round(predicted_price + confidence_range, 2),
                },
            }

        except Exception as e:
            return {"error": str(e), "model_type": "random_forest"}


# Global forecasting service instance
forecasting_service = ForecastingService()
