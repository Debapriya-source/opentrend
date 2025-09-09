# OpenTrend AI — Project Report

## 1. Executive Summary

OpenTrend AI is an AI-powered market intelligence platform that automatically identifies, analyzes, and forecasts financial market trends by combining multi-source data (market prices, news, social media) with advanced ML and LLM-driven insights. It exposes a secure FastAPI backend, a professional Streamlit frontend, and a modular services layer for data ingestion, forecasting, visualization, and AI insights.

## 2. Objectives and Scope

- Deliver proactive trend discovery and forecasting for equities and similar assets
- Provide real-time and historical analytics with explainable indicators
- Offer a developer-friendly API and an interactive dashboard for end users
- Support scalable data storage (relational and time-series) and background scheduling

## 3. System Architecture

- Backend: FastAPI application (`app/main.py`) exposing v1 API (`app/api/v1/`) with routers for auth, data, analysis, timeseries, llm, forecasting, and visualization
- Services: Modular services for analysis, forecasting (Prophet, LSTM, ensembles), data collection (yfinance, RSS), LLM insights, visualization, scheduler
- Storage: SQLModel ORM over PostgreSQL/SQLite for structured data; InfluxDB for time-series; Redis for cache
- Frontend: Streamlit apps (`frontend/advanced_streamlit_app.py`, `frontend/streamlit_app.py`) consuming the backend
- Containerization: Dockerfile and docker-compose for local/dev deployments

### Key Components

- `app/core/config.py`: Centralized settings via `pydantic-settings`
- `app/database/models.py`: `User`, `MarketData`, `NewsArticle`, `TrendAnalysis`, `Prediction`
- `app/services/*`: Analysis, Forecasting, LLM, Visualization, Data Collector, Scheduler
- `app/api/v1/endpoints/*`: REST endpoints for the above capabilities

## 4. Features

- Authentication & Security
  - Registration, login, JWT-based auth; `get_current_active_user` dependency gating sensitive endpoints
- Data Ingestion
  - Market data via yfinance; RSS-based news collection (Reuters, Bloomberg, FT); duplicate-safe persistence
  - Batch ingestion and scheduler-driven updates (hourly market data, 30-min news)
  - Optional InfluxDB writes for time-series analytics
- Analysis & Insights
  - Trend analysis with technical indicators (SMA, EMA, RSI, MACD, Bollinger, volume)
  - Current sentiment using VADER/Transformers; AI market insights and recommendations via LLM service
- Forecasting
  - Prophet-based forecasts with seasonalities and performance metrics (MAE/RMSE)
  - LSTM sequence modeling with PyTorch; Random Forest feature-based model; weighted ensembles
  - Batch and comparative forecasts with consensus metrics
- Visualization
  - Plotly-based price, OHLC, technical analysis, forecast visuals
  - Correlation matrices; portfolio performance and composition visuals
- Frontend
  - Advanced Streamlit interface integrating all backend features with auth, charts, insights, and portfolio tools

## 5. API Overview (v1)

- Auth: `/auth/register`, `/auth/login`, `/auth/me`
- Data: `/data/ingest/market-data`, `/data/ingest/news`, `/data/market-data/{symbol}`, `/data/news`, `/data/batch/market-data`, `/data/scheduler/trigger`
- Analysis: `/analysis/trends/analyze`, `/analysis/predictions/generate`, `/analysis/trends`, `/analysis/predictions`, `/analysis/market-data/{symbol}`, `/analysis/sentiment/current`, `/analysis/models/train`
- Time-series (Influx): `/timeseries/market-data/{symbol}`, `/timeseries/market-data/{symbol}/latest`, `/timeseries/market-data/{symbol}/ohlcv`, `/timeseries/market-data/multiple`, `/timeseries/market-data/compare`, `/timeseries/health`
- Forecasting: `/forecasting/prophet/{symbol}`, `/forecasting/lstm/{symbol}`, `/forecasting/ensemble/{symbol}`, `/forecasting/compare/{symbol}`, `/forecasting/batch`, `/forecasting/models/status`
- LLM Insights: `/llm/analyze/sentiment`, `/llm/insights/{symbol}`, `/llm/analyze/enhanced-trend`, `/llm/insights/market-summary`, `/llm/model-status`
- Visualization: `/visualization/price-chart/{symbol}`, `/visualization/technical-analysis/{symbol}`, `/visualization/forecast-chart`, `/visualization/sentiment-chart`, `/visualization/correlation-matrix`, `/visualization/portfolio-performance`, `/visualization/dashboard/{symbol}`, `/visualization/chart-types`

## 6. Data Model

- Users with `email`, `username`, hashed passwords, roles, timestamps
- MarketData with OHLCV, symbol, timestamp, source
- NewsArticle with title/content/source/url/published_at, sentiment fields
- TrendAnalysis with indicators JSON, confidence, timeframe, description
- Prediction with type, bounds, model name, features JSON

## 7. ML/AI Methods

- Technical Indicators: SMA-20/50, EMA-12/26, RSI, MACD, Bollinger, volume signals
- Forecasting:
  - Prophet with custom seasonalities (monthly, quarterly) and 80% intervals; backtest-like MAE/RMSE on history
  - LSTM (PyTorch) with sequence windows, train/test split, iterative horizon prediction
  - RandomForestRegressor on engineered features; simple CI from returns std
  - Ensemble weighted aggregation of model outputs
- Sentiment & LLM:
  - Transformers pipeline (configurable model), CPU-friendly fallback to DistilBERT, VADER ultimate fallback
  - Insight generation with key metrics, risk scoring, recommendations, structured highlights

## 8. Scheduler and Automation

- Async scheduler starts on app lifespan; tasks include:
  - Hourly market data update for stale symbols (or default popular list)
  - 30-minute news updates with duplicate filtering
  - Writes to both SQL and InfluxDB where available

## 9. Frontend (Streamlit)

- Authenticated dashboard with health checks and model status
- Advanced Forecasting (Prophet/LSTM/Ensemble/Compare)
- AI Insights with sentiment visualization
- Visualizations: price/technical/dashboard
- Portfolio analysis and batch forecasting

## 10. Deployment

- Dockerfile with uv-based dependency management and healthcheck
- docker-compose with services: app, postgres, redis, influxdb; `.env` support
- Local run: `uv run fastapi dev app/main.py` or `uvicorn app.main:app --reload`
- Frontend: `streamlit run frontend/advanced_streamlit_app.py`

## 11. Configuration

- `.env` (see README):
  - Required: `SECRET_KEY`, `DATABASE_URL`, `REDIS_URL`
  - Optional: `INFLUXDB_*`, `ALPHA_VANTAGE_API_KEY`, `NEWS_API_KEY`, `TWITTER_BEARER_TOKEN`, `LOG_LEVEL`

## 12. Testing & QA (Planned)

- Pytest scaffolding with coverage, ruff/black/isort/mypy configured in `pyproject.toml`
- Suggested tests: endpoint auth, data ingestion sanity, forecasting service unit tests, LLM fallback behavior

## 13. Risks & Mitigations

- Data Quality/Availability: fallback sources, caching, validation
- Model Performance Drift: retraining endpoint, periodic evaluation
- Scalability: InfluxDB for time-series, background scheduler, Redis cache
- Cost/Latency: CPU-first transformer config, progressive enhancement

## 14. Roadmap

- Phase 1 (MVP): Core backend, auth, ingestion, basic analysis, Streamlit UI
- Phase 2: Advanced ML, real-time streaming, broader integrations, richer visuals
- Phase 3: Production hardening, Kubernetes, monitoring, alerts, backtesting engine

## 15. How to Run (Quick Start)

1. Install dependencies: `pip install uv && uv sync`
2. Create `.env` (see README for keys)
3. Start backend: `uv run fastapi dev app/main.py` (or `uvicorn app.main:app --reload`)
4. Start frontend: `streamlit run frontend/advanced_streamlit_app.py`
5. Browse docs at `http://localhost:8000/docs`

## 16. Conclusion

OpenTrend AI integrates robust data engineering, classical and deep learning models, and LLM-based insights in a modular, API-first architecture. The system provides actionable analytics, forecasts, and visualizations suitable for research, portfolio exploration, and future productionization.
