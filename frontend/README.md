# OpenTrend AI Frontend

This directory contains two Streamlit frontend interfaces for OpenTrend AI:

## 🔧 Basic Interface (`streamlit_app.py`)

- **Purpose**: Simple MVP interface
- **Features**: Basic forecasting, trends, data exploration
- **Usage**: `streamlit run frontend/streamlit_app.py`

## 🚀 Advanced Interface (`advanced_streamlit_app.py`)

- **Purpose**: Full-featured professional interface
- **Features**:
  - ✅ **Advanced Forecasting**: Prophet, LSTM, Ensemble models
  - ✅ **AI Insights**: LLM-powered market analysis and recommendations
  - ✅ **Advanced Visualizations**: Interactive charts, technical analysis, dashboards
  - ✅ **Portfolio Analysis**: Correlation analysis, batch forecasting
  - ✅ **Model Comparison**: Compare multiple forecasting models
  - ✅ **Professional UI**: Tabbed interface, comprehensive metrics

## 🚀 Quick Start (Advanced Interface)

1. **Start the backend**:

   ```bash
   cd opentrend/
   uv run fastapi dev app/main.py
   ```

2. **Start the advanced frontend**:

   ```bash
   streamlit run frontend/advanced_streamlit_app.py
   ```

3. **Login** using your OpenTrend credentials

4. **Explore Features**:
   - 🔮 **Advanced Forecasting**: Try Prophet, LSTM, or Ensemble models
   - 🧠 **AI Insights**: Get AI-powered market analysis
   - 📊 **Visualizations**: Create interactive charts and dashboards
   - 💼 **Portfolio Analysis**: Analyze multiple stocks and correlations

## 📊 New API Endpoints Integrated

### Forecasting Endpoints:

- `/api/v1/forecasting/prophet/{symbol}` - Facebook Prophet forecasting
- `/api/v1/forecasting/lstm/{symbol}` - LSTM neural network forecasting
- `/api/v1/forecasting/ensemble/{symbol}` - Multi-model ensemble
- `/api/v1/forecasting/compare/{symbol}` - Compare all models
- `/api/v1/forecasting/batch` - Batch forecasting for multiple symbols

### AI Insights Endpoints:

- `/api/v1/llm/insights/{symbol}` - AI-powered market insights
- `/api/v1/llm/analyze/sentiment` - Advanced sentiment analysis

### Visualization Endpoints:

- `/api/v1/visualization/price-chart/{symbol}` - Interactive price charts
- `/api/v1/visualization/technical-analysis/{symbol}` - Technical indicators
- `/api/v1/visualization/dashboard/{symbol}` - Complete dashboards
- `/api/v1/visualization/correlation-matrix` - Multi-symbol correlation
- `/api/v1/visualization/portfolio-performance` - Portfolio analysis

## 🎯 Key Features Demonstrated

1. **Professional Forecasting**:

   - Facebook Prophet for seasonal trends
   - LSTM neural networks for complex patterns
   - Ensemble methods combining multiple models
   - Model comparison and consensus forecasting

2. **AI-Powered Analysis**:

   - Sentiment analysis using transformers (RoBERTa)
   - Risk assessment and scoring
   - AI-generated market insights and recommendations
   - News sentiment visualization

3. **Advanced Visualizations**:

   - Interactive Plotly charts
   - Technical analysis with RSI, MACD, moving averages
   - Candlestick and OHLC charts
   - Portfolio correlation heatmaps

4. **Portfolio Management**:
   - Multi-symbol analysis
   - Correlation analysis for diversification
   - Batch forecasting for entire portfolios
   - Performance metrics and recommendations

## 🔧 Configuration

The frontend automatically connects to `http://localhost:8000` by default. You can configure this in Streamlit secrets or modify the `BASE_URL` variable.

## 🚀 Next Steps

The advanced interface demonstrates the full capabilities of your OpenTrend AI system, transforming it from a basic MVP into a professional-grade financial analytics platform!
