# OpenTrend AI: Market Trend Analysis and Prediction Platform

An AI-powered platform to automatically identify, analyze, and predict market trends using various data sources.

## 🚀 Features

- **Automatic Trend Discovery**: Proactively identify emerging and declining market trends
- **Multi-Source Analysis**: Combine financial data, news sentiment, and social media
- **Real-time Processing**: Continuously monitor and update trend analysis
- **AI/ML Analysis**: Time series forecasting, sentiment analysis, and trend identification
- **Interactive Dashboard**: Streamlit-based dashboard for data visualization
- **RESTful API**: Comprehensive API for data access and analysis

## 🛠️ Tech Stack

### Backend
- **FastAPI**: High-performance web framework with automatic API documentation
- **SQLModel**: Modern SQL database library with Pydantic integration
- **PostgreSQL**: Primary relational database
- **Redis**: Caching and session management
- **InfluxDB**: Time-series database for market data
- **MinIO**: Object storage for large datasets

### AI/ML
- **Pandas & NumPy**: Data processing and analysis
- **Scikit-learn**: Traditional machine learning models
- **Prophet**: Time series forecasting
- **TensorFlow/PyTorch**: Deep learning models
- **NLTK & spaCy**: Natural language processing
- **Hugging Face Transformers**: Advanced NLP and sentiment analysis
- **VADER**: Lexicon-based sentiment analysis

### Frontend (MVP)
- **Streamlit**: Rapid dashboard development
- **Plotly**: Interactive visualizations
- **Matplotlib & Seaborn**: Statistical graphics

## 📋 Prerequisites

- Python 3.13+
- PostgreSQL
- Redis
- Docker (optional)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd opentrend
```

### 2. Install Dependencies

```bash
# Install uv if not already installed
pip install uv

# Install project dependencies
uv sync
```

### 3. Environment Setup

```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
# At minimum, set:
# - SECRET_KEY
# - DATABASE_URL
# - REDIS_URL
```

### 4. Database Setup

```bash
# Create PostgreSQL database
createdb opentrend

# Start Redis (if not running)
redis-server
```

### 5. Run the Application

```bash
# Start the FastAPI backend
python main.py

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📚 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Get current user info

### Data Management
- `POST /api/v1/data/ingest/market-data` - Ingest market data
- `POST /api/v1/data/ingest/news` - Ingest news articles
- `GET /api/v1/data/market-data/{symbol}` - Get market data
- `GET /api/v1/data/news` - Get news articles

### Analysis
- `POST /api/v1/analysis/trends/analyze` - Analyze trends
- `POST /api/v1/analysis/predictions/generate` - Generate predictions
- `GET /api/v1/analysis/trends` - Get trend analyses
- `GET /api/v1/analysis/predictions` - Get predictions
- `GET /api/v1/analysis/sentiment/current` - Get current sentiment
- `POST /api/v1/analysis/models/train` - Train models

## 🏗️ Project Structure

```
opentrend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── auth.py
│   │   │   │   ├── data.py
│   │   │   │   └── analysis.py
│   │   │   └── api.py
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── database/
│   │   ├── connection.py
│   │   └── models.py
│   ├── services/
│   │   ├── data_collector.py
│   │   └── analysis_service.py
│   └── main.py
├── main.py
├── pyproject.toml
├── env.example
└── README.md
```

## 🔧 Configuration

The application uses environment variables for configuration. Key settings include:

### Required
- `SECRET_KEY`: Secret key for JWT tokens
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

### Optional
- `DEBUG`: Enable debug mode
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key for market data
- `NEWS_API_KEY`: News API key for news articles
- `TWITTER_BEARER_TOKEN`: Twitter API bearer token

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
uv sync --group dev

# Run tests
pytest

# Run with coverage
pytest --cov=app
```

### Code Quality

```bash
# Format code
black app/

# Sort imports
isort app/

# Type checking
mypy app/

# Linting
flake8 app/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install
```

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t opentrend .

# Run container
docker run -p 8000:8000 --env-file .env opentrend
```

## 📊 Data Sources

### Market Data
- **Yahoo Finance**: Primary source via yfinance library
- **Alpha Vantage**: Alternative source (requires API key)

### News Sources
- **Reuters**: Business news RSS feeds
- **Bloomberg**: Market news RSS feeds
- **Financial Times**: Financial news RSS feeds

### Social Media
- **Twitter**: Sentiment analysis (requires API key)
- **Reddit**: Community sentiment (planned)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the API documentation at `/docs`
- Review the project outline in `project_outline.md`

## 🔮 Roadmap

### Phase 1: MVP (Current)
- ✅ Basic FastAPI backend
- ✅ Authentication system
- ✅ Data ingestion endpoints
- ✅ Basic analysis services
- 🔄 Streamlit dashboard

### Phase 2: Enhanced Features
- 🔄 Advanced ML models
- 🔄 Real-time data streaming
- 🔄 Social media integration
- 🔄 Advanced visualizations

### Phase 3: Production Ready
- 🔄 Kubernetes deployment
- 🔄 Advanced monitoring
- 🔄 Custom alerts
- 🔄 Backtesting engine

---

**OpenTrend AI** - Empowering market insights through AI-driven analysis.

