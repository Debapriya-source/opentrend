# Data Ingestion Guide for OpenTrend

This guide explains the different ways to trigger data ingestion in the OpenTrend application and how data flows through the system.

## 📊 Data Storage Architecture

OpenTrend uses a **dual-storage architecture** for optimal performance:

### PostgreSQL (Relational Database)

- **Purpose**: Transactional data, user management, metadata
- **Data**: User accounts, API keys, data source configurations
- **Use Cases**: Authentication, data management, complex queries

### InfluxDB (Time-Series Database)

- **Purpose**: High-frequency market data and metrics
- **Data**: OHLCV data, real-time prices, volume metrics
- **Use Cases**: Charting, technical analysis, time-series queries
- **Optimizations**: Automatic downsampling, efficient compression

**Data Flow**: Market data is simultaneously stored in both databases - PostgreSQL for relational queries and InfluxDB for time-series visualization.

## 🚀 Available Methods

### 1. **Automatic Background Scheduler (Recommended)**

The application now includes an automatic background scheduler that runs continuously:

- **Market Data**: Updates every hour for active symbols
- **News Articles**: Updates every 30 minutes with financial keywords
- **Smart Symbol Detection**: Automatically detects which symbols need updates
- **Duplicate Prevention**: Automatically skips duplicate articles and data points

**Setup**: No additional setup required - starts automatically with the application.

```bash
# The scheduler starts automatically when you run:
python main.py
```

**Logs**: Check the application logs for scheduler activity:

```bash
tail -f logs/app.log | grep "scheduler"
```

### 2. **Time-Series API Endpoints (New!)**

#### Query Market Data from InfluxDB

```bash
# Get time-series data for a single symbol
curl -X GET "http://localhost:8000/api/v1/timeseries/market-data/AAPL?start_time=2024-01-01T00:00:00Z&end_time=2024-01-31T23:59:59Z&aggregation_window=1h" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get OHLCV data optimized for charting
curl -X GET "http://localhost:8000/api/v1/timeseries/market-data/AAPL/ohlcv?start_time=2024-01-01T00:00:00Z&end_time=2024-01-31T23:59:59Z&aggregation_window=1d" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Compare multiple symbols
curl -X GET "http://localhost:8000/api/v1/timeseries/market-data/compare?symbols=AAPL&symbols=GOOGL&symbols=MSFT&start_time=2024-01-01T00:00:00Z&end_time=2024-01-31T23:59:59Z&field=close&normalize=true" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get latest data point
curl -X GET "http://localhost:8000/api/v1/timeseries/market-data/AAPL/latest" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Check InfluxDB health
curl -X GET "http://localhost:8000/api/v1/timeseries/health" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### Aggregation Windows

- `1m`, `5m`, `15m`, `30m` - Minute intervals
- `1h`, `4h`, `12h` - Hour intervals
- `1d`, `1w`, `1mo` - Day/week/month intervals

### 3. **Manual API Endpoints**

#### Batch Market Data Ingestion

```bash
# Ingest data for multiple symbols at once
curl -X POST "http://localhost:8000/api/v1/data/batch/market-data" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
    "days_back": 30
  }'
```

#### Single Symbol Market Data

```bash
# Ingest data for a single symbol
curl -X POST "http://localhost:8000/api/v1/data/ingest/market-data" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "start_date": "2024-01-01T00:00:00",
    "end_date": "2024-01-31T23:59:59"
  }'
```

#### News Articles Ingestion

```bash
# Ingest news articles
curl -X POST "http://localhost:8000/api/v1/data/ingest/news" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "keywords": ["stock market", "trading", "finance"],
    "days_back": 7
  }'
```

#### Trigger Scheduled Update

```bash
# Manually trigger the scheduler
curl -X POST "http://localhost:8000/api/v1/data/scheduler/trigger" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "update_type": "all"
  }'
```

### 3. **Cron Jobs (External Scheduling)**

For more control over scheduling, use the provided cron script:

#### Setup Authentication

```bash
# Set environment variables for cron authentication
export CRON_USERNAME="your_username"
export CRON_PASSWORD="your_password"
```

#### Manual Execution

```bash
# Market data only
python scripts/data_ingestion_cron.py --type market --symbols "AAPL,GOOGL,MSFT"

# News data only
python scripts/data_ingestion_cron.py --type news --keywords "stock market,finance"

# Full update
python scripts/data_ingestion_cron.py --type all
```

#### Cron Schedule Examples

Add these to your crontab (`crontab -e`):

```bash
# Update market data every hour during market hours (9 AM - 4 PM EST, Mon-Fri)
0 9-16 * * 1-5 cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type market

# Update news every 30 minutes
*/30 * * * * cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type news

# Full update daily at midnight
0 0 * * * cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type all

# Weekend maintenance - full update
0 2 * * 6 cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type all --days-back 30
```

### 4. **On-Demand via Analysis Endpoints**

The application automatically ingests data when needed:

```bash
# This will automatically ingest data if it doesn't exist
curl -X POST "http://localhost:8000/api/v1/analysis/predictions/generate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "prediction_type": "price",
    "horizon_days": 30
  }'
```

## 📊 Monitoring Data Ingestion

### Check Data Availability

```bash
# Check what data is available
curl -X GET "http://localhost:8000/api/v1/analysis/market-data/AAPL?days=30" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### View Logs

```bash
# Application logs
tail -f logs/app.log

# Error logs
tail -f logs/errors.log

# Cron job logs (if using cron)
tail -f /tmp/opentrend_cron.log
```

## 🎯 Recommended Setup

### For Development

- Use the **automatic background scheduler** (default)
- Manually trigger updates via API endpoints when testing

### For Production

- Use the **automatic background scheduler** for continuous updates
- Set up **cron jobs** as backup/additional scheduling
- Monitor logs and set up alerting

### For High-Frequency Trading

- Use **cron jobs** with more frequent schedules
- Consider dedicated data ingestion services
- Implement real-time data streaming

## ⚙️ Configuration

### Environment Variables

```bash
# Authentication for cron jobs
CRON_USERNAME=your_api_username
CRON_PASSWORD=your_api_password

# API endpoints
API_BASE_URL=http://localhost:8000

# Data sources
ALPHA_VANTAGE_API_KEY=your_key
NEWS_API_KEY=your_key
```

### Popular Symbols List

The system includes these popular symbols by default:

- **Stocks**: AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META
- **ETFs**: SPY, QQQ, IWM, VTI
- **Crypto**: BTC-USD, ETH-USD

## 🚨 Troubleshooting

### Common Issues

1. **Authentication Errors**

   ```bash
   # Check if your token is valid
   curl -X GET "http://localhost:8000/api/v1/auth/me" \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

2. **Data Source Limits**

   - Yahoo Finance (yfinance) has rate limits
   - Consider adding delays between requests for large batches

3. **Database Locks**

   - The system handles duplicates automatically
   - Check logs for database connection issues

4. **Network Timeouts**
   - Increase timeout values for large data requests
   - Consider breaking large requests into smaller batches

### Health Checks

```bash
# Check application health
curl -X GET "http://localhost:8000/health"

# Check database connectivity
curl -X GET "http://localhost:8000/api/v1/data/market-data/AAPL?days=1" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 📈 Best Practices

1. **Scheduling**

   - Market data: Update during/after market hours
   - News: Update frequently (every 15-30 minutes)
   - Avoid overlapping heavy operations

2. **Error Handling**

   - Monitor logs regularly
   - Set up alerts for failed ingestion
   - Have backup data sources

3. **Performance**

   - Use batch endpoints for multiple symbols
   - Implement exponential backoff for API failures
   - Monitor database size and performance

4. **Data Quality**
   - Validate data before storage
   - Handle missing data gracefully
   - Implement data quality checks
