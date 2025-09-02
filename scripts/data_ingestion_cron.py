#!/usr/bin/env python3
"""
Cron job script for automated data ingestion.

Usage:
    python scripts/data_ingestion_cron.py --type market --symbols AAPL,GOOGL,MSFT
    python scripts/data_ingestion_cron.py --type news --keywords "stock market,finance"
    python scripts/data_ingestion_cron.py --type all

Cron examples:
    # Update market data every hour during market hours (9 AM - 4 PM EST, Mon-Fri)
    0 9-16 * * 1-5 cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type market

    # Update news every 30 minutes
    */30 * * * * cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type news

    # Full update daily at midnight
    0 0 * * * cd /path/to/opentrend && python scripts/data_ingestion_cron.py --type all
"""

import sys
import os
import argparse
import requests
from typing import List, Optional

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger


class DataIngestionCron:
    """Cron job handler for data ingestion."""

    def __init__(
        self, base_url: str = "http://localhost:8000", username: Optional[str] = None, password: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.username = username or os.getenv("CRON_USERNAME")
        self.password = password or os.getenv("CRON_PASSWORD")

        # Authenticate if credentials provided
        if self.username and self.password:
            self._authenticate()

    def _authenticate(self):
        """Authenticate with the API."""
        try:
            login_data = {
                "username": self.username,
                "password": self.password,
            }

            response = self.session.post(f"{self.base_url}/api/v1/auth/login", data=login_data, timeout=10)

            if response.status_code == 200:
                token_data = response.json()
                self.session.headers.update({"Authorization": f"Bearer {token_data['access_token']}"})
                logger.info("Successfully authenticated with API")
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Error during authentication: {e}")
            sys.exit(1)

    def ingest_market_data(self, symbols: List[str], days_back: int = 7):
        """Ingest market data for specified symbols."""
        try:
            logger.info(f"Starting market data ingestion for symbols: {symbols}")

            response = self.session.post(
                f"{self.base_url}/api/v1/data/batch/market-data",
                json={"symbols": symbols, "days_back": days_back},
                timeout=300,  # 5 minutes timeout for batch operations
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Market data ingestion completed: {result['message']}")
                logger.info(f"Total data points: {result['total_data_points']}")

                # Log individual symbol results
                for symbol_result in result["results"]:
                    if symbol_result["status"] == "success":
                        logger.info(f"✓ {symbol_result['symbol']}: {symbol_result['data_points']} points")
                    else:
                        logger.error(f"✗ {symbol_result['symbol']}: {symbol_result['error']}")

            else:
                logger.error(f"Market data ingestion failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error during market data ingestion: {e}")

    def ingest_news_data(self, keywords: List[str], days_back: int = 1):
        """Ingest news articles for specified keywords."""
        try:
            logger.info(f"Starting news ingestion for keywords: {keywords}")

            response = self.session.post(
                f"{self.base_url}/api/v1/data/ingest/news",
                json={"keywords": keywords, "days_back": days_back},
                timeout=120,  # 2 minutes timeout
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"News ingestion completed: {result['message']}")
                logger.info(f"Articles collected: {result['articles']}")
            else:
                logger.error(f"News ingestion failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error during news ingestion: {e}")

    def run_full_update(self):
        """Run a complete data update."""
        logger.info("Starting full data update...")

        # Popular stock symbols
        market_symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "SPY",
            "QQQ",
            "IWM",
            "VTI",
            "BTC-USD",
            "ETH-USD",
        ]

        # Financial news keywords
        news_keywords = [
            "stock market",
            "trading",
            "finance",
            "economy",
            "earnings",
            "federal reserve",
            "inflation",
            "GDP",
            "unemployment",
        ]

        # Ingest market data
        self.ingest_market_data(market_symbols, days_back=7)

        # Ingest news data
        self.ingest_news_data(news_keywords, days_back=1)

        logger.info("Full data update completed")


def main():
    """Main function for cron job execution."""
    parser = argparse.ArgumentParser(description="OpenTrend Data Ingestion Cron Job")
    parser.add_argument("--type", choices=["market", "news", "all"], required=True, help="Type of data to ingest")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols for market data")
    parser.add_argument("--keywords", type=str, help="Comma-separated list of keywords for news data")
    parser.add_argument("--days-back", type=int, default=7, help="Number of days to look back for data")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Base URL of the OpenTrend API")
    parser.add_argument("--username", type=str, help="Username for API authentication")
    parser.add_argument("--password", type=str, help="Password for API authentication")

    args = parser.parse_args()

    # Setup logging
    logger.add("/tmp/opentrend_cron.log", rotation="1 week", retention="4 weeks", level="INFO")

    # Initialize cron job handler
    cron = DataIngestionCron(base_url=args.base_url, username=args.username, password=args.password)

    try:
        if args.type == "market":
            symbols = args.symbols.split(",") if args.symbols else ["AAPL", "GOOGL", "MSFT"]
            cron.ingest_market_data(symbols, args.days_back)

        elif args.type == "news":
            keywords = args.keywords.split(",") if args.keywords else ["stock market", "finance"]
            cron.ingest_news_data(keywords, args.days_back)

        elif args.type == "all":
            cron.run_full_update()

    except KeyboardInterrupt:
        logger.info("Cron job interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Cron job failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
