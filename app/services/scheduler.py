"""Background task scheduler for data ingestion."""

import asyncio
from datetime import datetime, timedelta
from loguru import logger
from sqlmodel import select

from app.database.connection import get_session
from app.database.models import MarketData
from app.services.data_collector import DataCollectorService
from app.services.influxdb_client import influxdb_service


class DataScheduler:
    """Scheduler for automated data collection tasks."""

    def __init__(self):
        self.data_service = DataCollectorService()
        self.is_running = False
        self._tasks = []

    async def start_scheduler(self):
        """Start the background scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return

        self.is_running = True
        logger.info("Starting data scheduler...")

        # Schedule different tasks
        self._tasks = [
            asyncio.create_task(self._market_data_updater()),
            asyncio.create_task(self._news_updater()),
        ]

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop_scheduler(self):
        """Stop the background scheduler."""
        self.is_running = False
        logger.info("Stopping data scheduler...")

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _market_data_updater(self):
        """Update market data for active symbols every hour."""
        while self.is_running:
            try:
                logger.info("Running scheduled market data update...")

                # Get list of symbols that need updates
                symbols_to_update = await self._get_symbols_needing_update()

                for symbol in symbols_to_update:
                    await self._update_symbol_data(symbol)

                logger.info(f"Updated market data for {len(symbols_to_update)} symbols")

            except Exception as e:
                logger.error(f"Error in market data updater: {e}")

            # Wait 1 hour before next update
            await asyncio.sleep(3600)

    async def _news_updater(self):
        """Update news articles every 30 minutes."""
        while self.is_running:
            try:
                logger.info("Running scheduled news update...")

                # Update with common financial keywords
                keywords = ["stock market", "trading", "finance", "economy", "earnings"]
                articles = self.data_service.collect_news_articles(keywords, days_back=1)

                # Store articles in database
                session_gen = get_session()
                session = next(session_gen)
                try:
                    from app.database.models import NewsArticle
                    from sqlmodel import select

                    new_articles_count = 0
                    for article in articles:
                        # Check if article already exists to avoid duplicates
                        existing = session.exec(select(NewsArticle).where(NewsArticle.url == article["url"])).first()

                        if not existing:
                            news_article = NewsArticle(
                                title=article["title"],
                                content=article["content"],
                                source=article["source"],
                                url=article["url"],
                                published_at=article["published_at"],
                                sentiment_score=article.get("sentiment_score"),
                                sentiment_label=article.get("sentiment_label"),
                            )
                            session.add(news_article)
                            new_articles_count += 1

                    session.commit()
                    logger.info(
                        f"Added {new_articles_count} new news articles (skipped {len(articles) - new_articles_count} duplicates)"
                    )

                except Exception as e:
                    session.rollback()
                    logger.error(f"Error storing news articles: {e}")
                finally:
                    session.close()

            except Exception as e:
                logger.error(f"Error in news updater: {e}")

            # Wait 30 minutes before next update
            await asyncio.sleep(1800)

    async def _get_symbols_needing_update(self) -> list[str]:
        """Get symbols that need data updates."""
        session_gen = get_session()
        session = next(session_gen)

        try:
            # Get symbols that haven't been updated in the last 24 hours
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            query = select(MarketData.symbol).where(MarketData.timestamp < cutoff_time).distinct()

            result = session.exec(query).all()
            symbols = [row for row in result]

            # Add some default popular symbols if no data exists
            if not symbols:
                symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]

            return symbols[:10]  # Limit to 10 symbols per update

        except Exception as e:
            logger.error(f"Error getting symbols to update: {e}")
            return []
        finally:
            session.close()

    async def _update_symbol_data(self, symbol: str):
        """Update data for a specific symbol."""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)  # Get last week of data

            data_points = self.data_service.collect_market_data(symbol, start_date, end_date)

            if not data_points:
                logger.warning(f"No data collected for {symbol}")
                return

            # Store in database
            session_gen = get_session()
            session = next(session_gen)

            try:
                new_data_points = []

                for data_point in data_points:
                    # Check if data point already exists to avoid duplicates
                    existing = session.exec(
                        select(MarketData).where(
                            MarketData.symbol == data_point["symbol"], MarketData.timestamp == data_point["timestamp"]
                        )
                    ).first()

                    if not existing:
                        market_data = MarketData(
                            symbol=data_point["symbol"],
                            timestamp=data_point["timestamp"],
                            open_price=data_point["open"],
                            high_price=data_point["high"],
                            low_price=data_point["low"],
                            close_price=data_point["close"],
                            volume=data_point["volume"],
                            source=data_point["source"],
                        )
                        session.add(market_data)
                        new_data_points.append(data_point)

                session.commit()

                # Store new data points in InfluxDB as well
                if new_data_points:
                    influx_success = influxdb_service.write_market_data(new_data_points)
                    if influx_success:
                        logger.info(f"Updated data for {symbol} in both PostgreSQL and InfluxDB")
                    else:
                        logger.warning(f"Updated data for {symbol} in PostgreSQL only (InfluxDB write failed)")
                else:
                    logger.info(f"No new data points for {symbol}")

            except Exception as e:
                session.rollback()
                logger.error(f"Error storing data for {symbol}: {e}")
            finally:
                session.close()

        except Exception as e:
            logger.error(f"Error updating symbol {symbol}: {e}")


# Global scheduler instance
scheduler = DataScheduler()
