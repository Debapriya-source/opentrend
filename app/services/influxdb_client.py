"""InfluxDB client service for time-series data operations."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from app.core.config import get_settings


class InfluxDBService:
    """Service for InfluxDB operations."""

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        self.write_api = None
        self.query_api = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize InfluxDB client."""
        try:
            if not self.settings.influxdb_token:
                logger.warning("InfluxDB token not configured, skipping initialization")
                return

            self.client = InfluxDBClient(
                url=self.settings.influxdb_url, token=self.settings.influxdb_token, org=self.settings.influxdb_org
            )
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            self.query_api = self.client.query_api()
            logger.info("InfluxDB client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB client: {e}")
            self.client = None

    def is_connected(self) -> bool:
        """Check if InfluxDB client is connected."""
        try:
            if not self.client:
                return False
            # Ping the server
            health = self.client.health()
            return health.status == "pass"
        except Exception as e:
            logger.error(f"InfluxDB connection check failed: {e}")
            return False

    def write_market_data(self, market_data: List[Dict[str, Any]]) -> bool:
        """Write market data points to InfluxDB."""
        if not self.client or not self.write_api:
            logger.warning("InfluxDB client not initialized, skipping write")
            return False

        try:
            points = []
            for data_point in market_data:
                point = (
                    Point("market_data")
                    .tag("symbol", data_point["symbol"])
                    .tag("source", data_point.get("source", "unknown"))
                    .field("open", float(data_point["open"]))
                    .field("high", float(data_point["high"]))
                    .field("low", float(data_point["low"]))
                    .field("close", float(data_point["close"]))
                    .field("volume", int(data_point["volume"]))
                    .time(data_point["timestamp"], WritePrecision.S)
                )
                points.append(point)

            self.write_api.write(bucket=self.settings.influxdb_bucket, org=self.settings.influxdb_org, record=points)

            logger.info(f"Successfully wrote {len(points)} market data points to InfluxDB")
            return True

        except Exception as e:
            logger.error(f"Error writing market data to InfluxDB: {e}")
            return False

    def write_single_market_data(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: int,
        source: str = "unknown",
    ) -> bool:
        """Write a single market data point to InfluxDB."""
        if not self.client or not self.write_api:
            logger.warning("InfluxDB client not initialized, skipping write")
            return False

        try:
            point = (
                Point("market_data")
                .tag("symbol", symbol)
                .tag("source", source)
                .field("open", float(open_price))
                .field("high", float(high_price))
                .field("low", float(low_price))
                .field("close", float(close_price))
                .field("volume", int(volume))
                .time(timestamp, WritePrecision.S)
            )

            self.write_api.write(bucket=self.settings.influxdb_bucket, org=self.settings.influxdb_org, record=point)

            logger.debug(f"Successfully wrote market data point for {symbol} to InfluxDB")
            return True

        except Exception as e:
            logger.error(f"Error writing single market data point to InfluxDB: {e}")
            return False

    def query_market_data(
        self, symbol: str, start_time: datetime, end_time: datetime, aggregation_window: str = "1h"
    ) -> List[Dict[str, Any]]:
        """Query market data from InfluxDB."""
        if not self.client or not self.query_api:
            logger.warning("InfluxDB client not initialized, cannot query")
            return []

        try:
            # Build Flux query
            query = f'''
                from(bucket: "{self.settings.influxdb_bucket}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "market_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> aggregateWindow(every: {aggregation_window}, fn: last, createEmpty: false)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''

            result = self.query_api.query(org=self.settings.influxdb_org, query=query)

            data_points = []
            for table in result:
                for record in table.records:
                    data_point = {
                        "timestamp": record.get_time(),
                        "symbol": record.values.get("symbol"),
                        "open": record.values.get("open"),
                        "high": record.values.get("high"),
                        "low": record.values.get("low"),
                        "close": record.values.get("close"),
                        "volume": record.values.get("volume"),
                        "source": record.values.get("source"),
                    }
                    data_points.append(data_point)

            logger.info(f"Retrieved {len(data_points)} data points for {symbol} from InfluxDB")
            return data_points

        except Exception as e:
            logger.error(f"Error querying market data from InfluxDB: {e}")
            return []

    def query_multiple_symbols(
        self, symbols: List[str], start_time: datetime, end_time: datetime, aggregation_window: str = "1h"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Query market data for multiple symbols from InfluxDB."""
        if not self.client or not self.query_api:
            logger.warning("InfluxDB client not initialized, cannot query")
            return {}

        try:
            # Build Flux query for multiple symbols
            symbols_filter = " or ".join([f'r["symbol"] == "{symbol}"' for symbol in symbols])

            query = f'''
                from(bucket: "{self.settings.influxdb_bucket}")
                |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
                |> filter(fn: (r) => r["_measurement"] == "market_data")
                |> filter(fn: (r) => {symbols_filter})
                |> aggregateWindow(every: {aggregation_window}, fn: last, createEmpty: false)
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''

            result = self.query_api.query(org=self.settings.influxdb_org, query=query)

            symbol_data: Dict[str, List[Dict[str, Any]]] = {symbol: [] for symbol in symbols}

            for table in result:
                for record in table.records:
                    symbol = record.values.get("symbol")
                    if symbol in symbol_data:
                        data_point = {
                            "timestamp": record.get_time(),
                            "symbol": symbol,
                            "open": record.values.get("open"),
                            "high": record.values.get("high"),
                            "low": record.values.get("low"),
                            "close": record.values.get("close"),
                            "volume": record.values.get("volume"),
                            "source": record.values.get("source"),
                        }
                        symbol_data[symbol].append(data_point)

            total_points = sum(len(points) for points in symbol_data.values())
            logger.info(f"Retrieved {total_points} data points for {len(symbols)} symbols from InfluxDB")
            return symbol_data

        except Exception as e:
            logger.error(f"Error querying multiple symbols from InfluxDB: {e}")
            return {}

    def get_latest_data_point(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest data point for a symbol."""
        if not self.client or not self.query_api:
            logger.warning("InfluxDB client not initialized, cannot query")
            return None

        try:
            query = f'''
                from(bucket: "{self.settings.influxdb_bucket}")
                |> range(start: -30d)
                |> filter(fn: (r) => r["_measurement"] == "market_data")
                |> filter(fn: (r) => r["symbol"] == "{symbol}")
                |> last()
                |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            '''

            result = self.query_api.query(org=self.settings.influxdb_org, query=query)

            for table in result:
                for record in table.records:
                    return {
                        "timestamp": record.get_time(),
                        "symbol": record.values.get("symbol"),
                        "open": record.values.get("open"),
                        "high": record.values.get("high"),
                        "low": record.values.get("low"),
                        "close": record.values.get("close"),
                        "volume": record.values.get("volume"),
                        "source": record.values.get("source"),
                    }

            return None

        except Exception as e:
            logger.error(f"Error getting latest data point for {symbol}: {e}")
            return None

    def close(self):
        """Close InfluxDB client connection."""
        if self.client:
            self.client.close()
            logger.info("InfluxDB client connection closed")


# Global InfluxDB service instance
influxdb_service = InfluxDBService()
