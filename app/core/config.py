"""Application configuration management."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Application
    app_name: str = "OpenTrend AI"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Enable debug mode")

    # API
    api_v1_prefix: str = "/api/v1"
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key for JWT tokens")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Database
    database_url: str = Field(default="sqlite:///./opentrend.db", description="Database URL")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")

    # InfluxDB
    influxdb_url: str = Field(default="http://localhost:8086", description="InfluxDB URL")
    influxdb_token: Optional[str] = Field(default=None, description="InfluxDB token")
    influxdb_org: str = Field(default="opentrend", description="InfluxDB organization")
    influxdb_bucket: str = Field(default="metrics", description="InfluxDB bucket")

    # External APIs
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    news_api_key: Optional[str] = Field(default=None, description="News API key")
    twitter_bearer_token: Optional[str] = Field(default=None, description="Twitter API bearer token")

    # ML Models
    model_cache_dir: str = Field(default="./models", description="Directory for cached models")
    sentiment_model_name: str = Field(
        default="cardiffnlp/twitter-roberta-base-sentiment-latest", description="Sentiment analysis model"
    )

    # Data Processing
    batch_size: int = Field(default=1000, description="Batch size for data processing")
    max_workers: int = Field(default=4, description="Maximum number of worker processes")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance - will be created when needed
_settings = None


def get_settings() -> Settings:
    """Get settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# For backward compatibility
settings = get_settings()
