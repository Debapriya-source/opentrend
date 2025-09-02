"""Database connection management."""

from typing import Generator
from sqlmodel import Session, create_engine, SQLModel
from sqlalchemy import text
from redis import Redis
from loguru import logger
from app.core.config import settings

# Database engine (PostgreSQL or SQLite)
engine = create_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)

# Redis client
redis_client = Redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


def create_db_and_tables() -> None:
    """Create database tables."""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def get_session() -> Generator[Session, None, None]:
    """Get database session."""
    with Session(engine) as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()


def get_redis() -> Redis:
    """Get Redis client."""
    return redis_client


def test_connections() -> bool:
    """Test database connections."""
    try:
        # Test database (PostgreSQL or SQLite)
        with Session(engine) as session:
            if "sqlite" in settings.database_url:
                session.execute(text("SELECT 1"))
                logger.info("SQLite connection successful")
            else:
                session.execute(text("SELECT 1"))
                logger.info("PostgreSQL connection successful")

        # Test Redis (optional for development)
        try:
            redis_client.ping()
            logger.info("Redis connection successful")
        except Exception as redis_error:
            logger.warning(f"Redis connection failed (optional): {redis_error}")

        # Test InfluxDB (optional for development)
        try:
            from app.services.influxdb_client import influxdb_service

            if influxdb_service.is_connected():
                logger.info("InfluxDB connection successful")
            else:
                logger.warning("InfluxDB connection failed (optional)")
        except Exception as influx_error:
            logger.warning(f"InfluxDB connection test failed (optional): {influx_error}")

        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
