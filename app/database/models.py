"""Database models for the application."""

from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel
from pydantic import EmailStr


class UserBase(SQLModel):
    """Base user model."""
    email: EmailStr = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    full_name: Optional[str] = None
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)


class User(UserBase, table=True):
    """User model for database."""
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserCreate(UserBase):
    """User creation model."""
    password: str


class UserUpdate(SQLModel):
    """User update model."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None


class UserRead(UserBase):
    """User read model."""
    id: int
    created_at: datetime
    updated_at: datetime


class MarketDataBase(SQLModel):
    """Base market data model."""
    symbol: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    source: str = Field(default="yfinance")


class MarketData(MarketDataBase, table=True):
    """Market data model for database."""
    __tablename__ = "market_data"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class NewsArticleBase(SQLModel):
    """Base news article model."""
    title: str
    content: str
    source: str
    url: str = Field(unique=True, index=True)
    published_at: datetime = Field(index=True)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None


class NewsArticle(NewsArticleBase, table=True):
    """News article model for database."""
    __tablename__ = "news_articles"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TrendAnalysisBase(SQLModel):
    """Base trend analysis model."""
    symbol: str = Field(index=True)
    trend_type: str  # "bullish", "bearish", "neutral"
    confidence_score: float
    timeframe: str  # "short", "medium", "long"
    analysis_date: datetime = Field(index=True)
    indicators: str = Field(default="{}")  # JSON string for technical indicators
    description: str


class TrendAnalysis(TrendAnalysisBase, table=True):
    """Trend analysis model for database."""
    __tablename__ = "trend_analyses"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionBase(SQLModel):
    """Base prediction model."""
    symbol: str = Field(index=True)
    prediction_type: str  # "price", "trend", "volatility"
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_date: datetime = Field(index=True)
    model_used: str
    features_used: str = Field(default="{}")  # JSON string for features


class Prediction(PredictionBase, table=True):
    """Prediction model for database."""
    __tablename__ = "predictions"
    __table_args__ = {"extend_existing": True}
    
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

