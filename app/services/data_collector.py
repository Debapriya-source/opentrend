"""Data collection service for market data and news articles."""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import feedparser
from loguru import logger

from app.core.config import settings


class DataCollectorService:
    """Service for collecting market data and news articles."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
    
    def collect_market_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Collect market data for a specific symbol."""
        try:
            logger.info(f"Collecting market data for {symbol} from {start_date} to {end_date}")
            
            # Use yfinance to get market data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            data_points = []
            for timestamp, row in data.iterrows():
                data_point = {
                    "symbol": symbol,
                    "timestamp": timestamp.to_pydatetime(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "source": "yfinance",
                }
                data_points.append(data_point)
            
            logger.info(f"Collected {len(data_points)} data points for {symbol}")
            return data_points
            
        except Exception as e:
            logger.error(f"Error collecting market data for {symbol}: {e}")
            raise
    
    def collect_news_articles(
        self, keywords: List[str], sources: List[str] = None, days_back: int = 7
    ) -> List[Dict[str, Any]]:
        """Collect news articles based on keywords."""
        try:
            logger.info(f"Collecting news articles for keywords: {keywords}")
            
            articles = []
            
            # Default sources if none provided
            if not sources:
                sources = [
                    "https://feeds.reuters.com/reuters/businessNews",
                    "https://feeds.bloomberg.com/markets/news.rss",
                    "https://www.ft.com/rss/home",
                ]
            
            for source in sources:
                try:
                    source_articles = self._collect_from_rss(source, keywords, days_back)
                    articles.extend(source_articles)
                except Exception as e:
                    logger.warning(f"Error collecting from {source}: {e}")
                    continue
            
            logger.info(f"Collected {len(articles)} news articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting news articles: {e}")
            raise
    
    def _collect_from_rss(
        self, rss_url: str, keywords: List[str], days_back: int
    ) -> List[Dict[str, Any]]:
        """Collect articles from RSS feed."""
        try:
            feed = feedparser.parse(rss_url)
            articles = []
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    pub_date = datetime(*entry.published_parsed[:6])
                    
                    if pub_date < cutoff_date:
                        continue
                    
                    # Check if article contains any keywords
                    title = entry.title.lower()
                    summary = getattr(entry, 'summary', '').lower()
                    
                    if not any(keyword.lower() in title or keyword.lower() in summary 
                             for keyword in keywords):
                        continue
                    
                    # Extract content
                    content = self._extract_article_content(entry.link)
                    
                    article = {
                        "title": entry.title,
                        "content": content or entry.summary,
                        "source": feed.feed.title if hasattr(feed.feed, 'title') else rss_url,
                        "url": entry.link,
                        "published_at": pub_date,
                        "sentiment_score": None,  # Will be calculated later
                        "sentiment_label": None,
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Error processing RSS entry: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {e}")
            return []
    
    def _extract_article_content(self, url: str) -> str:
        """Extract article content from URL."""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                'main',
                '.content',
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            # Fallback to body text if no content found
            if not content:
                content = soup.get_text().strip()
            
            # Clean up content
            content = ' '.join(content.split())
            
            return content[:5000]  # Limit content length
            
        except Exception as e:
            logger.warning(f"Error extracting content from {url}: {e}")
            return ""
    
    def collect_social_media_sentiment(
        self, keywords: List[str], platform: str = "twitter"
    ) -> List[Dict[str, Any]]:
        """Collect social media sentiment data."""
        # This would integrate with Twitter API, Reddit API, etc.
        # For now, return empty list as placeholder
        logger.info(f"Social media sentiment collection not yet implemented for {platform}")
        return []

