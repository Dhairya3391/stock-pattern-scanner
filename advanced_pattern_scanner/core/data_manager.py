"""
Data Manager for efficient stock data fetching, caching, and preprocessing.

This module provides optimized data operations with intelligent caching,
error handling, and preprocessing capabilities specifically designed for
pattern detection workflows.
"""

import os
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import yfinance as yf
from diskcache import Cache
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

from .models import MarketData, PatternConfig
from .apple_silicon_optimizer import get_global_optimizer, get_optimal_numpy_config
from .error_handler import (
    DataError, NetworkError, error_handler, performance_monitor,
    ErrorCategory, ErrorSeverity
)

# Set up logging
logger = logging.getLogger(__name__)


class DataCache:
    """
    Intelligent caching system with TTL and size management.
    
    Uses disk-based caching for persistence across sessions with
    automatic cleanup and size management.
    """
    
    def __init__(self, cache_dir: str = ".cache", max_size_gb: float = 1.0, default_ttl: int = 3600):
        """
        Initialize the cache system.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            default_ttl: Default TTL in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize disk cache with size limit
        max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache = Cache(
            directory=str(self.cache_dir),
            size_limit=max_size_bytes,
            eviction_policy='least-recently-used'
        )
        self.default_ttl = default_ttl
        
        logger.info(f"Initialized cache at {cache_dir} with {max_size_gb}GB limit")
    
    def _generate_key(self, symbol: str, period: str, interval: str = "1d") -> str:
        """Generate a unique cache key for the data request."""
        key_string = f"{symbol}_{period}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, symbol: str, period: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and not expired.
        
        Args:
            symbol: Stock symbol
            period: Time period (e.g., '1y', '6mo')
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            Cached DataFrame or None if not available/expired
        """
        key = self._generate_key(symbol, period, interval)
        
        try:
            cached_data = self.cache.get(key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {symbol} ({period}, {interval})")
                return cached_data
        except Exception as e:
            logger.warning(f"Cache retrieval error for {key}: {e}")
        
        return None
    
    def set(self, symbol: str, period: str, data: pd.DataFrame, 
            interval: str = "1d", ttl: Optional[int] = None) -> None:
        """
        Store data in cache with TTL.
        
        Args:
            symbol: Stock symbol
            period: Time period
            data: DataFrame to cache
            interval: Data interval
            ttl: Time to live in seconds (uses default if None)
        """
        key = self._generate_key(symbol, period, interval)
        ttl = ttl or self.default_ttl
        
        try:
            self.cache.set(key, data, expire=ttl)
            logger.debug(f"Cached data for {symbol} ({period}, {interval}) with TTL {ttl}s")
        except Exception as e:
            logger.warning(f"Cache storage error for {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'volume': self.cache.volume(),
            'hits': getattr(self.cache, 'hits', 0),
            'misses': getattr(self.cache, 'misses', 0)
        }


class DataValidator:
    """Validates market data quality and completeness."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate a market data DataFrame.
        
        Args:
            df: DataFrame to validate
            symbol: Stock symbol for error reporting
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if df is None or df.empty:
            errors.append(f"No data available for {symbol}")
            return False, errors
        
        # Check required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing columns for {symbol}: {missing_columns}")
        
        # Check for sufficient data points
        if len(df) < 50:
            errors.append(f"Insufficient data points for {symbol}: {len(df)} (minimum 50)")
        
        # Check for data quality issues
        if df['High'].lt(df['Low']).any():
            errors.append(f"Invalid OHLC data for {symbol}: High < Low detected")
        
        if df['High'].lt(df['Close']).any() or df['Low'].gt(df['Close']).any():
            errors.append(f"Invalid OHLC data for {symbol}: Close outside High-Low range")
        
        # Check for excessive missing values
        missing_pct = df.isnull().sum().max() / len(df) * 100
        if missing_pct > 5:
            errors.append(f"Excessive missing data for {symbol}: {missing_pct:.1f}%")
        
        # Check for suspicious price movements (>50% in one day)
        if len(df) > 1:
            price_changes = df['Close'].pct_change().abs()
            if price_changes.max() > 0.5:
                errors.append(f"Suspicious price movement detected for {symbol}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame for analysis.
        
        Args:
            df: Raw DataFrame from data source
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Ensure datetime index
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Remove duplicates
        df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
        
        # Forward fill missing values (conservative approach)
        df_clean = df_clean.fillna(method='ffill')
        
        # Remove any remaining NaN rows
        df_clean = df_clean.dropna()
        
        # Ensure positive volumes
        if 'Volume' in df_clean.columns:
            df_clean['Volume'] = df_clean['Volume'].abs()
        
        return df_clean


class DataManager:
    """
    Main data management class for efficient stock data operations.
    
    Provides high-level interface for data fetching, caching, and preprocessing
    with built-in error handling and performance optimization.
    """
    
    def __init__(self, config: PatternConfig):
        """
        Initialize DataManager with configuration.
        
        Args:
            config: PatternConfig instance with data settings
        """
        self.config = config
        self.cache = DataCache(
            cache_dir=".cache/market_data",
            max_size_gb=1.0,
            default_ttl=config.data_cache_ttl
        )
        self.validator = DataValidator()
        
        # Apply Apple Silicon optimizations
        self.optimizer = get_global_optimizer()
        numpy_config = get_optimal_numpy_config()
        
        # Optimize thread pool for Apple Silicon
        max_workers = config.max_concurrent_requests
        if self.optimizer.is_apple_silicon:
            # Apple Silicon benefits from more conservative threading for I/O
            max_workers = min(max_workers, 4)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"DataManager initialized with Apple Silicon optimizations: {self.optimizer.is_apple_silicon}")
        if numpy_config:
            logger.info(f"NumPy optimizations applied: {numpy_config}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @error_handler(ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM)
    @performance_monitor("yfinance_fetch")
    def _fetch_from_yfinance(self, symbol: str, period: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance with retry logic.
        
        Args:
            symbol: Stock symbol
            period: Time period
            interval: Data interval
            
        Returns:
            DataFrame or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period=period,
                interval=interval,
                timeout=self.config.request_timeout
            )
            
            if df is not None and not df.empty:
                logger.debug(f"Fetched {len(df)} records for {symbol} from yfinance")
                return df
            else:
                logger.warning(f"No data returned for {symbol} from yfinance")
                return None
                
        except Exception as e:
            logger.error(f"yfinance fetch error for {symbol}: {e}")
            raise NetworkError(f"Failed to fetch data for {symbol}", endpoint="yfinance", context={"symbol": symbol, "period": period})
    
    def fetch_stock_data(self, symbol: str, period: str = "1y", 
                        interval: str = "1d", use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch stock data with caching and validation.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        symbol = symbol.upper().strip()
        
        # Try cache first if enabled
        if use_cache:
            cached_data = self.cache.get(symbol, period, interval)
            if cached_data is not None:
                is_valid, errors = self.validator.validate_dataframe(cached_data, symbol)
                if is_valid:
                    return cached_data
                else:
                    logger.warning(f"Cached data invalid for {symbol}: {errors}")
        
        # Fetch fresh data
        logger.info(f"Fetching fresh data for {symbol} ({period}, {interval})")
        
        try:
            df = self._fetch_from_yfinance(symbol, period, interval)
            
            if df is not None:
                # Validate data
                is_valid, errors = self.validator.validate_dataframe(df, symbol)
                
                if not is_valid:
                    if self.config.data_validation_strict:
                        logger.error(f"Data validation failed for {symbol}: {errors}")
                        return None
                    else:
                        logger.warning(f"Data validation warnings for {symbol}: {errors}")
                
                # Clean data
                df_clean = self.validator.clean_dataframe(df)
                
                # Cache the cleaned data
                if use_cache and df_clean is not None and not df_clean.empty:
                    self.cache.set(symbol, period, df_clean, interval)
                
                return df_clean
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        return None
    
    def fetch_multiple_stocks(self, symbols: List[str], period: str = "1y", 
                            interval: str = "1d", use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks concurrently.
        
        Args:
            symbols: List of stock symbols
            period: Time period
            interval: Data interval
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        # Submit all fetch tasks
        future_to_symbol = {
            self.executor.submit(self.fetch_stock_data, symbol, period, interval, use_cache): symbol
            for symbol in symbols
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if df is not None:
                    results[symbol] = df
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def preprocess_for_analysis(self, df: pd.DataFrame, symbol: str) -> Optional[MarketData]:
        """
        Preprocess DataFrame for pattern analysis.
        
        Args:
            df: Raw OHLCV DataFrame
            symbol: Stock symbol
            
        Returns:
            MarketData object ready for analysis
        """
        if df is None or df.empty:
            return None
        
        try:
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {symbol}")
                return None
            
            # Convert to numpy array for performance
            ohlcv_data = df[required_cols].values.astype(np.float64)
            
            # Extract timestamps
            timestamps = df.index.to_pydatetime().tolist()
            
            # Create MarketData object
            market_data = MarketData(
                symbol=symbol,
                timeframe="1d",  # TODO: Make this configurable
                data=ohlcv_data,
                timestamps=timestamps
            )
            
            logger.debug(f"Preprocessed {len(df)} data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Preprocessing error for {symbol}: {e}")
            return None
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)