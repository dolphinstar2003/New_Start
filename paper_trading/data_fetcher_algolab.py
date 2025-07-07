#!/usr/bin/env python3
"""
AlgoLab Data Fetcher for Paper Trading
Fetches real-time and historical price data using AlgoLab API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from core.algolab_api import AlgoLabAPI
from utils.algolab_auth import AlgoLabAuth
import logging

logger = logging.getLogger(__name__)


class AlgoLabDataFetcher:
    """Fetches market data using AlgoLab API"""
    
    def __init__(self):
        # Initialize AlgoLab API
        self.auth = AlgoLabAuth()
        credentials = self.auth.get_credentials()
        
        if not credentials:
            raise ValueError("AlgoLab credentials not found. Please configure in .env file")
            
        self.api = AlgoLabAPI(
            api_key=credentials['api_key'],
            username=credentials['username'],
            password=credentials['password']
        )
        
        # Check if authenticated (will use cached session if available)
        if not self.api.is_authenticated():
            logger.info("Not authenticated, attempting login...")
            # Use auth helper for interactive login if needed
            self.api = self.auth.authenticate()
            if not self.api:
                raise ConnectionError("Failed to authenticate with AlgoLab API")
            
        self.symbols = SACRED_SYMBOLS
        self.cache_dir = Path("paper_trading/data/cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for intraday data
        self.price_cache = {}
        self.last_fetch_time = None
        self.cache_duration = 30  # 30 seconds for real-time data to allow sharing between processes
        
    def get_current_prices(self, use_cache: bool = True) -> dict:
        """Get current prices for all symbols from AlgoLab"""
        # Check cache
        if use_cache and self.last_fetch_time:
            if (datetime.now() - self.last_fetch_time).seconds < self.cache_duration:
                logger.info("Using cached price data")
                return self.price_cache
        
        logger.info("Fetching current prices from AlgoLab...")
        current_prices = {}
        
        try:
            # Check if we're still authenticated
            if not self.api.is_authenticated():
                logger.warning("Session expired, re-authenticating...")
                self.api = self.auth.authenticate()
                if not self.api:
                    raise ConnectionError("Failed to re-authenticate")
            
            # Fetch candle data for each symbol
            for symbol in self.symbols:
                try:
                    # Get symbol info for current price
                    symbol_info = self.api.get_symbol_info(symbol)
                    
                    if symbol_info:
                        # According to documentation, the field is 'lst' for last price
                        if 'lst' in symbol_info:
                            current_price = float(symbol_info['lst'])
                            current_prices[symbol] = current_price
                            logger.debug(f"{symbol}: {current_price} (last)")
                        elif 'ask' in symbol_info and 'bid' in symbol_info:
                            # Use mid price if last price not available
                            ask = float(symbol_info['ask'])
                            bid = float(symbol_info['bid'])
                            current_price = (ask + bid) / 2
                            current_prices[symbol] = current_price
                            logger.debug(f"{symbol}: {current_price} (mid)")
                        else:
                            logger.warning(f"No price data in symbol info for {symbol}")
                    else:
                        logger.warning(f"No symbol info received for {symbol}")
                        
                except Exception as e:
                    logger.warning(f"Error fetching price for {symbol}: {e}")
                    continue
            
            # Update cache
            self.price_cache = current_prices
            self.last_fetch_time = datetime.now()
            
            # Save to file for backup
            self.save_price_cache(current_prices)
            
            logger.info(f"Fetched prices for {len(current_prices)} symbols")
            
        except Exception as e:
            logger.error(f"Error fetching current prices: {e}")
            # Try to load from cache file
            current_prices = self.load_price_cache()
        
        return current_prices
    
    def get_historical_data(self, symbol: str, period: str = "D", bar_count: int = 100) -> pd.DataFrame:
        """
        Get historical data for a symbol from AlgoLab
        
        Note: AlgoLab API doesn't provide historical candle data endpoint.
        This method returns empty DataFrame as historical data is not available.
        Use Yahoo Finance for historical data instead.
        
        Args:
            symbol: Stock symbol
            period: Period (1, 5, 15, 30, 60, D, W, M)
            bar_count: Number of bars to fetch
        """
        logger.warning(f"AlgoLab API doesn't provide historical data. Use Yahoo Finance instead.")
        return pd.DataFrame()
    
    def get_order_book(self, symbol: str) -> dict:
        """Get order book (depth) data for a symbol"""
        logger.warning("AlgoLab API doesn't provide depth/order book data endpoint")
        return {}
    
    def save_price_cache(self, prices: dict):
        """Save price cache to file"""
        cache_file = self.cache_dir / "latest_prices_algolab.json"
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'prices': prices
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving price cache: {e}")
    
    def load_price_cache(self) -> dict:
        """Load price cache from file"""
        cache_file = self.cache_dir / "latest_prices_algolab.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is not too old (15 minutes for real-time)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.now() - cache_time).seconds < 900:
                    return cache_data['prices']
                    
            except Exception as e:
                logger.error(f"Error loading price cache: {e}")
        
        return {}
    
    def get_market_status(self) -> dict:
        """Get market status and trading hours"""
        now = datetime.now()
        
        # BIST trading hours (Turkey time UTC+3)
        market_open = now.replace(hour=10, minute=0, second=0, microsecond=0)
        market_close = now.replace(hour=18, minute=0, second=0, microsecond=0)
        
        is_open = False
        status = "Closed"
        
        # Check if weekday
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            if market_open <= now <= market_close:
                is_open = True
                status = "Open"
            elif now < market_open:
                status = "Pre-market"
            else:
                status = "After-hours"
        else:
            status = "Weekend"
        
        return {
            'is_open': is_open,
            'status': status,
            'current_time': now.strftime('%H:%M:%S'),
            'market_open': market_open.strftime('%H:%M'),
            'market_close': market_close.strftime('%H:%M'),
            'time_to_open': str(market_open - now) if now < market_open else None,
            'time_to_close': str(market_close - now) if is_open else None
        }
    
    def get_intraday_data(self, symbol: str, interval: str = "5") -> pd.DataFrame:
        """
        Get intraday data for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Interval in minutes (1, 5, 15, 30, 60)
        """
        bar_count = {
            "1": 390,   # Full day in 1-min bars
            "5": 78,    # Full day in 5-min bars
            "15": 26,   # Full day in 15-min bars
            "30": 13,   # Full day in 30-min bars
            "60": 7     # Full day in 60-min bars
        }.get(interval, 100)
        
        return self.get_historical_data(symbol, period=interval, bar_count=bar_count)


if __name__ == "__main__":
    # Test AlgoLab data fetcher
    try:
        fetcher = AlgoLabDataFetcher()
        
        # Get market status
        print("Market Status:")
        status = fetcher.get_market_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Get current prices
        print("\nFetching current prices...")
        prices = fetcher.get_current_prices()
        
        if prices:
            print(f"\nFetched {len(prices)} prices:")
            for symbol, price in sorted(prices.items())[:5]:
                print(f"  {symbol}: {price:.2f} TL")
            print("  ...")
        else:
            print("No prices fetched")
        
        # Test historical data
        print("\nFetching historical data for GARAN...")
        hist = fetcher.get_historical_data('GARAN', period='D', bar_count=5)
        if not hist.empty:
            print(f"Got {len(hist)} days of data")
            print(hist.tail())
            
        # Test order book
        print("\nFetching order book for GARAN...")
        order_book = fetcher.get_order_book('GARAN')
        if order_book:
            print(f"Bids: {len(order_book.get('bids', []))}, Asks: {len(order_book.get('asks', []))}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()