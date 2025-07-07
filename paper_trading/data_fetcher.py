#!/usr/bin/env python3
"""
Data Fetcher for Paper Trading
Fetches real-time and historical price data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import yfinance as yf
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data for paper trading"""
    
    def __init__(self):
        # Add .IS suffix for Istanbul Stock Exchange
        self.symbols = [f"{symbol}.IS" for symbol in SACRED_SYMBOLS]
        self.cache_dir = Path("paper_trading/data/cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for intraday data
        self.price_cache = {}
        self.last_fetch_time = None
        self.cache_duration = 300  # 5 minutes
        
    def get_current_prices(self, use_cache: bool = True) -> dict:
        """Get current prices for all symbols"""
        # Check cache
        if use_cache and self.last_fetch_time:
            if (datetime.now() - self.last_fetch_time).seconds < self.cache_duration:
                logger.info("Using cached price data")
                return self.price_cache
        
        logger.info("Fetching current prices...")
        current_prices = {}
        
        try:
            # Fetch data for all symbols at once
            tickers = yf.Tickers(' '.join(self.symbols))
            
            for symbol in self.symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    
                    # Get current price (try multiple fields)
                    current_price = None
                    
                    if 'currentPrice' in info and info['currentPrice']:
                        current_price = info['currentPrice']
                    elif 'regularMarketPrice' in info and info['regularMarketPrice']:
                        current_price = info['regularMarketPrice']
                    elif 'previousClose' in info and info['previousClose']:
                        current_price = info['previousClose']
                    
                    if current_price:
                        # Remove .IS suffix for internal use
                        clean_symbol = symbol.replace('.IS', '')
                        current_prices[clean_symbol] = float(current_price)
                        
                except Exception as e:
                    logger.warning(f"Error fetching price for {symbol}: {e}")
                    continue
            
            # If live data fails, use last close from historical data
            if len(current_prices) < len(SACRED_SYMBOLS) / 2:
                logger.warning("Live data incomplete, fetching historical prices...")
                current_prices = self.get_last_close_prices()
            
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
    
    def get_last_close_prices(self) -> dict:
        """Get last closing prices from historical data"""
        prices = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get last 5 days of data
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    last_close = hist['Close'].iloc[-1]
                    clean_symbol = symbol.replace('.IS', '')
                    prices[clean_symbol] = float(last_close)
                    
            except Exception as e:
                logger.warning(f"Error fetching historical data for {symbol}: {e}")
                continue
        
        return prices
    
    def get_historical_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Get historical data for a symbol"""
        # Add .IS suffix if not present
        if not symbol.endswith('.IS'):
            symbol = f"{symbol}.IS"
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Clean column names
                hist.columns = [col.lower() for col in hist.columns]
                hist.index.name = 'date'
                return hist
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def save_price_cache(self, prices: dict):
        """Save price cache to file"""
        cache_file = self.cache_dir / "latest_prices.json"
        
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
        cache_file = self.cache_dir / "latest_prices.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is not too old (1 day)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                if (datetime.now() - cache_time).days < 1:
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
    
    def simulate_price_movement(self, base_prices: dict, volatility: float = 0.02) -> dict:
        """Simulate price movements for testing"""
        simulated_prices = {}
        
        for symbol, base_price in base_prices.items():
            # Random walk with mean reversion
            change = np.random.normal(0, volatility)
            change = np.clip(change, -0.05, 0.05)  # Limit to 5% moves
            
            new_price = base_price * (1 + change)
            simulated_prices[symbol] = round(new_price, 2)
        
        return simulated_prices


# Fallback data for testing when market is closed
FALLBACK_PRICES = {
    'GARAN': 45.50,
    'AKBNK': 32.80,
    'ISCTR': 12.45,
    'YKBNK': 17.25,
    'SAHOL': 28.90,
    'KCHOL': 39.75,
    'SISE': 15.60,
    'EREGL': 48.20,
    'KRDMD': 8.95,
    'TUPRS': 185.40,
    'ASELS': 78.90,
    'THYAO': 285.60,
    'TCELL': 62.45,
    'BIMAS': 435.20,
    'MGROS': 168.90,
    'ULKER': 52.30,
    'AKSEN': 24.75,
    'ENKAI': 14.85,
    'PETKM': 19.40,
    'KOZAL': 125.60
}


if __name__ == "__main__":
    # Test data fetcher
    fetcher = DataFetcher()
    
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
        print("No prices fetched, using fallback data")
        prices = FALLBACK_PRICES
    
    # Test historical data
    print("\nFetching historical data for GARAN...")
    hist = fetcher.get_historical_data('GARAN', period='5d')
    if not hist.empty:
        print(f"Got {len(hist)} days of data")
        print(hist.tail())