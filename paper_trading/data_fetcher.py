#!/usr/bin/env python3
"""
Data Fetcher for Paper Trading
Fetches real-time and historical price data using AlgoLab API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data for paper trading using AlgoLab API"""
    
    def __init__(self, use_algolab=True):
        self.use_algolab = use_algolab
        self.symbols = SACRED_SYMBOLS
        self.cache_dir = Path("paper_trading/data/cache")
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache for intraday data
        self.price_cache = {}
        self.last_fetch_time = None
        self.cache_duration = 60 if use_algolab else 300  # 1 min for AlgoLab, 5 min for yfinance
        
        # Initialize AlgoLab API if enabled
        self.algolab_fetcher = None
        if use_algolab:
            self._init_algolab()
        
        # Prepare yfinance symbols as fallback
        self.yf_symbols = [f"{symbol}.IS" for symbol in SACRED_SYMBOLS]
        
    def _init_algolab(self):
        """Initialize AlgoLab API connection"""
        try:
            # Import AlgoLab data fetcher
            from paper_trading.data_fetcher_algolab import AlgoLabDataFetcher
            self.algolab_fetcher = AlgoLabDataFetcher()
            logger.info("AlgoLab data fetcher initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AlgoLab: {e}")
            self.use_algolab = False
            self.algolab_fetcher = None
        
    def get_current_prices(self, use_cache: bool = True) -> dict:
        """Get current prices for all symbols"""
        # Check cache
        if use_cache and self.last_fetch_time:
            if (datetime.now() - self.last_fetch_time).seconds < self.cache_duration:
                logger.info("Using cached price data")
                return self.price_cache
        
        logger.info(f"Fetching current prices using {'AlgoLab' if self.use_algolab else 'yfinance'}...")
        current_prices = {}
        
        try:
            if self.use_algolab and self.algolab_fetcher:
                # Use AlgoLab API - with cache support for concurrent access
                current_prices = self.algolab_fetcher.get_current_prices(use_cache=True)
            else:
                # Use yfinance as fallback
                import yfinance as yf
                # Use download method which is more reliable
                for symbol in self.symbols:
                    try:
                        yf_symbol = f"{symbol}.IS"
                        ticker = yf.Ticker(yf_symbol)
                        hist = ticker.history(period="1d", interval="1m")
                        
                        if not hist.empty:
                            current_price = float(hist['Close'].iloc[-1])
                            current_prices[symbol] = current_price
                        else:
                            logger.warning(f"No data for {symbol}")
                            
                    except Exception as e:
                        logger.warning(f"Error fetching price for {symbol}: {e}")
                        continue
            
            # If live data fails, load from cache
            if len(current_prices) < len(SACRED_SYMBOLS) / 2:
                logger.warning("Live data incomplete, loading from cache...")
                cached_prices = self.load_price_cache()
                if cached_prices:
                    current_prices = cached_prices
                else:
                    logger.error("No cached prices available")
            
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
            if not current_prices:
                current_prices = FALLBACK_PRICES.copy()
        
        return current_prices
    
    def get_last_close_prices(self) -> dict:
        """Get last closing prices from historical data"""
        prices = {}
        
        if self.use_algolab and self.api:
            # Use AlgoLab for historical data
            for symbol in self.symbols:
                try:
                    # Get symbol info for last close price
                    symbol_info = self.api.get_symbol_info(symbol)
                    
                    if symbol_info and symbol_info.get('success'):
                        content = symbol_info.get('content', {})
                        price = content.get('lst')  # Last price
                        if price:
                            prices[symbol] = float(price)
                        
                except Exception as e:
                    logger.warning(f"Error fetching historical data for {symbol}: {e}")
                    continue
        else:
            # Use yfinance
            import yfinance as yf
            for symbol in self.symbols:
                try:
                    yf_symbol = f"{symbol}.IS"
                    ticker = yf.Ticker(yf_symbol)
                    
                    # Get last 5 days of data
                    hist = ticker.history(period="5d")
                    
                    if not hist.empty:
                        last_close = hist['Close'].iloc[-1]
                        prices[symbol] = float(last_close)
                        
                except Exception as e:
                    logger.warning(f"Error fetching historical data for {symbol}: {e}")
                    continue
        
        return prices
    
    def get_historical_data(self, symbol: str, period: str = "1mo", bar_count: int = None) -> pd.DataFrame:
        """Get historical data for a symbol"""
        
        if self.use_algolab and self.api:
            # Convert period to AlgoLab format and determine bar_count
            period_map = {
                "1d": ("5", 78),    # 5-min bars for 1 day
                "5d": ("30", 65),   # 30-min bars for 5 days
                "1mo": ("D", 22),   # Daily bars for 1 month
                "3mo": ("D", 66),   # Daily bars for 3 months
                "6mo": ("D", 132),  # Daily bars for 6 months
                "1y": ("D", 252),   # Daily bars for 1 year
            }
            
            algolab_period, default_bars = period_map.get(period, ("D", 100))
            bar_count = bar_count or default_bars
            
            try:
                # Fetch candle data
                candles = self.api.get_candles(
                    symbol=symbol,
                    period=algolab_period,
                    bar_count=bar_count
                )
                
                if not candles:
                    logger.warning(f"No historical data for {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('date', inplace=True)
                
                # Convert price columns to float
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    df[col] = df[col].astype(float)
                
                # Add volume if available
                if 'volume' in df.columns:
                    df['volume'] = df['volume'].astype(float)
                
                # Sort by date
                df.sort_index(inplace=True)
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching AlgoLab data for {symbol}: {e}")
        
        # Use yfinance as fallback
        if not symbol.endswith('.IS'):
            symbol = f"{symbol}.IS"
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Clean column names
                hist.columns = [col.lower() for col in hist.columns]
                hist.index.name = 'date'
                return hist
                
        except Exception as e:
            logger.error(f"Error fetching yfinance data for {symbol}: {e}")
        
        return pd.DataFrame()
    
    def save_price_cache(self, prices: dict):
        """Save price cache to file"""
        cache_file = self.cache_dir / "latest_prices_algolab.json" if self.use_algolab else self.cache_dir / "latest_prices.json"
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'prices': prices,
            'source': 'algolab' if self.use_algolab else 'yfinance'
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving price cache: {e}")
    
    def load_price_cache(self) -> dict:
        """Load price cache from file"""
        cache_file = self.cache_dir / "latest_prices_algolab.json" if self.use_algolab else self.cache_dir / "latest_prices.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is not too old (15 minutes for AlgoLab, 1 day for yfinance)
                cache_time = datetime.fromisoformat(cache_data['timestamp'])
                max_age = 900 if self.use_algolab else 86400  # seconds
                if (datetime.now() - cache_time).total_seconds() < max_age:
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
    
    def get_order_book(self, symbol: str) -> dict:
        """Get order book (depth) data for a symbol"""
        if not self.use_algolab or not self.api:
            # Order book not available in yfinance
            return {}
            
        try:
            depth = self.api.get_depth(symbol)
            
            if not depth:
                return {}
            
            # Parse bid/ask data
            order_book = {
                'bids': [],
                'asks': [],
                'timestamp': datetime.now()
            }
            
            # Process bids
            if 'bids' in depth:
                for bid in depth['bids']:
                    order_book['bids'].append({
                        'price': float(bid['price']),
                        'volume': float(bid['volume']),
                        'count': int(bid.get('count', 1))
                    })
            
            # Process asks
            if 'asks' in depth:
                for ask in depth['asks']:
                    order_book['asks'].append({
                        'price': float(ask['price']),
                        'volume': float(ask['volume']),
                        'count': int(ask.get('count', 1))
                    })
            
            return order_book
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {}
    
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
    import os
    import sys
    
    # Check if AlgoLab should be used
    use_algolab = '--algolab' in sys.argv or '-a' in sys.argv
    
    print(f"\nTesting DataFetcher with {'AlgoLab API' if use_algolab else 'Yahoo Finance'}...")
    print("="*60)
    
    fetcher = DataFetcher(use_algolab=use_algolab)
    
    # Get market status
    print("\nMarket Status:")
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
    
    # Test order book (AlgoLab only)
    if use_algolab and fetcher.use_algolab:
        print("\nFetching order book for GARAN...")
        order_book = fetcher.get_order_book('GARAN')
        if order_book:
            print(f"Order Book:")
            print(f"  Bids: {len(order_book.get('bids', []))} levels")
            print(f"  Asks: {len(order_book.get('asks', []))} levels")
            if order_book.get('bids'):
                print(f"  Best Bid: {order_book['bids'][0]['price']:.2f} TL")
            if order_book.get('asks'):
                print(f"  Best Ask: {order_book['asks'][0]['price']:.2f} TL")