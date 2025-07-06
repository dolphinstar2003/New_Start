"""
TradingView Data Fetcher for BIST100
Using tvDatafeed for better historical data
"""
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR


class TradingViewDataFetcher:
    """Fetch data from TradingView with extended history"""
    
    # TradingView intervals mapping
    INTERVAL_MAPPING = {
        '1m': Interval.in_1_minute,
        '5m': Interval.in_5_minute,
        '15m': Interval.in_15_minute,
        '30m': Interval.in_30_minute,
        '1h': Interval.in_1_hour,
        '2h': Interval.in_2_hour,
        '4h': Interval.in_4_hour,
        '1d': Interval.in_daily,
        '1wk': Interval.in_weekly,
        '1mo': Interval.in_monthly,
    }
    
    # Maximum bars per request (TradingView limit)
    MAX_BARS = {
        '1m': 10000,    # ~7 days
        '5m': 10000,    # ~35 days
        '15m': 10000,   # ~104 days
        '30m': 10000,   # ~208 days
        '1h': 10000,    # ~416 days (1.1 years)
        '2h': 10000,    # ~833 days (2.3 years)
        '4h': 10000,    # ~1666 days (4.5 years)
        '1d': 50000,    # ~136 years
        '1wk': 10000,   # ~192 years
        '1mo': 5000,    # ~416 years
    }
    
    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """Initialize TradingView fetcher"""
        self.symbols = SACRED_SYMBOLS
        self.data_dir = DATA_DIR
        
        # Create TV data directories
        self.tv_dir = self.data_dir / 'tradingview'
        for interval in ['1h', '4h', '1d', '1wk']:
            (self.tv_dir / interval).mkdir(exist_ok=True, parents=True)
        
        # Initialize TvDatafeed
        if username and password:
            self.tv = TvDatafeed(username, password)
            logger.info("TradingView fetcher initialized with login")
        else:
            self.tv = TvDatafeed()
            logger.info("TradingView fetcher initialized (no login)")
    
    def fetch_symbol_data(self, symbol: str, interval: str = '1h', 
                         n_bars: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol from TradingView
        
        Args:
            symbol: Stock symbol (without .IS suffix)
            interval: Time interval (1h, 4h, 1d, etc.)
            n_bars: Number of bars to fetch (default: maximum)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get TradingView interval
            tv_interval = self.INTERVAL_MAPPING.get(interval)
            if not tv_interval:
                logger.error(f"Invalid interval: {interval}")
                return None
            
            # Set number of bars
            if n_bars is None:
                n_bars = self.MAX_BARS.get(interval, 5000)
            
            logger.info(f"Fetching {symbol} {interval} from TradingView (up to {n_bars} bars)...")
            
            # Fetch data - BIST stocks are on 'BIST' exchange
            data = self.tv.get_hist(
                symbol=symbol,
                exchange='BIST',
                interval=tv_interval,
                n_bars=n_bars,
                extended_session=False  # Regular session only
            )
            
            if data is None or data.empty:
                logger.warning(f"No data returned for {symbol} {interval}")
                return None
            
            # Reset index to get datetime as column
            data = data.reset_index()
            
            # Rename columns to match our format
            data.columns = [col.lower() for col in data.columns]
            if 'datetime' in data.columns:
                data['datetime'] = pd.to_datetime(data['datetime'])
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Sort by datetime
            data.sort_values('datetime', inplace=True)
            
            logger.info(f"✓ Fetched {len(data)} bars from {data['datetime'].iloc[0]} to {data['datetime'].iloc[-1]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} {interval}: {e}")
            return None
    
    def save_data(self, data: pd.DataFrame, symbol: str, interval: str) -> None:
        """Save data to CSV file"""
        if data is None or data.empty:
            return
        
        filename = f"{symbol}_{interval}_tv.csv"
        filepath = self.tv_dir / interval / filename
        
        data.to_csv(filepath, index=False)
        logger.debug(f"Saved {len(data)} rows to {filepath}")
    
    def fetch_all_symbols(self, intervals: List[str] = None, 
                         delay: float = 2.0) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch data for all symbols and intervals
        
        Args:
            intervals: List of intervals to fetch (default: ['1h', '4h', '1d', '1wk'])
            delay: Delay between requests in seconds
            
        Returns:
            Nested dict: {symbol: {interval: DataFrame}}
        """
        if intervals is None:
            intervals = ['1h', '4h', '1d', '1wk']
        
        all_data = {}
        total_requests = len(self.symbols) * len(intervals)
        request_count = 0
        
        logger.info(f"Fetching {len(self.symbols)} symbols × {len(intervals)} intervals = {total_requests} requests")
        
        for symbol in self.symbols:
            symbol_data = {}
            
            for interval in intervals:
                request_count += 1
                logger.info(f"[{request_count}/{total_requests}] Fetching {symbol} {interval}...")
                
                try:
                    # Fetch data
                    data = self.fetch_symbol_data(symbol, interval)
                    
                    if data is not None:
                        symbol_data[interval] = data
                        self.save_data(data, symbol, interval)
                    
                    # Rate limiting
                    if request_count < total_requests:
                        time.sleep(delay)
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol} {interval}: {e}")
            
            if symbol_data:
                all_data[symbol] = symbol_data
        
        logger.info(f"✓ Completed: fetched data for {len(all_data)} symbols")
        return all_data
    
    def compare_with_yahoo(self, symbol: str = 'GARAN') -> pd.DataFrame:
        """Compare TradingView data with Yahoo Finance data"""
        comparison = []
        
        for interval in ['1h', '1d']:
            # Load Yahoo data
            yahoo_file = DATA_DIR / 'raw' / interval / f"{symbol}_{interval}_raw.csv"
            if yahoo_file.exists():
                yahoo_df = pd.read_csv(yahoo_file)
                yahoo_df['datetime'] = pd.to_datetime(yahoo_df['datetime'])
                
                yahoo_start = yahoo_df['datetime'].min()
                yahoo_end = yahoo_df['datetime'].max()
                yahoo_bars = len(yahoo_df)
            else:
                yahoo_start = yahoo_end = None
                yahoo_bars = 0
            
            # Load TradingView data
            tv_file = self.tv_dir / interval / f"{symbol}_{interval}_tv.csv"
            if tv_file.exists():
                tv_df = pd.read_csv(tv_file)
                tv_df['datetime'] = pd.to_datetime(tv_df['datetime'])
                
                tv_start = tv_df['datetime'].min()
                tv_end = tv_df['datetime'].max()
                tv_bars = len(tv_df)
            else:
                tv_start = tv_end = None
                tv_bars = 0
            
            comparison.append({
                'interval': interval,
                'yahoo_bars': yahoo_bars,
                'yahoo_start': yahoo_start,
                'yahoo_end': yahoo_end,
                'tv_bars': tv_bars,
                'tv_start': tv_start,
                'tv_end': tv_end,
                'tv_advantage': tv_bars - yahoo_bars
            })
        
        return pd.DataFrame(comparison)


def main():
    """Test TradingView data fetcher"""
    logger.info("Testing TradingView Data Fetcher")
    logger.info("="*80)
    
    # Initialize fetcher
    fetcher = TradingViewDataFetcher()
    
    # Test with one symbol
    test_symbol = 'GARAN'
    
    # Test different intervals
    for interval in ['1h', '4h', '1d']:
        logger.info(f"\nTesting {test_symbol} {interval}...")
        data = fetcher.fetch_symbol_data(test_symbol, interval)
        
        if data is not None:
            date_range = (data['datetime'].iloc[-1] - data['datetime'].iloc[0]).days
            logger.info(f"Got {len(data)} bars covering {date_range} days")
    
    # Compare with Yahoo Finance
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: TradingView vs Yahoo Finance")
    logger.info("="*80)
    
    comparison = fetcher.compare_with_yahoo(test_symbol)
    print(comparison)
    
    # Fetch all symbols for just 1h and 4h (to save time)
    logger.info("\n" + "="*80)
    logger.info("FETCHING ALL SYMBOLS (1h and 4h only)")
    logger.info("="*80)
    
    # Uncomment to fetch all:
    # fetcher.fetch_all_symbols(intervals=['1h', '4h'])


if __name__ == "__main__":
    main()