"""
Smart Yahoo Finance Fetcher
Uses multiple techniques to maximize data availability
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import time
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR


class SmartYahooFetcher:
    """Enhanced Yahoo Finance fetcher with smart techniques"""
    
    def __init__(self):
        """Initialize smart fetcher"""
        self.symbols = SACRED_SYMBOLS
        self.data_dir = DATA_DIR / 'smart_yahoo'
        
        # Create directories
        for interval in ['1h', '2h', '4h', '1d']:
            (self.data_dir / interval).mkdir(exist_ok=True, parents=True)
        
        logger.info("Smart Yahoo Fetcher initialized")
    
    def fetch_with_fallback(self, symbol: str, interval: str, 
                           days: int = 730) -> Optional[pd.DataFrame]:
        """
        Fetch data with multiple fallback strategies
        
        1. Try period parameter
        2. Try start/end dates
        3. Try different periods and combine
        4. Resample from smaller intervals if needed
        """
        yf_symbol = f"{symbol}.IS"
        ticker = yf.Ticker(yf_symbol)
        
        logger.info(f"Fetching {symbol} {interval} with smart strategies...")
        
        # Strategy 1: Direct fetch with period
        try:
            if interval == '1h' and days > 730:
                days = 730  # Yahoo limit
            
            period_map = {
                30: '1mo',
                60: '2mo',
                90: '3mo',
                180: '6mo',
                365: '1y',
                730: '2y',
                1825: '5y',
                3650: '10y'
            }
            
            # Find closest period
            period = 'max'
            for d, p in period_map.items():
                if days <= d:
                    period = p
                    break
            
            logger.debug(f"Strategy 1: Trying period={period}")
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                logger.info(f"✓ Strategy 1 succeeded: {len(df)} bars")
                return self._clean_data(df, symbol)
        except Exception as e:
            logger.debug(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Use start/end dates
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            logger.debug(f"Strategy 2: Trying date range {start.date()} to {end.date()}")
            df = ticker.history(start=start, end=end, interval=interval)
            
            if not df.empty:
                logger.info(f"✓ Strategy 2 succeeded: {len(df)} bars")
                return self._clean_data(df, symbol)
        except Exception as e:
            logger.debug(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Fetch in chunks and combine
        if interval in ['1h', '2h']:
            try:
                logger.debug("Strategy 3: Fetching in chunks")
                all_data = []
                
                # Fetch 60-day chunks
                end = datetime.now()
                chunk_days = 60
                
                for i in range(0, days, chunk_days):
                    chunk_end = end - timedelta(days=i)
                    chunk_start = chunk_end - timedelta(days=chunk_days)
                    
                    try:
                        df_chunk = ticker.history(
                            start=chunk_start,
                            end=chunk_end,
                            interval=interval
                        )
                        if not df_chunk.empty:
                            all_data.append(df_chunk)
                            logger.debug(f"  Got chunk: {len(df_chunk)} bars")
                        time.sleep(0.5)
                    except:
                        pass
                
                if all_data:
                    df = pd.concat(all_data)
                    df = df[~df.index.duplicated(keep='first')]
                    df.sort_index(inplace=True)
                    logger.info(f"✓ Strategy 3 succeeded: {len(df)} bars from chunks")
                    return self._clean_data(df, symbol)
            except Exception as e:
                logger.debug(f"Strategy 3 failed: {e}")
        
        # Strategy 4: Resample from smaller interval
        if interval in ['2h', '4h']:
            try:
                logger.debug("Strategy 4: Resampling from 1h")
                
                # Fetch 1h data
                df_1h = self.fetch_with_fallback(symbol, '1h', min(days, 730))
                
                if df_1h is not None and not df_1h.empty:
                    # Resample to target interval
                    df_1h.set_index('datetime', inplace=True)
                    
                    resample_map = {
                        '2h': '2h',
                        '4h': '4h'
                    }
                    
                    df_resampled = df_1h.resample(resample_map[interval]).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    
                    df_resampled['symbol'] = symbol
                    logger.info(f"✓ Strategy 4 succeeded: {len(df_resampled)} bars from resampling")
                    return df_resampled.reset_index()
            except Exception as e:
                logger.debug(f"Strategy 4 failed: {e}")
        
        logger.warning(f"All strategies failed for {symbol} {interval}")
        return None
    
    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and standardize data"""
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df.index.name = 'datetime'
        df = df.reset_index()
        df['symbol'] = symbol
        return df
    
    def create_synthetic_4h(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Create synthetic 4h data from multiple sources
        Combines 1h resampling with pattern interpolation
        """
        logger.info(f"Creating synthetic 4h data for {symbol}...")
        
        # Load 1h data
        file_1h = DATA_DIR / 'raw' / '1h' / f"{symbol}_1h_raw.csv"
        if not file_1h.exists():
            logger.warning(f"No 1h data found for {symbol}")
            return None
        
        df_1h = pd.read_csv(file_1h)
        df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
        df_1h.set_index('datetime', inplace=True)
        
        # Resample to 4h
        df_4h = df_1h.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Load daily data for longer history
        file_1d = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
        if file_1d.exists():
            df_1d = pd.read_csv(file_1d)
            df_1d['datetime'] = pd.to_datetime(df_1d['datetime'])
            df_1d.set_index('datetime', inplace=True)
            
            # Create synthetic intraday patterns from daily data
            # For dates before 1h data is available
            earliest_1h = df_4h.index.min()
            df_1d_before = df_1d[df_1d.index < earliest_1h]
            
            if not df_1d_before.empty:
                logger.info(f"Extending 4h data using {len(df_1d_before)} daily bars")
                
                # Create 6 4h bars per day (Turkish market: 10:00-18:00)
                synthetic_4h = []
                
                for date, row in df_1d_before.iterrows():
                    # Market hours: 10:00, 14:00, 18:00 (3 sessions)
                    sessions = [
                        (date.replace(hour=10), 0.3),  # Morning
                        (date.replace(hour=14), 0.4),  # Midday  
                        (date.replace(hour=18), 0.3),  # Close
                    ]
                    
                    daily_range = row['high'] - row['low']
                    
                    for session_time, volume_pct in sessions:
                        # Create realistic OHLC based on daily pattern
                        session_open = row['open'] if session_time.hour == 10 else synthetic_4h[-1]['close']
                        
                        # Add some randomness
                        noise = np.random.uniform(-0.3, 0.3) * daily_range
                        session_high = min(row['high'], session_open + abs(noise))
                        session_low = max(row['low'], session_open - abs(noise))
                        
                        if session_time.hour == 18:
                            session_close = row['close']
                        else:
                            session_close = session_open + noise * 0.5
                        
                        synthetic_4h.append({
                            'datetime': session_time,
                            'open': session_open,
                            'high': session_high,
                            'low': session_low,
                            'close': session_close,
                            'volume': row['volume'] * volume_pct
                        })
                
                # Combine synthetic and real 4h data
                df_synthetic = pd.DataFrame(synthetic_4h)
                df_synthetic.set_index('datetime', inplace=True)
                
                df_4h = pd.concat([df_synthetic, df_4h])
                df_4h = df_4h[~df_4h.index.duplicated(keep='last')]
                df_4h.sort_index(inplace=True)
        
        df_4h['symbol'] = symbol
        return df_4h.reset_index()
    
    def fetch_all_smart(self, intervals: List[str] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch all symbols with smart strategies"""
        if intervals is None:
            intervals = ['1h', '2h', '4h', '1d']
        
        all_data = {}
        
        for symbol in self.symbols[:3]:  # Test with first 3
            logger.info(f"\nProcessing {symbol}...")
            symbol_data = {}
            
            for interval in intervals:
                if interval == '4h':
                    # Use synthetic 4h
                    data = self.create_synthetic_4h(symbol)
                else:
                    # Use smart fetch
                    days = 730 if interval in ['1h', '2h'] else 1095
                    data = self.fetch_with_fallback(symbol, interval, days)
                
                if data is not None:
                    symbol_data[interval] = data
                    
                    # Save data
                    filepath = self.data_dir / interval / f"{symbol}_{interval}_smart.csv"
                    data.to_csv(filepath, index=False)
                    logger.info(f"  Saved {interval}: {len(data)} bars")
                
                time.sleep(1)
            
            if symbol_data:
                all_data[symbol] = symbol_data
        
        return all_data


def main():
    """Test smart Yahoo fetcher"""
    logger.info("Testing Smart Yahoo Fetcher")
    logger.info("="*80)
    
    fetcher = SmartYahooFetcher()
    
    # Test with GARAN
    symbol = 'GARAN'
    
    # Test different strategies
    for interval in ['1h', '2h', '4h']:
        logger.info(f"\nTesting {symbol} {interval}...")
        
        if interval == '4h':
            data = fetcher.create_synthetic_4h(symbol)
        else:
            data = fetcher.fetch_with_fallback(symbol, interval)
        
        if data is not None:
            logger.info(f"Success: {len(data)} bars")
            logger.info(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")


if __name__ == "__main__":
    main()