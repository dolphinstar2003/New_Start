"""
Multi-Timeframe Data Fetcher
Fetch data for multiple timeframes
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR


class MultiTimeframeDataFetcher:
    """Fetch multi-timeframe data for BIST stocks"""
    
    # Timeframe configurations
    TIMEFRAMES = {
        '1h': {
            'interval': '1h',
            'period': '60d',  # 60 days
            'description': 'Hourly data for short-term signals'
        },
        '4h': {
            'interval': '1h',  # YF doesn't have 4h, we'll resample
            'period': '180d',  # 180 days
            'description': '4-hour data for intraday trends'
        },
        '1d': {
            'interval': '1d',
            'period': '3y',  # 3 years
            'description': 'Daily data for main trend'
        },
        '1wk': {
            'interval': '1wk',
            'period': '5y',  # 5 years
            'description': 'Weekly data for long-term perspective'
        }
    }
    
    def __init__(self):
        """Initialize MTF data fetcher"""
        self.symbols = SACRED_SYMBOLS
        self.data_dir = DATA_DIR
        self.symbol_suffix = ".IS"
        
        # Create directories for each timeframe
        for tf in self.TIMEFRAMES:
            tf_dir = self.data_dir / 'raw' / tf
            tf_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"MTF DataFetcher initialized with {len(self.symbols)} symbols")
        logger.info(f"Timeframes: {list(self.TIMEFRAMES.keys())}")
    
    def _get_yf_symbol(self, symbol: str) -> str:
        """Convert BIST symbol to Yahoo Finance format"""
        return f"{symbol}{self.symbol_suffix}"
    
    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to 4-hour"""
        df_4h = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_4h
    
    def fetch_symbol_mtf(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all timeframes for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of {timeframe: DataFrame}
        """
        yf_symbol = self._get_yf_symbol(symbol)
        logger.info(f"Fetching multi-timeframe data for {symbol} ({yf_symbol})...")
        
        results = {}
        ticker = yf.Ticker(yf_symbol)
        
        for tf, config in self.TIMEFRAMES.items():
            try:
                logger.info(f"  Fetching {tf} data...")
                
                # Fetch data
                df = ticker.history(
                    period=config['period'],
                    interval=config['interval']
                )
                
                if df.empty:
                    logger.warning(f"  No data for {symbol} {tf}")
                    continue
                
                # Clean data
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df.index.name = 'datetime'
                
                # Handle 4h timeframe
                if tf == '4h':
                    df = self._resample_to_4h(df)
                
                # Add symbol column
                df['symbol'] = symbol
                
                results[tf] = df
                logger.info(f"  ✓ {tf}: {len(df)} bars")
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"  Error fetching {tf} data: {e}")
        
        return results
    
    def save_mtf_data(self, symbol: str, data: Dict[str, pd.DataFrame]) -> None:
        """Save multi-timeframe data to CSV files"""
        for tf, df in data.items():
            if df is not None and not df.empty:
                filename = f"{symbol}_{tf}_raw.csv"
                filepath = self.data_dir / 'raw' / tf / filename
                
                # Reset index to save datetime as column
                df_save = df.reset_index()
                df_save.to_csv(filepath, index=False)
                
                logger.debug(f"  Saved {symbol} {tf} to {filepath}")
    
    def fetch_all_symbols_mtf(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch multi-timeframe data for all symbols"""
        logger.info(f"Fetching MTF data for {len(self.symbols)} symbols...")
        
        all_data = {}
        success_count = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{len(self.symbols)}] Processing {symbol}...")
            
            try:
                # Fetch data
                symbol_data = self.fetch_symbol_mtf(symbol)
                
                if symbol_data:
                    all_data[symbol] = symbol_data
                    self.save_mtf_data(symbol, symbol_data)
                    success_count += 1
                    
                # Delay between symbols
                if i < len(self.symbols):
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info(f"✓ Completed: {success_count}/{len(self.symbols)} symbols")
        return all_data
    
    def load_mtf_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Load saved multi-timeframe data for a symbol"""
        data = {}
        
        for tf in self.TIMEFRAMES:
            filepath = self.data_dir / 'raw' / tf / f"{symbol}_{tf}_raw.csv"
            
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                data[tf] = df
        
        return data
    
    def get_mtf_summary(self) -> pd.DataFrame:
        """Get summary of available MTF data"""
        summary_data = []
        
        for tf in self.TIMEFRAMES:
            tf_dir = self.data_dir / 'raw' / tf
            if tf_dir.exists():
                files = list(tf_dir.glob("*_raw.csv"))
                
                for file in files:
                    symbol = file.stem.split('_')[0]
                    
                    # Get file info
                    df = pd.read_csv(file, nrows=1)
                    total_rows = sum(1 for _ in open(file)) - 1
                    
                    summary_data.append({
                        'symbol': symbol,
                        'timeframe': tf,
                        'rows': total_rows,
                        'file_size_kb': file.stat().st_size / 1024
                    })
        
        if summary_data:
            return pd.DataFrame(summary_data).sort_values(['symbol', 'timeframe'])
        else:
            return pd.DataFrame()


def main():
    """Test multi-timeframe data fetching"""
    fetcher = MultiTimeframeDataFetcher()
    
    # Test with one symbol
    test_symbol = 'GARAN'
    logger.info(f"\nTesting with {test_symbol}...")
    
    data = fetcher.fetch_symbol_mtf(test_symbol)
    
    if data:
        logger.info(f"\nData summary for {test_symbol}:")
        for tf, df in data.items():
            if df is not None and not df.empty:
                logger.info(f"{tf}: {len(df)} bars, from {df.index[0]} to {df.index[-1]}")
    
    # Optionally fetch all symbols (uncomment to run)
    # fetcher.fetch_all_symbols_mtf()


if __name__ == "__main__":
    main()