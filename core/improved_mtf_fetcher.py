"""
Improved Multi-Timeframe Data Fetcher
Handles Yahoo Finance lookback limitations with chunked fetching
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


class ImprovedMTFDataFetcher:
    """Fetch multi-timeframe data with proper handling of Yahoo Finance limits"""
    
    # Updated timeframe configurations based on test results
    TIMEFRAMES = {
        '1h': {
            'interval': '1h',
            'max_lookback_days': 730,  # Yahoo limit: 2 years
            'chunk_days': 180,  # Fetch in 6-month chunks
            'target_years': 2,
            'description': 'Hourly data for short-term signals'
        },
        '4h': {
            'interval': '1h',  # Fetch 1h and resample
            'max_lookback_days': 730,
            'chunk_days': 180,
            'target_years': 2,
            'description': '4-hour data for intraday trends'
        },
        '1d': {
            'interval': '1d',
            'max_lookback_days': 10000,  # No practical limit
            'chunk_days': 365,
            'target_years': 3,
            'description': 'Daily data for main trend'
        },
        '1wk': {
            'interval': '1wk',
            'max_lookback_days': 10000,
            'chunk_days': 1825,  # 5 years
            'target_years': 5,
            'description': 'Weekly data for long-term perspective'
        }
    }
    
    def __init__(self):
        """Initialize improved MTF data fetcher"""
        self.symbols = SACRED_SYMBOLS
        self.data_dir = DATA_DIR
        self.symbol_suffix = ".IS"
        
        # Create directories for each timeframe
        for tf in self.TIMEFRAMES:
            tf_dir = self.data_dir / 'raw' / tf
            tf_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Improved MTF DataFetcher initialized")
        logger.info(f"Symbols: {len(self.symbols)}")
        logger.info(f"Timeframes: {list(self.TIMEFRAMES.keys())}")
    
    def _get_yf_symbol(self, symbol: str) -> str:
        """Convert BIST symbol to Yahoo Finance format"""
        return f"{symbol}{self.symbol_suffix}"
    
    def _resample_to_4h(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly data to 4-hour"""
        df_4h = df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return df_4h
    
    def fetch_chunked_data(self, ticker: yf.Ticker, interval: str, 
                          start_date: datetime, end_date: datetime,
                          chunk_days: int) -> pd.DataFrame:
        """
        Fetch data in chunks to handle Yahoo Finance limitations
        
        Args:
            ticker: yfinance Ticker object
            interval: Time interval (1h, 1d, etc.)
            start_date: Start date for data
            end_date: End date for data
            chunk_days: Number of days per chunk
            
        Returns:
            Combined DataFrame with all data
        """
        all_data = []
        current_end = end_date
        
        while current_end > start_date:
            # Calculate chunk start
            chunk_start = max(current_end - timedelta(days=chunk_days), start_date)
            
            try:
                logger.debug(f"  Fetching chunk: {chunk_start.date()} to {current_end.date()}")
                
                # Fetch chunk
                df_chunk = ticker.history(
                    start=chunk_start,
                    end=current_end,
                    interval=interval
                )
                
                if not df_chunk.empty:
                    all_data.append(df_chunk)
                    logger.debug(f"    Got {len(df_chunk)} bars")
                
                # Small delay between requests
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"    Error fetching chunk: {e}")
            
            # Move to next chunk
            current_end = chunk_start - timedelta(days=1)
        
        # Combine all chunks
        if all_data:
            df_combined = pd.concat(all_data)
            # Remove duplicates and sort
            df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
            df_combined.sort_index(inplace=True)
            return df_combined
        else:
            return pd.DataFrame()
    
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
                
                # Calculate date range
                end_date = datetime.now()
                target_days = config['target_years'] * 365
                start_date = end_date - timedelta(days=target_days)
                
                # Adjust for max lookback
                if target_days > config['max_lookback_days']:
                    start_date = end_date - timedelta(days=config['max_lookback_days'])
                    logger.info(f"    Adjusted to max lookback: {config['max_lookback_days']} days")
                
                # Fetch data
                if target_days <= config['chunk_days']:
                    # Single request
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=config['interval']
                    )
                else:
                    # Chunked fetching
                    logger.info(f"    Using chunked fetching ({config['chunk_days']} days/chunk)")
                    df = self.fetch_chunked_data(
                        ticker=ticker,
                        interval=config['interval'],
                        start_date=start_date,
                        end_date=end_date,
                        chunk_days=config['chunk_days']
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
                logger.info(f"  ✓ {tf}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
                
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
    
    def fetch_all_symbols_mtf(self, max_symbols: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch multi-timeframe data for all symbols
        
        Args:
            max_symbols: Limit number of symbols (for testing)
            
        Returns:
            Dictionary of symbol -> {timeframe -> DataFrame}
        """
        symbols_to_fetch = self.symbols[:max_symbols] if max_symbols else self.symbols
        logger.info(f"Fetching MTF data for {len(symbols_to_fetch)} symbols...")
        
        all_data = {}
        success_count = 0
        
        for i, symbol in enumerate(symbols_to_fetch, 1):
            logger.info(f"[{i}/{len(symbols_to_fetch)}] Processing {symbol}...")
            
            try:
                # Fetch data
                symbol_data = self.fetch_symbol_mtf(symbol)
                
                if symbol_data:
                    all_data[symbol] = symbol_data
                    self.save_mtf_data(symbol, symbol_data)
                    success_count += 1
                    
                # Delay between symbols
                if i < len(symbols_to_fetch):
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        logger.info(f"✓ Completed: {success_count}/{len(symbols_to_fetch)} symbols")
        return all_data
    
    def update_existing_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Update existing data with latest values
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval
            
        Returns:
            Updated DataFrame
        """
        filepath = self.data_dir / 'raw' / timeframe / f"{symbol}_{timeframe}_raw.csv"
        
        if not filepath.exists():
            logger.info(f"No existing data for {symbol} {timeframe}, fetching all")
            data = self.fetch_symbol_mtf(symbol)
            return data.get(timeframe)
        
        # Load existing data
        existing_df = pd.read_csv(filepath)
        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
        existing_df.set_index('datetime', inplace=True)
        
        last_date = existing_df.index[-1]
        logger.info(f"Updating {symbol} {timeframe} from {last_date}")
        
        # Fetch only new data
        yf_symbol = self._get_yf_symbol(symbol)
        ticker = yf.Ticker(yf_symbol)
        config = self.TIMEFRAMES[timeframe]
        
        try:
            new_df = ticker.history(
                start=last_date + timedelta(days=1),
                end=datetime.now(),
                interval=config['interval']
            )
            
            if not new_df.empty:
                # Process new data
                new_df = new_df[['Open', 'High', 'Low', 'Close', 'Volume']]
                new_df.columns = ['open', 'high', 'low', 'close', 'volume']
                new_df.index.name = 'datetime'
                
                # Handle 4h resampling if needed
                if timeframe == '4h':
                    new_df = self._resample_to_4h(new_df)
                
                new_df['symbol'] = symbol
                
                # Combine with existing
                updated_df = pd.concat([existing_df, new_df])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df.sort_index(inplace=True)
                
                # Save updated data
                self.save_mtf_data(symbol, {timeframe: updated_df})
                logger.info(f"Added {len(new_df)} new bars to {symbol} {timeframe}")
                
                return updated_df
            else:
                logger.info(f"No new data for {symbol} {timeframe}")
                return existing_df
                
        except Exception as e:
            logger.error(f"Error updating {symbol} {timeframe}: {e}")
            return existing_df


def main():
    """Test improved MTF data fetcher"""
    fetcher = ImprovedMTFDataFetcher()
    
    # Test with one symbol
    test_symbol = 'GARAN'
    logger.info(f"\nTesting with {test_symbol}...")
    
    data = fetcher.fetch_symbol_mtf(test_symbol)
    
    if data:
        logger.info(f"\nData summary for {test_symbol}:")
        for tf, df in data.items():
            if df is not None and not df.empty:
                date_range = (df.index[-1] - df.index[0]).days
                logger.info(f"{tf}: {len(df)} bars, {date_range} days")
    
    # Optionally test updating
    logger.info(f"\nTesting update for {test_symbol} 1h...")
    updated = fetcher.update_existing_data(test_symbol, '1h')
    if updated is not None:
        logger.info(f"Updated data has {len(updated)} bars")


if __name__ == "__main__":
    main()