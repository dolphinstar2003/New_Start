"""
Data Fetcher Module
Fetch market data from various sources
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, RAW_DATA_DIR, TIMEFRAMES


class DataFetcher:
    """Fetch market data for BIST stocks"""
    
    def __init__(self):
        """Initialize data fetcher"""
        self.symbols = SACRED_SYMBOLS
        self.data_dir = RAW_DATA_DIR
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # BIST symbol suffix
        self.symbol_suffix = ".IS"
        
        logger.info(f"DataFetcher initialized with {len(self.symbols)} symbols")
    
    def _get_yf_symbol(self, symbol: str) -> str:
        """Convert BIST symbol to Yahoo Finance format"""
        return f"{symbol}{self.symbol_suffix}"
    
    def _get_history_period(self, timeframe: str) -> tuple:
        """Get start date based on timeframe requirements"""
        end_date = datetime.now()
        
        tf_config = TIMEFRAMES.get(timeframe, {})
        
        if 'history_years' in tf_config:
            years = tf_config['history_years']
            start_date = end_date - timedelta(days=365 * years)
        elif 'history_months' in tf_config:
            months = tf_config['history_months']
            start_date = end_date - timedelta(days=30 * months)
        else:
            # Default to 1 year
            start_date = end_date - timedelta(days=365)
        
        return start_date, end_date
    
    def fetch_symbol_data(self, symbol: str, timeframe: str = '1d', 
                         save: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch data for a single symbol
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval (1d, 1h, 15m)
            save: Whether to save to CSV
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            yf_symbol = self._get_yf_symbol(symbol)
            start_date, end_date = self._get_history_period(timeframe)
            
            logger.info(f"Fetching {symbol} {timeframe} from {start_date.date()} to {end_date.date()}")
            
            # Download data with error handling
            ticker = yf.Ticker(yf_symbol)
            try:
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=timeframe
                )
            except Exception as e:
                logger.error(f"Error downloading {yf_symbol}: {e}")
                # Try alternative approach
                try:
                    df = yf.download(yf_symbol, start=start_date, end=end_date, interval=timeframe, progress=False)
                    if not df.empty:
                        df = df.reset_index()
                except Exception as e2:
                    logger.error(f"Alternative download also failed for {yf_symbol}: {e2}")
                    return None
            
            if df.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return None
            
            # Clean and prepare data
            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            
            # Rename datetime column
            if 'date' in df.columns:
                df.rename(columns={'date': 'datetime'}, inplace=True)
            elif 'datetime' not in df.columns and df.index.name == 'Date':
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'datetime'}, inplace=True)
            
            # Ensure datetime column
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Select required columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in df.columns]
            df = df[available_cols]
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Sort by datetime
            df.sort_values('datetime', inplace=True)
            
            # Save to CSV
            if save:
                filename = f"{symbol}_{timeframe}_raw.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} rows to {filepath}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def fetch_all_symbols(self, timeframe: str = '1d', 
                         delay: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all sacred symbols
        
        Args:
            timeframe: Time interval
            delay: Delay between requests (seconds)
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        results = {}
        total = len(self.symbols)
        
        logger.info(f"Starting to fetch {total} symbols for {timeframe}")
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"[{i}/{total}] Fetching {symbol}...")
            
            df = self.fetch_symbol_data(symbol, timeframe)
            if df is not None:
                results[symbol] = df
            
            # Rate limiting
            if i < total:
                time.sleep(delay)
        
        logger.info(f"Completed. Successfully fetched {len(results)}/{total} symbols")
        return results
    
    def update_symbol_data(self, symbol: str, timeframe: str = '1d') -> Optional[pd.DataFrame]:
        """
        Update existing data with latest values
        
        Args:
            symbol: Stock symbol
            timeframe: Time interval
            
        Returns:
            Updated DataFrame
        """
        filename = f"{symbol}_{timeframe}_raw.csv"
        filepath = self.data_dir / filename
        
        # Load existing data if available
        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
            last_date = existing_df['datetime'].max()
            
            # Fetch only new data
            logger.info(f"Updating {symbol} from {last_date}")
            
            yf_symbol = self._get_yf_symbol(symbol)
            ticker = yf.Ticker(yf_symbol)
            
            new_df = ticker.history(
                start=last_date + timedelta(days=1),
                end=datetime.now(),
                interval=timeframe
            )
            
            if not new_df.empty:
                # Process new data
                new_df = new_df.reset_index()
                new_df.columns = [col.lower() for col in new_df.columns]
                new_df.rename(columns={'date': 'datetime'}, inplace=True)
                new_df['datetime'] = pd.to_datetime(new_df['datetime'])
                new_df['symbol'] = symbol
                
                # Combine with existing
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
                combined_df.sort_values('datetime', inplace=True)
                
                # Save updated data
                combined_df.to_csv(filepath, index=False)
                logger.info(f"Updated {symbol} with {len(new_df)} new rows")
                
                return combined_df
            else:
                logger.info(f"No new data for {symbol}")
                return existing_df
        else:
            # No existing data, fetch all
            return self.fetch_symbol_data(symbol, timeframe)
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary of available data"""
        summary_data = []
        
        for csv_file in self.data_dir.glob("*_raw.csv"):
            try:
                df = pd.read_csv(csv_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                
                parts = csv_file.stem.split('_')
                symbol = parts[0]
                timeframe = parts[1]
                
                summary_data.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'rows': len(df),
                    'start_date': df['datetime'].min(),
                    'end_date': df['datetime'].max(),
                    'days': (df['datetime'].max() - df['datetime'].min()).days
                })
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.sort_values(['symbol', 'timeframe'], inplace=True)
            return summary_df
        else:
            return pd.DataFrame()


def main():
    """Test data fetcher"""
    fetcher = DataFetcher()
    
    # Test with one symbol
    print("Testing with GARAN...")
    df = fetcher.fetch_symbol_data('GARAN', '1d')
    
    if df is not None:
        print(f"✅ Fetched {len(df)} rows")
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Show summary
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        summary = fetcher.get_data_summary()
        if not summary.empty:
            print(summary)


if __name__ == "__main__":
    main()