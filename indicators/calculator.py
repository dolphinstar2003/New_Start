"""
Indicator Calculator
Calculate all Core 5 indicators for given data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from supertrend import calculate_supertrend
from adx_di import calculate_adx_di
from squeeze_momentum import calculate_squeeze_momentum
from wavetrend import calculate_wavetrend
from macd_custom import calculate_macd_custom


class IndicatorCalculator:
    """Calculate and save technical indicators"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize calculator
        
        Args:
            data_dir: Path to data directory
        """
        self.raw_data_dir = data_dir / 'raw'
        self.indicators_dir = data_dir / 'indicators'
        self.indicators_dir.mkdir(exist_ok=True, parents=True)
        
        # Core 5 indicators
        self.core_indicators = {
            'supertrend': calculate_supertrend,
            'adx_di': calculate_adx_di,
            'squeeze_momentum': calculate_squeeze_momentum,
            'wavetrend': calculate_wavetrend,
            'macd_custom': calculate_macd_custom
        }
        
        logger.info(f"IndicatorCalculator initialized with {len(self.core_indicators)} indicators")
    
    def load_raw_data(self, symbol: str, timeframe: str = '1d') -> Optional[pd.DataFrame]:
        """
        Load raw OHLCV data
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        filename = f"{symbol}_{timeframe}_raw.csv"
        filepath = self.raw_data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Raw data not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in {filepath}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def calculate_indicator(
        self,
        df: pd.DataFrame,
        indicator_name: str,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Calculate single indicator
        
        Args:
            df: OHLCV DataFrame
            indicator_name: Name of indicator
            **kwargs: Additional parameters for indicator
            
        Returns:
            DataFrame with indicator values
        """
        if indicator_name not in self.core_indicators:
            logger.error(f"Unknown indicator: {indicator_name}")
            return None
        
        try:
            indicator_func = self.core_indicators[indicator_name]
            result = indicator_func(df, **kwargs)
            
            # Add datetime column for CSV export
            result.reset_index(inplace=True)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            return None
    
    def calculate_all_indicators(
        self,
        symbol: str,
        timeframe: str = '1d',
        save: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate all Core 5 indicators for a symbol
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            save: Whether to save results to CSV
            
        Returns:
            Dictionary of indicator results
        """
        logger.info(f"Calculating indicators for {symbol} {timeframe}")
        
        # Load raw data
        df = self.load_raw_data(symbol, timeframe)
        if df is None:
            return {}
        
        results = {}
        
        # Calculate each indicator
        for indicator_name in self.core_indicators:
            logger.info(f"  Calculating {indicator_name}...")
            
            indicator_df = self.calculate_indicator(df, indicator_name)
            if indicator_df is not None:
                results[indicator_name] = indicator_df
                
                # Save to CSV
                if save:
                    filename = f"{symbol}_{timeframe}_{indicator_name}.csv"
                    filepath = self.indicators_dir / filename
                    indicator_df.to_csv(filepath, index=False)
                    logger.info(f"    Saved {len(indicator_df)} rows to {filepath}")
            else:
                logger.warning(f"  Failed to calculate {indicator_name}")
        
        logger.info(f"Completed {symbol}: {len(results)}/{len(self.core_indicators)} indicators")
        return results
    
    def calculate_for_symbols(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        save: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate indicators for multiple symbols
        
        Args:
            symbols: List of stock symbols
            timeframe: Data timeframe
            save: Whether to save results to CSV
            
        Returns:
            Dictionary of {symbol: {indicator: DataFrame}}
        """
        logger.info(f"Calculating indicators for {len(symbols)} symbols")
        
        all_results = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
            
            symbol_results = self.calculate_all_indicators(symbol, timeframe, save)
            if symbol_results:
                all_results[symbol] = symbol_results
        
        logger.info(f"Completed indicator calculations for {len(all_results)} symbols")
        return all_results
    
    def get_indicators_summary(self) -> pd.DataFrame:
        """Get summary of calculated indicators"""
        summary_data = []
        
        for csv_file in self.indicators_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                
                # Parse filename: SYMBOL_TIMEFRAME_INDICATOR.csv
                parts = csv_file.stem.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    timeframe = parts[1]
                    indicator = '_'.join(parts[2:])
                    
                    summary_data.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'indicator': indicator,
                        'rows': len(df),
                        'file': csv_file.name
                    })
                    
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.sort_values(['symbol', 'timeframe', 'indicator'], inplace=True)
            return summary_df
        else:
            return pd.DataFrame()


def main():
    """Test indicator calculator"""
    from pathlib import Path
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    # Initialize calculator
    calc = IndicatorCalculator(data_dir)
    
    # Test with one symbol
    print("Testing with THYAO...")
    results = calc.calculate_all_indicators('THYAO', '1d')
    
    if results:
        print(f"✅ Calculated {len(results)} indicators")
        for indicator_name, df in results.items():
            print(f"  {indicator_name}: {len(df)} rows")
    
    # Show summary
    print("\n" + "="*60)
    print("INDICATORS SUMMARY")
    print("="*60)
    summary = calc.get_indicators_summary()
    if not summary.empty:
        print(summary.to_string())


if __name__ == "__main__":
    main()