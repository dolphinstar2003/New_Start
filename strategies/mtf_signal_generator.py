"""
Multi-Timeframe Signal Generator
Combine signals from multiple timeframes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DATA_DIR


class MTFSignalGenerator:
    """Generate trading signals using multiple timeframes"""
    
    # Timeframe weights for signal aggregation
    TIMEFRAME_WEIGHTS = {
        '1h': 0.20,   # Short-term momentum
        '4h': 0.30,   # Intraday trend
        '1d': 0.50,   # Primary trend
    }
    
    # Signal mapping
    SIGNAL_VALUES = {
        'STRONG_BUY': 2,
        'BUY': 1,
        'NEUTRAL': 0,
        'SELL': -1,
        'STRONG_SELL': -2
    }
    
    def __init__(self, indicators_dir: Path = None):
        """Initialize MTF signal generator"""
        self.indicators_dir = indicators_dir or (DATA_DIR / 'indicators')
        self.timeframes = list(self.TIMEFRAME_WEIGHTS.keys())
        
        logger.info(f"MTF Signal Generator initialized")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Weights: {self.TIMEFRAME_WEIGHTS}")
    
    def load_tf_indicators(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load indicators for a specific timeframe"""
        # For now, we'll use supertrend as the main indicator
        filepath = self.indicators_dir / timeframe / f"{symbol}_{timeframe}_supertrend.csv"
        
        if not filepath.exists():
            # Try without timeframe subdirectory (current structure)
            filepath = self.indicators_dir / f"{symbol}_{timeframe}_supertrend.csv"
        
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
        else:
            logger.warning(f"Indicator file not found: {filepath}")
            return None
    
    def get_signal_from_indicators(self, row: pd.Series) -> str:
        """Convert indicator values to signal"""
        # Based on supertrend
        if row.get('buy_signal', False):
            return 'STRONG_BUY'
        elif row.get('sell_signal', False):
            return 'STRONG_SELL'
        elif row.get('trend', 0) == 1:
            return 'BUY'
        elif row.get('trend', 0) == -1:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def align_timeframes(self, tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align data from different timeframes to daily"""
        # For simplicity, we'll resample everything to daily
        aligned_data = {}
        
        for tf, df in tf_data.items():
            if df is None or df.empty:
                continue
            
            # Get signal for each row
            df['signal'] = df.apply(self.get_signal_from_indicators, axis=1)
            df['signal_value'] = df['signal'].map(self.SIGNAL_VALUES)
            
            # Resample to daily (take last value of each day)
            daily = df[['signal', 'signal_value']].resample('D').last()
            daily = daily.rename(columns={
                'signal': f'signal_{tf}',
                'signal_value': f'signal_value_{tf}'
            })
            
            aligned_data[tf] = daily
        
        # Combine all timeframes
        if aligned_data:
            combined = pd.concat(aligned_data.values(), axis=1)
            return combined.dropna()  # Only keep days with all timeframes
        else:
            return pd.DataFrame()
    
    def calculate_mtf_signal(self, row: pd.Series) -> Tuple[str, float, Dict]:
        """Calculate combined signal from multiple timeframes"""
        # Calculate weighted signal
        weighted_sum = 0
        weights_sum = 0
        tf_signals = {}
        
        for tf, weight in self.TIMEFRAME_WEIGHTS.items():
            signal_col = f'signal_value_{tf}'
            if signal_col in row and pd.notna(row[signal_col]):
                weighted_sum += row[signal_col] * weight
                weights_sum += weight
                tf_signals[tf] = row[f'signal_{tf}']
        
        if weights_sum > 0:
            weighted_signal = weighted_sum / weights_sum
        else:
            return 'NEUTRAL', 0.0, tf_signals
        
        # Determine final signal
        if weighted_signal >= 1.5:
            signal = 'STRONG_BUY'
        elif weighted_signal >= 0.5:
            signal = 'BUY'
        elif weighted_signal <= -1.5:
            signal = 'STRONG_SELL'
        elif weighted_signal <= -0.5:
            signal = 'SELL'
        else:
            signal = 'NEUTRAL'
        
        # Calculate confidence (0-1)
        confidence = min(abs(weighted_signal) / 2.0, 1.0)
        
        return signal, confidence, tf_signals
    
    def generate_mtf_signals(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Generate multi-timeframe signals for a symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with MTF signals
        """
        logger.info(f"Generating MTF signals for {symbol}...")
        
        # Load indicators for each timeframe
        tf_data = {}
        for tf in self.timeframes:
            df = self.load_tf_indicators(symbol, tf)
            if df is not None:
                tf_data[tf] = df
                logger.info(f"  Loaded {tf}: {len(df)} rows")
        
        if not tf_data:
            logger.warning(f"No indicator data found for {symbol}")
            return pd.DataFrame()
        
        # Align timeframes
        aligned = self.align_timeframes(tf_data)
        
        if aligned.empty:
            logger.warning(f"No aligned data for {symbol}")
            return pd.DataFrame()
        
        # Calculate MTF signals
        results = []
        for idx, row in aligned.iterrows():
            signal, confidence, tf_signals = self.calculate_mtf_signal(row)
            
            results.append({
                'datetime': idx,
                'mtf_signal': signal,
                'mtf_confidence': confidence,
                **{f'{tf}_signal': sig for tf, sig in tf_signals.items()}
            })
        
        result_df = pd.DataFrame(results)
        result_df.set_index('datetime', inplace=True)
        
        # Filter by date range if specified
        if start_date:
            result_df = result_df[result_df.index >= pd.to_datetime(start_date)]
        if end_date:
            result_df = result_df[result_df.index <= pd.to_datetime(end_date)]
        
        logger.info(f"Generated {len(result_df)} MTF signals for {symbol}")
        
        # Log signal distribution
        if not result_df.empty:
            signal_counts = result_df['mtf_signal'].value_counts()
            logger.info("Signal distribution:")
            for signal, count in signal_counts.items():
                logger.info(f"  {signal}: {count} ({count/len(result_df)*100:.1f}%)")
        
        return result_df
    
    def generate_all_symbols_mtf(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Generate MTF signals for multiple symbols"""
        all_signals = {}
        
        for symbol in symbols:
            signals = self.generate_mtf_signals(symbol)
            if not signals.empty:
                all_signals[symbol] = signals
        
        logger.info(f"Generated MTF signals for {len(all_signals)}/{len(symbols)} symbols")
        return all_signals


def test_mtf_signals():
    """Test MTF signal generation"""
    generator = MTFSignalGenerator()
    
    # Test with GARAN
    symbol = 'GARAN'
    
    # Note: This will only work if we have indicators for multiple timeframes
    # For now, it will use only 1d data
    signals = generator.generate_mtf_signals(
        symbol,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
    
    if not signals.empty:
        print(f"\nMTF Signals for {symbol}:")
        print(signals.head(10))
        
        # Show some statistics
        print(f"\nTotal signals: {len(signals)}")
        print("\nSignal distribution:")
        print(signals['mtf_signal'].value_counts())
        
        # Show strong signals
        strong_signals = signals[signals['mtf_signal'].isin(['STRONG_BUY', 'STRONG_SELL'])]
        if not strong_signals.empty:
            print(f"\nStrong signals: {len(strong_signals)}")
            print(strong_signals.head())


if __name__ == "__main__":
    test_mtf_signals()