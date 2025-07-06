"""
MACD Custom Indicator - Multiple Time Frame
Based on TradingView implementation
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple


def calculate_macd_custom(
    df: pd.DataFrame,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
    source: str = 'close'
) -> pd.DataFrame:
    """
    Calculate MACD Custom Indicator
    
    Args:
        df: DataFrame with OHLC data
        fast_length: Fast EMA period (default: 12)
        slow_length: Slow EMA period (default: 26)
        signal_length: Signal SMA period (default: 9)
        source: Price source (default: 'close')
        
    Returns:
        DataFrame with MACD values and signals
    """
    # Get source price
    if source == 'close':
        src = df['close']
    elif source == 'open':
        src = df['open']
    elif source == 'high':
        src = df['high']
    elif source == 'low':
        src = df['low']
    elif source == 'hl2':
        src = (df['high'] + df['low']) / 2
    elif source == 'hlc3':
        src = (df['high'] + df['low'] + df['close']) / 3
    elif source == 'ohlc4':
        src = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    else:
        src = df['close']
    
    # Calculate EMAs
    fast_ma = src.ewm(span=fast_length, adjust=False).mean()
    slow_ma = src.ewm(span=slow_length, adjust=False).mean()
    
    # Calculate MACD line
    macd = fast_ma - slow_ma
    
    # Calculate Signal line (using SMA as per custom indicator)
    signal = macd.rolling(window=signal_length).mean()
    
    # Calculate Histogram
    histogram = macd - signal
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['macd'] = macd
    result['signal'] = signal
    result['histogram'] = histogram
    
    # Histogram analysis
    hist_diff = histogram.diff()
    
    # Color coding based on TradingView logic
    result['hist_color'] = 'gray'
    result.loc[(histogram > 0) & (hist_diff > 0), 'hist_color'] = 'aqua'  # Above zero, rising
    result.loc[(histogram > 0) & (hist_diff <= 0), 'hist_color'] = 'blue'  # Above zero, falling
    result.loc[(histogram < 0) & (hist_diff < 0), 'hist_color'] = 'red'  # Below zero, falling
    result.loc[(histogram < 0) & (hist_diff >= 0), 'hist_color'] = 'maroon'  # Below zero, rising
    
    # Cross signals
    result['macd_cross_up'] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
    result['macd_cross_down'] = (macd < signal) & (macd.shift(1) >= signal.shift(1))
    
    # Zero line crosses
    result['macd_above_zero'] = macd > 0
    result['macd_zero_cross_up'] = (macd > 0) & (macd.shift(1) <= 0)
    result['macd_zero_cross_down'] = (macd < 0) & (macd.shift(1) >= 0)
    
    # Momentum states
    result['bullish_momentum'] = (histogram > 0) & (hist_diff > 0)
    result['bearish_momentum'] = (histogram < 0) & (hist_diff < 0)
    result['weakening_bullish'] = (histogram > 0) & (hist_diff < 0)
    result['weakening_bearish'] = (histogram < 0) & (hist_diff > 0)
    
    # Divergence helper columns
    result['macd_peak'] = (macd.shift(1) < macd) & (macd > macd.shift(-1))
    result['macd_trough'] = (macd.shift(1) > macd) & (macd < macd.shift(-1))
    
    return result


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample data to different timeframe for multi-timeframe analysis
    
    Args:
        df: DataFrame with datetime index
        timeframe: Target timeframe ('5T', '15T', '30T', '1H', '4H', '1D')
        
    Returns:
        Resampled DataFrame
    """
    # Define aggregation rules
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Resample
    resampled = df.resample(timeframe).agg(agg_rules)
    resampled = resampled.dropna()
    
    return resampled


def calculate_mtf_macd(
    df: pd.DataFrame,
    timeframes: list = ['15T', '1H', '4H'],
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9
) -> pd.DataFrame:
    """
    Calculate Multi-Timeframe MACD
    
    Args:
        df: DataFrame with OHLC data (should have datetime index)
        timeframes: List of timeframes to analyze
        fast_length: Fast EMA period
        slow_length: Slow EMA period
        signal_length: Signal SMA period
        
    Returns:
        DataFrame with MTF MACD values
    """
    result = pd.DataFrame(index=df.index)
    
    # Current timeframe MACD
    current_macd = calculate_macd_custom(df, fast_length, slow_length, signal_length)
    result['macd'] = current_macd['macd']
    result['signal'] = current_macd['signal']
    result['histogram'] = current_macd['histogram']
    
    # Higher timeframe MACDs
    for tf in timeframes:
        try:
            # Resample to higher timeframe
            df_resampled = resample_to_timeframe(df, tf)
            
            # Calculate MACD on higher timeframe
            htf_macd = calculate_macd_custom(df_resampled, fast_length, slow_length, signal_length)
            
            # Align to original timeframe
            htf_aligned = htf_macd.reindex(df.index, method='ffill')
            
            # Add to results
            result[f'macd_{tf}'] = htf_aligned['macd']
            result[f'signal_{tf}'] = htf_aligned['signal']
            result[f'histogram_{tf}'] = htf_aligned['histogram']
            
        except Exception as e:
            print(f"Error calculating {tf} MACD: {e}")
            continue
    
    # Multi-timeframe alignment signals
    if len(timeframes) > 0:
        # Check if all timeframes are bullish/bearish
        macd_cols = ['histogram'] + [f'histogram_{tf}' for tf in timeframes if f'histogram_{tf}' in result.columns]
        
        result['mtf_all_bullish'] = result[macd_cols].gt(0).all(axis=1)
        result['mtf_all_bearish'] = result[macd_cols].lt(0).all(axis=1)
        
        # Count aligned timeframes
        result['mtf_bullish_count'] = result[macd_cols].gt(0).sum(axis=1)
        result['mtf_bearish_count'] = result[macd_cols].lt(0).sum(axis=1)
    
    return result