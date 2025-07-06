"""
Supertrend Indicator
Based on TradingView implementation
"""
import pandas as pd
import numpy as np
from typing import Tuple


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    source: str = 'hl2'
) -> pd.DataFrame:
    """
    Calculate Supertrend indicator
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default: 10)
        multiplier: ATR multiplier (default: 3.0)
        source: Price source - 'hl2', 'hlc3', 'close' (default: 'hl2')
        
    Returns:
        DataFrame with Supertrend values and signals
    """
    # Calculate source price
    if source == 'hl2':
        src = (df['high'] + df['low']) / 2
    elif source == 'hlc3':
        src = (df['high'] + df['low'] + df['close']) / 3
    else:
        src = df['close']
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate ATR using SMA (TradingView default)
    atr = tr.rolling(window=period).mean()
    
    # Calculate basic bands
    up = src - multiplier * atr
    dn = src + multiplier * atr
    
    # Initialize trend
    trend = pd.Series(index=df.index, dtype=float)
    trend.iloc[0] = 1
    
    # Calculate Supertrend
    up_trend = pd.Series(index=df.index, dtype=float)
    dn_trend = pd.Series(index=df.index, dtype=float)
    
    for i in range(1, len(df)):
        # Up trend calculation
        if pd.notna(up.iloc[i]):
            if up.iloc[i] > up_trend.iloc[i-1] or df['close'].iloc[i-1] > up_trend.iloc[i-1]:
                up_trend.iloc[i] = up.iloc[i]
            else:
                up_trend.iloc[i] = up_trend.iloc[i-1]
        
        # Down trend calculation
        if pd.notna(dn.iloc[i]):
            if dn.iloc[i] < dn_trend.iloc[i-1] or df['close'].iloc[i-1] < dn_trend.iloc[i-1]:
                dn_trend.iloc[i] = dn.iloc[i]
            else:
                dn_trend.iloc[i] = dn_trend.iloc[i-1]
        
        # Determine trend
        if pd.notna(dn_trend.iloc[i]) and pd.notna(up_trend.iloc[i]):
            if trend.iloc[i-1] == -1:
                if df['close'].iloc[i] > dn_trend.iloc[i]:
                    trend.iloc[i] = 1
                else:
                    trend.iloc[i] = -1
            else:
                if df['close'].iloc[i] < up_trend.iloc[i]:
                    trend.iloc[i] = -1
                else:
                    trend.iloc[i] = 1
    
    # Calculate Supertrend line
    supertrend = pd.Series(index=df.index, dtype=float)
    supertrend[trend == 1] = up_trend[trend == 1]
    supertrend[trend == -1] = dn_trend[trend == -1]
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['supertrend'] = supertrend
    result['trend'] = trend
    result['up_band'] = up_trend
    result['dn_band'] = dn_trend
    
    # Add buy/sell signals
    result['buy_signal'] = (trend == 1) & (trend.shift(1) == -1)
    result['sell_signal'] = (trend == -1) & (trend.shift(1) == 1)
    
    return result