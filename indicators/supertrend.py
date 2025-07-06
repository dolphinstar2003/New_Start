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
    source: str = 'hl2',
    change_atr: bool = True
) -> pd.DataFrame:
    """
    Calculate Supertrend indicator based on TradingView implementation
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period (default: 10)
        multiplier: ATR multiplier (default: 3.0)
        source: Price source - 'hl2', 'hlc3', 'close' (default: 'hl2')
        change_atr: Use ta.atr() vs sma(tr) (default: True)
        
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
    
    # Calculate ATR
    if change_atr:
        # Use Wilder's smoothing (like ta.atr)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
    else:
        # Use simple moving average
        atr = tr.rolling(window=period).mean()
    
    # Calculate basic bands
    basic_up = src - multiplier * atr
    basic_dn = src + multiplier * atr
    
    # Initialize arrays
    up = pd.Series(index=df.index, dtype=float)
    dn = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    
    # Set initial values
    up.iloc[0] = basic_up.iloc[0] if pd.notna(basic_up.iloc[0]) else np.nan
    dn.iloc[0] = basic_dn.iloc[0] if pd.notna(basic_dn.iloc[0]) else np.nan
    trend.iloc[0] = 1
    
    # Calculate Supertrend using TradingView logic
    for i in range(1, len(df)):
        # Skip if ATR not available
        if pd.isna(basic_up.iloc[i]) or pd.isna(basic_dn.iloc[i]):
            up.iloc[i] = up.iloc[i-1]
            dn.iloc[i] = dn.iloc[i-1]
            trend.iloc[i] = trend.iloc[i-1]
            continue
        
        # Update up band
        # up := close[1] > up1 ? max(up,up1) : up
        if df['close'].iloc[i-1] > up.iloc[i-1]:
            up.iloc[i] = max(basic_up.iloc[i], up.iloc[i-1])
        else:
            up.iloc[i] = basic_up.iloc[i]
        
        # Update down band
        # dn := close[1] < dn1 ? min(dn, dn1) : dn
        if df['close'].iloc[i-1] < dn.iloc[i-1]:
            dn.iloc[i] = min(basic_dn.iloc[i], dn.iloc[i-1])
        else:
            dn.iloc[i] = basic_dn.iloc[i]
        
        # Determine trend
        # trend := trend == -1 and close > dn1 ? 1 : trend == 1 and close < up1 ? -1 : trend
        if trend.iloc[i-1] == -1 and df['close'].iloc[i] > dn.iloc[i-1]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and df['close'].iloc[i] < up.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
    
    # Calculate Supertrend line
    supertrend = pd.Series(index=df.index, dtype=float)
    supertrend[trend == 1] = up[trend == 1]
    supertrend[trend == -1] = dn[trend == -1]
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['supertrend'] = supertrend
    result['trend'] = trend
    result['up_band'] = up
    result['dn_band'] = dn
    
    # Add buy/sell signals
    result['buy_signal'] = (trend == 1) & (trend.shift(1) == -1)
    result['sell_signal'] = (trend == -1) & (trend.shift(1) == 1)
    
    return result