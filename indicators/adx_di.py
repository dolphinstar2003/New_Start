"""
ADX and DI (Directional Indicators)
Based on TradingView implementation
"""
import pandas as pd
import numpy as np


def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing method"""
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()


def calculate_adx_di(
    df: pd.DataFrame,
    length: int = 14,
    threshold: int = 20
) -> pd.DataFrame:
    """
    Calculate ADX and +DI/-DI indicators
    
    Args:
        df: DataFrame with OHLC data
        length: Period for ADX calculation (default: 14)
        threshold: ADX threshold level (default: 20)
        
    Returns:
        DataFrame with ADX, +DI, -DI values
    """
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    # +DM and -DM
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    # Apply conditions
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    
    # Smooth using Wilder's method
    tr_smooth = wilder_smooth(tr, length)
    plus_dm_smooth = wilder_smooth(plus_dm, length)
    minus_dm_smooth = wilder_smooth(minus_dm, length)
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX
    di_sum = plus_di + minus_di
    di_diff = abs(plus_di - minus_di)
    dx = 100 * di_diff / di_sum
    
    # Replace inf and nan with 0
    dx = dx.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate ADX using Wilder's smoothing
    adx = wilder_smooth(dx, length)
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['adx'] = adx
    result['plus_di'] = plus_di
    result['minus_di'] = minus_di
    result['dx'] = dx
    
    # Add signals
    result['trend_strength'] = 'weak'
    result.loc[adx > threshold, 'trend_strength'] = 'strong'
    result.loc[adx > threshold * 2, 'trend_strength'] = 'very_strong'
    
    # Trend direction
    result['trend_direction'] = 'neutral'
    result.loc[plus_di > minus_di, 'trend_direction'] = 'bullish'
    result.loc[minus_di > plus_di, 'trend_direction'] = 'bearish'
    
    # DI crossovers
    result['di_bullish_cross'] = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    result['di_bearish_cross'] = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
    
    return result