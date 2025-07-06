"""
WaveTrend with Crosses [LazyBear]
Based on TradingView implementation
"""
import pandas as pd
import numpy as np


def calculate_wavetrend(
    df: pd.DataFrame,
    n1: int = 10,
    n2: int = 21,
    ob_level1: int = 60,
    ob_level2: int = 53,
    os_level1: int = -60,
    os_level2: int = -53
) -> pd.DataFrame:
    """
    Calculate WaveTrend Oscillator
    
    Args:
        df: DataFrame with OHLC data
        n1: Channel length (default: 10)
        n2: Average length (default: 21)
        ob_level1: Overbought level 1 (default: 60)
        ob_level2: Overbought level 2 (default: 53)
        os_level1: Oversold level 1 (default: -60)
        os_level2: Oversold level 2 (default: -53)
        
    Returns:
        DataFrame with WaveTrend values and signals
    """
    # Calculate Average Price (HLC3)
    ap = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate ESA (Exponentially Smoothed Average)
    esa = ap.ewm(span=n1, adjust=False).mean()
    
    # Calculate D (EMA of absolute difference)
    d = abs(ap - esa).ewm(span=n1, adjust=False).mean()
    
    # Calculate CI (Channel Index)
    ci = (ap - esa) / (0.015 * d)
    # Replace inf values with 0
    ci = ci.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate TCI (Trend Channel Index)
    tci = ci.ewm(span=n2, adjust=False).mean()
    
    # Calculate WT1 and WT2
    wt1 = tci
    wt2 = wt1.rolling(window=4).mean()
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['wt1'] = wt1
    result['wt2'] = wt2
    result['wt_diff'] = wt1 - wt2
    
    # Cross signals
    result['cross_up'] = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    result['cross_down'] = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    
    # Overbought/Oversold zones
    result['overbought'] = wt1 > ob_level2
    result['oversold'] = wt1 < os_level2
    result['extreme_overbought'] = wt1 > ob_level1
    result['extreme_oversold'] = wt1 < os_level1
    
    # Buy/Sell signals based on crosses in OS/OB zones
    result['buy_signal'] = result['cross_up'] & (wt1 < os_level2)
    result['sell_signal'] = result['cross_down'] & (wt1 > ob_level2)
    
    # Strong signals (crosses in extreme zones)
    result['strong_buy'] = result['cross_up'] & (wt1 < os_level1)
    result['strong_sell'] = result['cross_down'] & (wt1 > ob_level1)
    
    # Divergence detection helper columns
    result['wt1_peak'] = (wt1.shift(1) < wt1) & (wt1 > wt1.shift(-1))
    result['wt1_trough'] = (wt1.shift(1) > wt1) & (wt1 < wt1.shift(-1))
    
    # Momentum
    result['momentum'] = wt1.diff()
    result['momentum_increasing'] = result['momentum'] > 0
    result['momentum_decreasing'] = result['momentum'] < 0
    
    # Level references
    result['ob_level1'] = ob_level1
    result['ob_level2'] = ob_level2
    result['os_level1'] = os_level1
    result['os_level2'] = os_level2
    
    return result