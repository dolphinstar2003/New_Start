"""
CM Williams Vix Fix - Market Bottom Finder
Converted from Pine Script to Python
Measures market fear/volatility
"""
import pandas as pd
import numpy as np
from typing import Dict


def calculate_vixfix(
    df: pd.DataFrame,
    lookback_period: int = 22,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    percentile_lookback: int = 50,
    highest_percentile: float = 0.85,
    lowest_percentile: float = 1.01
) -> pd.DataFrame:
    """
    Calculate CM Williams Vix Fix indicator
    
    Args:
        df: DataFrame with OHLCV data
        lookback_period: Period for calculating highest close
        bb_length: Bollinger Band length
        bb_mult: BB standard deviation multiplier
        percentile_lookback: Lookback for percentile calculation
        highest_percentile: High percentile threshold (0.85 = 85%)
        lowest_percentile: Low percentile threshold (1.01 = 99%)
    
    Returns:
        DataFrame with VixFix values and signals
    """
    result = pd.DataFrame(index=df.index)
    
    # Williams Vix Fix calculation
    # WVF = ((Highest(Close, pd) - Low) / Highest(Close, pd)) * 100
    highest_close = df['close'].rolling(window=lookback_period).max()
    wvf = ((highest_close - df['low']) / highest_close) * 100
    result['vixfix'] = wvf
    
    # Bollinger Bands
    wvf_sma = wvf.rolling(window=bb_length).mean()
    wvf_std = wvf.rolling(window=bb_length).std()
    result['bb_upper'] = wvf_sma + (bb_mult * wvf_std)
    result['bb_middle'] = wvf_sma
    result['bb_lower'] = wvf_sma - (bb_mult * wvf_std)
    
    # Percentile ranges
    result['range_high'] = wvf.rolling(window=percentile_lookback).max() * highest_percentile
    result['range_low'] = wvf.rolling(window=percentile_lookback).min() * lowest_percentile
    
    # Signals
    result['high_volatility'] = (wvf >= result['bb_upper']) | (wvf >= result['range_high'])
    result['market_bottom'] = result['high_volatility'] & (wvf > wvf.shift(1))
    
    # Volatility levels
    result['vix_level'] = pd.cut(
        wvf,
        bins=[0, 10, 20, 30, 100],
        labels=['low', 'normal', 'high', 'extreme']
    )
    # Fill NaN values with 'low' as default
    result['vix_level'] = result['vix_level'].fillna('low')
    
    # Position sizing factor based on VIX
    # When VIX is high, reduce position size
    max_vix = wvf.rolling(window=252).max().fillna(50)
    min_vix = wvf.rolling(window=252).min().fillna(5)
    vix_normalized = (wvf - min_vix) / (max_vix - min_vix)
    vix_normalized = vix_normalized.fillna(0.5)  # Default to middle if NaN
    result['position_factor'] = 1 - (vix_normalized * 0.5)  # Reduce up to 50% in high volatility
    result['position_factor'] = result['position_factor'].fillna(1.0)  # Default to 1.0 if still NaN
    
    # Trading zones
    result['buy_zone'] = result['high_volatility']
    result['sell_zone'] = wvf < result['bb_middle']
    
    # Trend filter using VIX
    result['vix_trend'] = np.where(
        wvf > wvf.rolling(window=10).mean(),
        -1,  # Increasing volatility (bearish)
        1    # Decreasing volatility (bullish)
    )
    
    return result


def get_vix_risk_adjustment(vix_value: float, vix_mean: float = 20) -> Dict[str, float]:
    """
    Get risk management adjustments based on VIX level
    
    Args:
        vix_value: Current VIX value
        vix_mean: Long-term VIX average
    
    Returns:
        Dict with risk adjustments
    """
    vix_ratio = vix_value / vix_mean
    
    adjustments = {
        'position_size_multiplier': 1.0,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 1.0,
        'trailing_stop_distance': 0.03  # 3% default
    }
    
    if vix_ratio < 0.5:  # Very low volatility
        adjustments['position_size_multiplier'] = 1.2
        adjustments['stop_loss_multiplier'] = 0.8  # Tighter stops
        adjustments['trailing_stop_distance'] = 0.02
        
    elif vix_ratio < 0.8:  # Low volatility
        adjustments['position_size_multiplier'] = 1.1
        adjustments['stop_loss_multiplier'] = 0.9
        adjustments['trailing_stop_distance'] = 0.025
        
    elif vix_ratio > 2.0:  # Extreme volatility
        adjustments['position_size_multiplier'] = 0.3
        adjustments['stop_loss_multiplier'] = 1.5  # Wider stops
        adjustments['take_profit_multiplier'] = 1.5  # Higher targets
        adjustments['trailing_stop_distance'] = 0.05
        
    elif vix_ratio > 1.5:  # High volatility
        adjustments['position_size_multiplier'] = 0.5
        adjustments['stop_loss_multiplier'] = 1.3
        adjustments['take_profit_multiplier'] = 1.3
        adjustments['trailing_stop_distance'] = 0.04
        
    return adjustments


def calculate_kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    vix_factor: float = 1.0
) -> float:
    """
    Calculate optimal position size using Kelly Criterion
    
    Args:
        win_rate: Probability of winning (0-1)
        avg_win: Average winning amount
        avg_loss: Average losing amount (positive number)
        vix_factor: VIX adjustment factor (0-1)
    
    Returns:
        Optimal position size as fraction of capital
    """
    if avg_loss == 0:
        return 0
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win probability, q = loss probability, b = win/loss ratio
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly = (p * b - q) / b
    
    # Apply VIX adjustment and safety factor
    kelly_adjusted = kelly * vix_factor * 0.25  # Use 25% of Kelly for safety
    
    # Cap at 10% per position
    return min(max(kelly_adjusted, 0), 0.10)