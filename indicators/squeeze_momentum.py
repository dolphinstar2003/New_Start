"""
Squeeze Momentum Indicator [LazyBear]
Based on TradingView implementation
"""
import pandas as pd
import numpy as np
from scipy import stats


def calculate_squeeze_momentum(
    df: pd.DataFrame,
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    use_true_range: bool = True
) -> pd.DataFrame:
    """
    Calculate Squeeze Momentum Indicator
    
    Args:
        df: DataFrame with OHLC data
        bb_length: Bollinger Band period (default: 20)
        bb_mult: Bollinger Band multiplier (default: 2.0)
        kc_length: Keltner Channel period (default: 20)
        kc_mult: Keltner Channel multiplier (default: 1.5)
        use_true_range: Use True Range for KC (default: True)
        
    Returns:
        DataFrame with Squeeze values and momentum
    """
    # Bollinger Bands
    bb_basis = df['close'].rolling(window=bb_length).mean()
    bb_dev = df['close'].rolling(window=bb_length).std()
    bb_upper = bb_basis + bb_mult * bb_dev
    bb_lower = bb_basis - bb_mult * bb_dev
    
    # Keltner Channels
    kc_basis = df['close'].rolling(window=kc_length).mean()
    
    if use_true_range:
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        range_ma = true_range.rolling(window=kc_length).mean()
    else:
        range_ma = (df['high'] - df['low']).rolling(window=kc_length).mean()
    
    kc_upper = kc_basis + kc_mult * range_ma
    kc_lower = kc_basis - kc_mult * range_ma
    
    # Squeeze detection
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_off = (bb_lower < kc_lower) & (bb_upper > kc_upper)
    no_squeeze = ~(squeeze_on | squeeze_off)
    
    # Momentum calculation using linear regression
    # Source for momentum
    highest = df['high'].rolling(window=kc_length).max()
    lowest = df['low'].rolling(window=kc_length).min()
    avg_hl = (highest + lowest) / 2
    avg_hls = (avg_hl + kc_basis) / 2
    source = df['close'] - avg_hls
    
    # Linear regression values
    momentum = pd.Series(index=df.index, dtype=float)
    
    for i in range(kc_length - 1, len(df)):
        if i >= kc_length - 1:
            y = source.iloc[i-kc_length+1:i+1].values
            if len(y) == kc_length and not np.any(np.isnan(y)):
                x = np.arange(kc_length)
                slope, intercept, _, _, _ = stats.linregress(x, y)
                # Get the projected value (end of regression line)
                momentum.iloc[i] = intercept + slope * (kc_length - 1)
    
    # Momentum direction and strength
    momentum_diff = momentum.diff()
    
    # Create result DataFrame
    result = pd.DataFrame(index=df.index)
    result['squeeze_on'] = squeeze_on
    result['squeeze_off'] = squeeze_off
    result['no_squeeze'] = no_squeeze
    result['momentum'] = momentum
    result['momentum_hist'] = momentum  # Same as momentum for compatibility
    
    # Momentum colors/states
    result['momentum_color'] = 'gray'
    result.loc[(momentum > 0) & (momentum_diff > 0), 'momentum_color'] = 'lime'  # Strong bullish
    result.loc[(momentum > 0) & (momentum_diff < 0), 'momentum_color'] = 'green'  # Weak bullish
    result.loc[(momentum < 0) & (momentum_diff < 0), 'momentum_color'] = 'red'  # Strong bearish
    result.loc[(momentum < 0) & (momentum_diff > 0), 'momentum_color'] = 'maroon'  # Weak bearish
    
    # Squeeze signals
    result['squeeze_release'] = squeeze_off & squeeze_on.shift(1)  # Squeeze just released
    result['squeeze_start'] = squeeze_on & squeeze_off.shift(1)  # Squeeze just started
    
    # Additional info
    result['bb_upper'] = bb_upper
    result['bb_lower'] = bb_lower
    result['kc_upper'] = kc_upper
    result['kc_lower'] = kc_lower
    
    return result