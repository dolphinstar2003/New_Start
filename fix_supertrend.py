"""Fix supertrend calculation"""
import pandas as pd
import numpy as np

def calculate_supertrend_fixed(df, period=10, multiplier=3.0):
    """Calculate Supertrend with proper implementation"""
    
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    # Source price (HL2)
    hl2 = (df['high'] + df['low']) / 2
    
    # Basic bands
    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr
    
    # Create empty series
    supertrend = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=float)
    
    # Initialize
    trend.iloc[0] = 1
    
    # Calculate iteratively
    for i in range(1, len(df)):
        # Skip if ATR not available
        if pd.isna(atr.iloc[i]):
            trend.iloc[i] = trend.iloc[i-1]
            continue
            
        # Current close price
        curr_close = df['close'].iloc[i]
        
        # Previous trend
        prev_trend = trend.iloc[i-1]
        
        # Calculate current bands
        curr_up = up.iloc[i]
        curr_dn = dn.iloc[i]
        
        # Trend logic
        if prev_trend == 1:
            if curr_close <= curr_up:
                trend.iloc[i] = -1
                supertrend.iloc[i] = curr_dn
            else:
                trend.iloc[i] = 1
                supertrend.iloc[i] = curr_up
        else:  # prev_trend == -1
            if curr_close >= curr_dn:
                trend.iloc[i] = 1
                supertrend.iloc[i] = curr_up
            else:
                trend.iloc[i] = -1
                supertrend.iloc[i] = curr_dn
    
    # Create result
    result = pd.DataFrame(index=df.index)
    result['supertrend'] = supertrend
    result['trend'] = trend
    result['up_band'] = up
    result['dn_band'] = dn
    result['buy_signal'] = (trend == 1) & (trend.shift(1) == -1)
    result['sell_signal'] = (trend == -1) & (trend.shift(1) == 1)
    
    return result


# Test with GARAN
df = pd.read_csv('data/raw/GARAN_1d_raw.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Try with smaller multiplier
result = calculate_supertrend_fixed(df, period=10, multiplier=2.0)

print("Result summary:")
print(f"Trend value counts:")
print(result['trend'].value_counts())
print(f"\nBuy signals: {result['buy_signal'].sum()}")
print(f"Sell signals: {result['sell_signal'].sum()}")

# Save for testing
result.reset_index().to_csv('test_supertrend_fixed.csv', index=False)