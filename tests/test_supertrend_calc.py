"""Test supertrend calculation step by step"""
import pandas as pd
import numpy as np
from indicators.supertrend import calculate_supertrend

# Load raw data
df = pd.read_csv('data/raw/GARAN_1d_raw.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

print("Raw data shape:", df.shape)
print("Raw data columns:", df.columns.tolist())
print("\nFirst 5 rows of raw data:")
print(df.head())

# Calculate supertrend
print("\n\nCalculating supertrend...")
result = calculate_supertrend(df.copy())

print("\nResult shape:", result.shape)
print("Result columns:", result.columns.tolist())

# Check results
print("\nTrend value counts:")
print(result['trend'].value_counts())

print("\nNon-NaN counts:")
print(f"supertrend: {result['supertrend'].notna().sum()}")
print(f"up_band: {result['up_band'].notna().sum()}")
print(f"dn_band: {result['dn_band'].notna().sum()}")

# Debug ATR calculation
print("\n\nDebug ATR calculation:")
# Calculate True Range manually
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift(1))
low_close = abs(df['low'] - df['close'].shift(1))

tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
atr = tr.rolling(window=10).mean()

print(f"True Range non-NaN: {tr.notna().sum()}")
print(f"ATR non-NaN: {atr.notna().sum()}")
print("\nFirst 20 ATR values:")
print(atr.head(20))

# Calculate source price
src = (df['high'] + df['low']) / 2
print(f"\nSource price non-NaN: {src.notna().sum()}")

# Calculate bands
up = src - 3 * atr
dn = src + 3 * atr
print(f"\nUp band non-NaN: {up.notna().sum()}")
print(f"Down band non-NaN: {dn.notna().sum()}")

print("\nFirst 20 up band values:")
print(up.head(20))