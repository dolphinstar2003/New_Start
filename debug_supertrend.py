"""Debug supertrend calculation"""
import pandas as pd

df = pd.read_csv('data/indicators/GARAN_1d_supertrend.csv')

# Check middle section
print('Middle section (rows 300-320):')
print(df[['datetime', 'supertrend', 'trend', 'up_band', 'dn_band']].iloc[300:320])

print('\n\nChecking where trend changes from initial value:')
# Find where trend is not 1.0 or NaN
non_one_trend = df[(df['trend'] != 1.0) & df['trend'].notna()]
print(f'Rows with trend != 1.0: {len(non_one_trend)}')

if len(non_one_trend) > 0:
    print('First 5 trend changes:')
    print(non_one_trend[['datetime', 'trend', 'supertrend']].head())

# Check the data after ATR period (10 days)  
print('\n\nData after ATR period (rows 10-30):')
print(df[['datetime', 'supertrend', 'trend', 'up_band', 'dn_band']].iloc[10:30])

# Check if bands are calculated
print('\n\nBand statistics:')
print(f'Non-NaN up_band values: {df["up_band"].notna().sum()}')
print(f'Non-NaN dn_band values: {df["dn_band"].notna().sum()}')

# Check a specific row where we should have data
print('\n\nRow 50 details:')
print(df.iloc[50])