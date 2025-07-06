"""Debug technical backtest"""
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR

# Check a specific date
test_date = pd.Timestamp('2022-10-18 00:00:00')

# Load GARAN supertrend
st_file = DATA_DIR / 'indicators' / 'GARAN_1d_supertrend.csv'
st_df = pd.read_csv(st_file)
st_df['datetime'] = pd.to_datetime(st_df['datetime'])
st_df.set_index('datetime', inplace=True)

# Remove timezone
if st_df.index.tz is not None:
    st_df.index = st_df.index.tz_localize(None)

print(f"Checking date: {test_date}")
print(f"Date in index: {test_date in st_df.index}")

if test_date in st_df.index:
    row = st_df.loc[test_date]
    print(f"Buy signal: {row.get('buy_signal', False)}")
    print(f"Row data: {row}")

# Check raw data for same date
raw_file = DATA_DIR / 'raw' / 'GARAN_1d_raw.csv'
raw_df = pd.read_csv(raw_file)
raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
raw_df.set_index('datetime', inplace=True)

if raw_df.index.tz is not None:
    raw_df.index = raw_df.index.tz_localize(None)

print(f"\nRaw data - Date in index: {test_date in raw_df.index}")

# List dates around the signal
print("\nDates in supertrend around signal:")
for i in range(65, 75):
    if i < len(st_df):
        print(f"{i}: {st_df.index[i]} - buy_signal: {st_df.iloc[i]['buy_signal']}")