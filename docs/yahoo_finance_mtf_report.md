# Yahoo Finance Multi-Timeframe Data Report

## Executive Summary

After investigating Yahoo Finance API limitations and implementing an improved multi-timeframe data fetcher, here are the key findings and recommendations.

## Yahoo Finance Lookback Limitations

### Confirmed Limits by Interval

1. **Minute Intervals**
   - 1m: 7-8 days maximum
   - 2m: 60 days maximum
   - 5m: 60-90 days maximum
   - 15m, 30m: 60-90 days maximum

2. **Hourly Intervals**
   - 1h: **730 days (2 years) maximum**
   - 4h: Not directly available, must resample from 1h data

3. **Daily/Weekly/Monthly Intervals**
   - 1d: 25+ years available
   - 1wk: 25+ years available
   - 1mo: 25+ years available

## Implementation Solution

### Improved MTF Data Fetcher Features

1. **Chunked Fetching**
   - Fetches data in 180-day chunks for hourly data
   - Handles Yahoo Finance rate limits with delays
   - Combines chunks into complete dataset

2. **4-Hour Resampling**
   ```python
   df_4h = df.resample('4h').agg({
       'open': 'first',
       'high': 'max',
       'low': 'min',
       'close': 'last',
       'volume': 'sum'
   })
   ```

3. **Automatic Retry and Error Handling**
   - Gracefully handles API errors
   - Continues with next chunk on failure

## Data Availability Summary

### Successfully Fetched Data (First 5 Symbols)

| Symbol | 1h Bars | 4h Bars | 1d Bars | 1wk Bars |
|--------|---------|---------|---------|----------|
| GARAN  | 4,196   | 1,473   | 744     | 261      |
| AKBNK  | 4,196   | 1,473   | 744     | 261      |
| ISCTR  | 4,196   | 1,473   | 744     | 261      |
| YKBNK  | 4,196   | 1,473   | 744     | 261      |
| SAHOL  | 4,196   | 1,473   | 744     | 261      |

### Date Ranges
- **1h/4h**: 2023-07-14 to 2025-07-04 (~2 years)
- **1d**: 2022-07-07 to 2025-07-04 (~3 years)
- **1wk**: 2020-07-06 to 2025-06-30 (~5 years)

## Recommendations

### 1. Data Strategy
- Use 1d as primary timeframe (most reliable, longest history)
- Use 1h for short-term signals (limited to 2 years)
- Resample 1h to get 4h data
- Use 1wk for long-term trend confirmation

### 2. Indicator Calculation
- Calculate indicators for each timeframe separately
- Store in timeframe-specific directories
- Use forward-fill for alignment when combining

### 3. ML Training Approach
- Train on 1.5 years of data (within 1h limits)
- Use all available timeframes as features
- Weight recent data more heavily

### 4. Backtest Considerations
- Limited to 2 years for full MTF strategy
- Can extend to 3 years using only daily/weekly
- Consider walk-forward optimization

## Next Steps

1. **Complete MTF Data Fetch**
   ```bash
   python fetch_mtf_data_improved.py
   ```

2. **Calculate Indicators for All Timeframes**
   ```bash
   python calculate_mtf_indicators.py
   ```

3. **Train ML Models with MTF Features**
   ```bash
   python train_mtf_models.py
   ```

4. **Run MTF Backtest**
   ```bash
   python run_mtf_backtest.py
   ```

## Alternative Data Sources

If longer intraday history is needed:
1. **Binance API**: Crypto pairs with TRY (indirect correlation)
2. **MetaTrader**: Some brokers offer BIST data
3. **TradingView**: Pine script data export (manual)
4. **Local data providers**: İş Yatırım, Garanti BBVA APIs

## Conclusion

The 2-year limitation on hourly data is a hard constraint from Yahoo Finance. Our implementation successfully works within these limits by:
- Fetching maximum available data for each timeframe
- Using chunked requests to handle large datasets
- Resampling to create 4h data
- Properly aligning multi-timeframe features for ML training

This provides a solid foundation for multi-timeframe trading strategies on BIST100.