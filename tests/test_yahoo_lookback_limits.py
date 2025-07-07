"""
Test Yahoo Finance lookback limits for different timeframes
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

# Test symbol
symbol = "GARAN.IS"

# Test different intervals and periods
test_configs = [
    # Minute intervals
    {"interval": "1m", "periods": ["1d", "7d", "30d", "60d"]},
    {"interval": "2m", "periods": ["1d", "7d", "30d", "60d"]},
    {"interval": "5m", "periods": ["1d", "7d", "30d", "60d"]},
    {"interval": "15m", "periods": ["1d", "7d", "30d", "60d"]},
    {"interval": "30m", "periods": ["1d", "7d", "30d", "60d"]},
    {"interval": "60m", "periods": ["1d", "7d", "30d", "60d", "730d"]},
    {"interval": "90m", "periods": ["1d", "7d", "30d", "60d"]},
    
    # Hour intervals
    {"interval": "1h", "periods": ["1d", "7d", "30d", "60d", "730d"]},
    
    # Day/week/month intervals
    {"interval": "1d", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]},
    {"interval": "5d", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]},
    {"interval": "1wk", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]},
    {"interval": "1mo", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]},
    {"interval": "3mo", "periods": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]},
]

logger.info(f"Testing Yahoo Finance lookback limits for {symbol}")
logger.info("=" * 80)

ticker = yf.Ticker(symbol)
results = []

for config in test_configs:
    interval = config["interval"]
    logger.info(f"\nTesting interval: {interval}")
    
    for period in config["periods"]:
        try:
            df = ticker.history(period=period, interval=interval)
            
            if not df.empty:
                start_date = df.index[0]
                end_date = df.index[-1]
                num_bars = len(df)
                days_span = (end_date - start_date).days
                
                result = {
                    "interval": interval,
                    "period": period,
                    "status": "SUCCESS",
                    "bars": num_bars,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "days": days_span
                }
                
                logger.info(f"  {period}: ✓ {num_bars} bars, {days_span} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
            else:
                result = {
                    "interval": interval,
                    "period": period,
                    "status": "EMPTY",
                    "bars": 0,
                    "start_date": None,
                    "end_date": None,
                    "days": 0
                }
                logger.warning(f"  {period}: ✗ No data returned")
                
        except Exception as e:
            result = {
                "interval": interval,
                "period": period,
                "status": "ERROR",
                "bars": 0,
                "start_date": None,
                "end_date": None,
                "days": 0,
                "error": str(e)
            }
            logger.error(f"  {period}: ✗ Error: {e}")
            
        results.append(result)

# Create summary DataFrame
df_results = pd.DataFrame(results)

# Show summary by interval
logger.info("\n" + "=" * 80)
logger.info("SUMMARY BY INTERVAL")
logger.info("=" * 80)

for interval in df_results['interval'].unique():
    interval_data = df_results[df_results['interval'] == interval]
    successful = interval_data[interval_data['status'] == 'SUCCESS']
    
    if not successful.empty:
        max_days = successful['days'].max()
        max_bars = successful['bars'].max()
        best_period = successful[successful['days'] == max_days].iloc[0]['period']
        
        logger.info(f"\n{interval}:")
        logger.info(f"  Max lookback: {max_days} days with period '{best_period}'")
        logger.info(f"  Max bars: {max_bars}")
        logger.info(f"  Working periods: {', '.join(successful['period'].tolist())}")

# Test alternative: using start/end dates for problematic intervals
logger.info("\n" + "=" * 80)
logger.info("TESTING START/END DATE APPROACH FOR 1h and 4h")
logger.info("=" * 80)

# Test fetching by date ranges
date_ranges = [
    {"days": 7, "name": "1 week"},
    {"days": 30, "name": "1 month"},
    {"days": 60, "name": "2 months"},
    {"days": 90, "name": "3 months"},
    {"days": 180, "name": "6 months"},
    {"days": 365, "name": "1 year"},
    {"days": 730, "name": "2 years"},
]

for interval in ["1h", "4h"]:
    logger.info(f"\nTesting {interval} with date ranges:")
    
    for date_range in date_ranges:
        try:
            end = datetime.now()
            start = end - timedelta(days=date_range["days"])
            
            # For 4h, we need to fetch 1h and resample
            fetch_interval = "1h" if interval == "4h" else interval
            
            df = ticker.history(start=start, end=end, interval=fetch_interval)
            
            if not df.empty:
                # Resample if needed
                if interval == "4h" and fetch_interval == "1h":
                    df = df.resample('4H').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).dropna()
                
                actual_start = df.index[0]
                actual_end = df.index[-1]
                num_bars = len(df)
                actual_days = (actual_end - actual_start).days
                
                logger.info(f"  {date_range['name']}: ✓ {num_bars} bars, {actual_days} days actual")
            else:
                logger.warning(f"  {date_range['name']}: ✗ No data")
                
        except Exception as e:
            logger.error(f"  {date_range['name']}: ✗ Error: {e}")

# Save results
df_results.to_csv("yahoo_finance_lookback_limits.csv", index=False)
logger.info(f"\n✅ Results saved to yahoo_finance_lookback_limits.csv")