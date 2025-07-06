"""
Test Alternative Data Sources for BIST100
TradingView and Investing.com options
"""
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import json

logger.info("="*80)
logger.info("ALTERNATIVE DATA SOURCES FOR BIST100")
logger.info("="*80)

# Option 1: TradingView (using unofficial library)
logger.info("\n1. TRADINGVIEW OPTIONS:")
logger.info("-" * 40)
print("""
pip install tradingview-ta
or
pip install tvDatafeed

Example usage:
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()
# No login required for free data
data = tv.get_hist(symbol='GARAN', exchange='BIST', interval=Interval.in_1_hour, n_bars=5000)

Pros:
- Much longer history for intraday data
- Real 4h interval available
- More reliable than Yahoo
- Free for basic use

Cons:
- Unofficial API (may break)
- Rate limits apply
""")

# Option 2: Investing.com (using investpy)
logger.info("\n2. INVESTING.COM OPTIONS:")
logger.info("-" * 40)
print("""
pip install investpy

Example usage:
import investpy

# Get BIST stocks
stocks = investpy.get_stocks(country='turkey')
# Get historical data
data = investpy.get_stock_historical_data(
    stock='GARAN',
    country='turkey',
    from_date='01/01/2020',
    to_date='01/01/2024',
    interval='Daily'
)

Pros:
- Official-ish Python library
- Good historical coverage
- Includes fundamentals

Cons:
- Only daily data reliably
- May require headers/cookies
- Can be blocked
""")

# Option 3: Direct web scraping with selenium
logger.info("\n3. TRADINGVIEW WEB SCRAPING:")
logger.info("-" * 40)
print("""
Using selenium to export from TradingView:

from selenium import webdriver
from selenium.webdriver.common.by import By
import time

driver = webdriver.Chrome()
driver.get('https://www.tradingview.com/chart/')
# Login, load BIST:GARAN
# Use chart's export function
# Can get CSV with full history

Pros:
- Access to ALL TradingView data
- Any timeframe available
- Most reliable for long history

Cons:
- Requires browser automation
- Slower than API
- May need TradingView account
""")

# Option 4: Alternative approach using cryptocurrencies
logger.info("\n4. CRYPTO CORRELATION APPROACH:")
logger.info("-" * 40)
print("""
Use BTCTRY, ETHTRY from Binance (unlimited history):

import ccxt
exchange = ccxt.binance()

# Get BTCTRY with unlimited history
ohlcv = exchange.fetch_ohlcv('BTC/TRY', '1h', limit=1000)

Then correlate with BIST100 index for signals.

Pros:
- Unlimited historical data
- 24/7 trading (more data)
- High correlation with risk assets

Cons:
- Not direct stock data
- Requires correlation analysis
""")

# Recommendation
logger.info("\n" + "="*80)
logger.info("RECOMMENDATION: TradingView via tvDatafeed")
logger.info("="*80)

print("""
Best approach for BIST100 MTF data:

1. Install tvDatafeed:
   pip install tvDatafeed

2. Fetch data with proper intervals:
   - 1h: up to 10,000 bars (~1 year)
   - 4h: up to 10,000 bars (~4 years)  
   - 1d: unlimited history
   - 1w: unlimited history

3. Benefits over Yahoo Finance:
   - Native 4h support (no resampling needed)
   - Longer intraday history
   - More stable API
   - Better data quality

Would you like me to implement a TradingView data fetcher?
""")