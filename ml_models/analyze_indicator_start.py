"""
Analyze indicator start dates to find optimal training period
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR


def analyze_indicator_start_dates():
    """Analyze when indicators start having valid data"""
    
    indicators_dir = DATA_DIR / 'indicators'
    indicator_types = ['supertrend', 'adx_di', 'squeeze_momentum', 'wavetrend', 'macd_custom']
    
    # Store first valid dates for each indicator and symbol
    first_valid_dates = {}
    
    for symbol in SACRED_SYMBOLS:
        logger.info(f"\nAnalyzing {symbol}...")
        symbol_dates = {}
        
        for indicator in indicator_types:
            file_path = indicators_dir / f"{symbol}_1d_{indicator}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                # Find first row with non-NaN values in main columns
                main_cols = [col for col in df.columns if col not in ['datetime']]
                
                # Check each row for valid data
                first_valid_idx = None
                for idx in df.index:
                    row = df.loc[idx]
                    # Check if at least 50% of columns have valid data
                    valid_count = row[main_cols].notna().sum()
                    if valid_count >= len(main_cols) * 0.5:
                        first_valid_idx = idx
                        break
                
                if first_valid_idx:
                    symbol_dates[indicator] = first_valid_idx
                    logger.info(f"  {indicator}: {first_valid_idx.strftime('%Y-%m-%d')}")
            else:
                logger.warning(f"  {indicator}: File not found")
        
        if symbol_dates:
            # Get the latest start date for this symbol
            latest_date = max(symbol_dates.values())
            first_valid_dates[symbol] = {
                'latest_start': latest_date,
                'indicator_starts': symbol_dates
            }
    
    # Find overall latest start date
    overall_latest = max(data['latest_start'] for data in first_valid_dates.values())
    
    logger.info("\n" + "="*60)
    logger.info("INDICATOR START DATE ANALYSIS")
    logger.info("="*60)
    
    # Summary by indicator type
    logger.info("\nBy Indicator Type:")
    for indicator in indicator_types:
        dates = []
        for symbol_data in first_valid_dates.values():
            if indicator in symbol_data['indicator_starts']:
                dates.append(symbol_data['indicator_starts'][indicator])
        
        if dates:
            earliest = min(dates)
            latest = max(dates)
            logger.info(f"{indicator:20} - Earliest: {earliest.strftime('%Y-%m-%d')}, Latest: {latest.strftime('%Y-%m-%d')}")
    
    # Summary by symbol
    logger.info("\nLatest Start Date by Symbol:")
    for symbol, data in sorted(first_valid_dates.items()):
        logger.info(f"{symbol:10} - {data['latest_start'].strftime('%Y-%m-%d')}")
    
    logger.info(f"\nOVERALL LATEST START DATE: {overall_latest.strftime('%Y-%m-%d')}")
    logger.info(f"Recommended training start: {overall_latest.strftime('%Y-%m-%d')}")
    
    # Check how much data we have after this date
    price_file = DATA_DIR / 'raw' / f"{SACRED_SYMBOLS[0]}_1d_raw.csv"
    if price_file.exists():
        df = pd.read_csv(price_file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        available_data = df[df.index >= overall_latest.tz_localize(None)]
        logger.info(f"Available data points after latest start: {len(available_data)} days")
    
    return overall_latest


if __name__ == "__main__":
    analyze_indicator_start_dates()