#!/usr/bin/env python3
"""
Calculate VixFix indicator for all symbols and timeframes
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from indicators.calculator import IndicatorCalculator
from config.settings import SACRED_SYMBOLS, DATA_DIR
from loguru import logger
import pandas as pd


def calculate_vixfix_for_all():
    """Calculate VixFix for all symbols"""
    calculator = IndicatorCalculator(DATA_DIR)
    
    timeframes = ['1d', '4h', '1h']
    
    for timeframe in timeframes:
        logger.info(f"\nCalculating VixFix for {timeframe} timeframe...")
        
        for symbol in SACRED_SYMBOLS:
            clean_symbol = symbol.replace('.IS', '')
            logger.info(f"Processing {clean_symbol}...")
            
            # Load raw data
            df = calculator.load_raw_data(clean_symbol, timeframe)
            if df is None:
                continue
            
            # Calculate VixFix
            vixfix_df = calculator.calculate_indicator(df, 'vixfix')
            if vixfix_df is not None:
                # Save to CSV
                output_dir = calculator.indicators_dir / timeframe
                output_dir.mkdir(exist_ok=True, parents=True)
                
                output_file = output_dir / f"{clean_symbol}_{timeframe}_vixfix.csv"
                vixfix_df.to_csv(output_file, index=False)
                logger.success(f"Saved {output_file}")


if __name__ == "__main__":
    logger.info("Starting VixFix calculation for all symbols...")
    calculate_vixfix_for_all()
    logger.success("VixFix calculation completed!")