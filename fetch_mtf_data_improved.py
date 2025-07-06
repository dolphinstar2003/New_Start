"""
Fetch Multi-Timeframe Data with Improved Handling
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from loguru import logger
from core.improved_mtf_fetcher import ImprovedMTFDataFetcher
from config.settings import SACRED_SYMBOLS
import time

logger.info("="*80)
logger.info("FETCHING MULTI-TIMEFRAME DATA WITH IMPROVED HANDLING")
logger.info("="*80)

# Initialize fetcher
fetcher = ImprovedMTFDataFetcher()

# For testing, let's start with first 5 symbols
test_symbols = SACRED_SYMBOLS[:5]
logger.info(f"Fetching data for {len(test_symbols)} symbols: {', '.join(test_symbols)}")

# Fetch data for each symbol
for i, symbol in enumerate(test_symbols, 1):
    logger.info(f"\n[{i}/{len(test_symbols)}] Processing {symbol}...")
    
    try:
        # Fetch all timeframes for this symbol
        data = fetcher.fetch_symbol_mtf(symbol)
        
        if data:
            logger.info(f"Successfully fetched data for {symbol}:")
            for tf, df in data.items():
                if df is not None and not df.empty:
                    logger.info(f"  {tf}: {len(df)} bars")
        else:
            logger.warning(f"No data fetched for {symbol}")
            
        # Delay between symbols
        if i < len(test_symbols):
            logger.info("Waiting 3 seconds before next symbol...")
            time.sleep(3)
            
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")

logger.info("\n" + "="*80)
logger.info("DATA FETCHING COMPLETED")
logger.info("="*80)

# Show summary
logger.info("\nChecking saved data...")
for tf in ['1h', '4h', '1d', '1wk']:
    tf_dir = fetcher.data_dir / 'raw' / tf
    if tf_dir.exists():
        files = list(tf_dir.glob("*.csv"))
        logger.info(f"{tf}: {len(files)} files saved")
        
logger.info("\n✅ Multi-timeframe data fetching completed!")
logger.info("Next steps:")
logger.info("1. Run indicator calculations for all timeframes")
logger.info("2. Train ML models with MTF features")
logger.info("3. Run backtest with MTF signals")