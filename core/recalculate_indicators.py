"""
Recalculate all indicators for all symbols
"""
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator


def main():
    """Recalculate all indicators"""
    logger.info("="*80)
    logger.info("RECALCULATING ALL INDICATORS")
    logger.info("="*80)
    
    # Initialize calculator
    calc = IndicatorCalculator(DATA_DIR)
    
    # Calculate for all sacred symbols
    results = calc.calculate_for_symbols(SACRED_SYMBOLS, timeframe='1d', save=True)
    
    # Show summary
    logger.info("\n" + "="*60)
    logger.info("CALCULATION SUMMARY")
    logger.info("="*60)
    
    success_count = 0
    for symbol, indicators in results.items():
        if indicators:
            success_count += 1
            logger.info(f"{symbol}: {len(indicators)} indicators calculated")
        else:
            logger.error(f"{symbol}: FAILED")
    
    logger.info(f"\nTotal: {success_count}/{len(SACRED_SYMBOLS)} symbols processed successfully")
    
    # Show detailed summary
    summary = calc.get_indicators_summary()
    if not summary.empty:
        logger.info("\nDetailed Summary:")
        logger.info(f"Total indicator files: {len(summary)}")
        
        # Group by indicator
        for indicator in summary['indicator'].unique():
            indicator_files = summary[summary['indicator'] == indicator]
            logger.info(f"{indicator}: {len(indicator_files)} symbols")


if __name__ == "__main__":
    main()