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
    """Recalculate all indicators for all timeframes"""
    logger.info("="*80)
    logger.info("RECALCULATING ALL INDICATORS FOR ALL TIMEFRAMES")
    logger.info("="*80)
    
    # Initialize calculator
    calc = IndicatorCalculator(DATA_DIR)
    
    # Define timeframes to process
    timeframes = ['1d', '4h', '1h', '1wk']
    
    # Track overall results
    overall_results = {}
    
    # Calculate for each timeframe
    for timeframe in timeframes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {timeframe} timeframe...")
        logger.info(f"{'='*60}")
        
        # Calculate for all sacred symbols
        results = calc.calculate_for_symbols(SACRED_SYMBOLS, timeframe=timeframe, save=True)
        overall_results[timeframe] = results
        
        # Show summary for this timeframe
        success_count = 0
        for symbol, indicators in results.items():
            if indicators:
                success_count += 1
            else:
                logger.warning(f"{symbol}: No indicators calculated")
        
        logger.info(f"{timeframe}: {success_count}/{len(SACRED_SYMBOLS)} symbols processed successfully")
    
    # Show overall summary
    logger.info("\n" + "="*80)
    logger.info("OVERALL CALCULATION SUMMARY")
    logger.info("="*80)
    
    for timeframe, results in overall_results.items():
        success_count = sum(1 for indicators in results.values() if indicators)
        logger.info(f"{timeframe}: {success_count}/{len(SACRED_SYMBOLS)} symbols")
    
    # Show detailed summary
    summary = calc.get_indicators_summary()
    if not summary.empty:
        logger.info("\nDetailed Summary:")
        logger.info(f"Total indicator files: {len(summary)}")
        
        # Group by timeframe and indicator
        for timeframe in timeframes:
            tf_summary = summary[summary['timeframe'] == timeframe]
            if not tf_summary.empty:
                logger.info(f"\n{timeframe} timeframe:")
                for indicator in calc.core_indicators.keys():
                    indicator_files = tf_summary[tf_summary['indicator'] == indicator]
                    logger.info(f"  {indicator}: {len(indicator_files)} symbols")
    
    logger.info("\n" + "="*80)
    logger.info("ALL INDICATORS RECALCULATED SUCCESSFULLY!")
    logger.info("="*80)


if __name__ == "__main__":
    main()