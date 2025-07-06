"""
Multi-Timeframe Setup for Trading System
"""
from pathlib import Path
from loguru import logger

# Supported timeframes
TIMEFRAMES = {
    '1h': {
        'period': '1h',
        'days': 60,  # 60 days of hourly data
        'weight': 0.2,  # Weight in signal generation
        'description': 'Hourly for short-term momentum'
    },
    '4h': {
        'period': '4h', 
        'days': 180,  # 180 days of 4-hour data
        'weight': 0.3,
        'description': '4-hour for intraday trends'
    },
    '1d': {
        'period': '1d',
        'days': 1095,  # 3 years of daily data
        'weight': 0.5,
        'description': 'Daily for main trend'
    }
}

# Multi-timeframe strategy rules
MTF_RULES = {
    'signal_alignment': {
        'strong_buy': 'All timeframes must show BUY',
        'buy': 'At least 2 timeframes show BUY',
        'hold': 'Mixed signals',
        'sell': 'At least 2 timeframes show SELL',
        'strong_sell': 'All timeframes must show SELL'
    },
    'confirmation': {
        '1h': 'Entry timing',
        '4h': 'Trend confirmation', 
        '1d': 'Major trend direction'
    }
}

def setup_multi_timeframe():
    """Setup multi-timeframe data structure"""
    logger.info("="*60)
    logger.info("MULTI-TIMEFRAME SETUP")
    logger.info("="*60)
    
    for tf, config in TIMEFRAMES.items():
        logger.info(f"\n{tf}: {config['description']}")
        logger.info(f"  Period: {config['period']}")
        logger.info(f"  History: {config['days']} days")
        logger.info(f"  Weight: {config['weight']*100}%")
    
    logger.info("\nSignal Generation Rules:")
    for signal, rule in MTF_RULES['signal_alignment'].items():
        logger.info(f"  {signal.upper()}: {rule}")
    
    logger.info("\nTimeframe Roles:")
    for tf, role in MTF_RULES['confirmation'].items():
        logger.info(f"  {tf}: {role}")
    
    # Suggested implementation steps
    logger.info("\n" + "="*60)
    logger.info("IMPLEMENTATION STEPS")
    logger.info("="*60)
    logger.info("1. Modify data_fetcher.py to support multiple timeframes")
    logger.info("2. Update indicator calculator for each timeframe")
    logger.info("3. Create MTF signal aggregator")
    logger.info("4. Update ML models to use multi-timeframe features")
    logger.info("5. Modify backtest engine for MTF strategies")
    
    return TIMEFRAMES

if __name__ == "__main__":
    setup_multi_timeframe()