#!/usr/bin/env python3
"""
Test Paper Trading System
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from paper_trader import PaperTrader
from data_fetcher import FALLBACK_PRICES
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("\n" + "="*80)
    print("🧪 TESTING PAPER TRADING SYSTEM")
    print("="*80)
    
    # Create paper trader
    trader = PaperTrader()
    
    # Override data fetcher to use fallback prices
    trader.data_fetcher.get_current_prices = lambda: FALLBACK_PRICES
    
    # Override trading hours check for testing
    trader.check_trading_hours = lambda: True
    
    print("\n1️⃣ Running single trading cycle...")
    trader.run_trading_cycle()
    
    print("\n2️⃣ Generating report...")
    trader.generate_daily_report()
    
    print("\n✅ Paper trading test completed!")


if __name__ == "__main__":
    main()