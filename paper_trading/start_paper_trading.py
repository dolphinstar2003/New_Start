#!/usr/bin/env python3
"""
Quick Start Script for Paper Trading
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from paper_trader import PaperTrader
import logging

# Setup simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def main():
    print("\n" + "="*60)
    print("🚀 PAPER TRADING SYSTEM")
    print("="*60)
    print("\nInitializing paper trading system...")
    
    # Create paper trader
    trader = PaperTrader()
    
    # Show initial state
    print("\n📊 Initial Portfolio Status:")
    print("-"*60)
    
    summary = trader.portfolio_manager.get_summary()
    for _, portfolio in summary.iterrows():
        print(f"\n{portfolio['name'].upper()} Strategy:")
        print(f"  Initial Capital: 50,000 TL")
        print(f"  Current Value: {portfolio['portfolio_value']:,.2f} TL")
        print(f"  Positions: {portfolio['num_positions']}")
    
    print("\n" + "="*60)
    print("Options:")
    print("  1. Run once (single trading cycle)")
    print("  2. Generate report")
    print("  3. Run continuous (live trading)")
    print("  4. Exit")
    print("="*60)
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        print("\n▶️  Running single trading cycle...")
        trader.run_trading_cycle()
        
    elif choice == '2':
        print("\n📄 Generating report...")
        trader.generate_daily_report()
        
    elif choice == '3':
        print("\n🔄 Starting continuous trading mode...")
        print("   (Press Ctrl+C to stop)")
        trader.run_continuous()
        
    elif choice == '4':
        print("\n👋 Exiting...")
        return
        
    else:
        print("\n❌ Invalid option!")


if __name__ == "__main__":
    main()