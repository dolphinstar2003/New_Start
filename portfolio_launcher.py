#!/usr/bin/env python3
"""
Portfolio Strategy Launcher
Easy start for portfolio-based paper trading
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from trading.portfolio_paper_trading import PortfolioPaperTrader
from loguru import logger


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*80)
    print("🚀 PORTFOLIO PAPER TRADING SYSTEM")
    print("="*80)
    print("\n📊 Strategy Allocation:")
    print("  • 60% Realistic (Conservative)")
    print("  • 30% Hierarchical 4H + Trailing Stop (Aggressive)")
    print("  • 10% Cash Reserve")
    print("\n⚡ Features:")
    print("  • Automatic rebalancing")
    print("  • Risk management per strategy")
    print("  • Real-time performance tracking")
    print("  • Telegram notifications (if enabled)")
    print("\n" + "="*80)


async def main():
    """Main entry point"""
    print_welcome()
    
    # Get initial capital
    try:
        capital_input = input("\n💰 Enter initial capital (default: 100000): ").strip()
        initial_capital = float(capital_input) if capital_input else 100000
    except ValueError:
        print("Invalid input, using default: 100000")
        initial_capital = 100000
    
    print(f"\n✅ Starting with ${initial_capital:,.2f}")
    
    # Create trader
    trader = PortfolioPaperTrader(initial_capital=initial_capital)
    
    print("\n🔄 Starting portfolio paper trading...")
    print("📌 Press Ctrl+C to stop\n")
    
    try:
        # Start trading
        await trader.start_trading()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping portfolio trading...")
        trader.stop_trading()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        trader.stop_trading()
    
    # Show final status
    print("\n📊 Final Portfolio Status:")
    trader.print_portfolio_status()
    
    print("\n✅ Portfolio trading session ended")
    print("📁 Check logs for detailed history")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)