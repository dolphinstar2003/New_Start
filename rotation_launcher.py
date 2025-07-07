#!/usr/bin/env python3
"""
Dynamic Rotation Portfolio Launcher
"""
import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from trading.portfolio_rotation_trader import PortfolioRotationTrader
from loguru import logger


def print_welcome():
    """Print welcome message"""
    print("\n" + "="*80)
    print("🔄 DYNAMIC TOP 10 ROTATION PORTFOLIO")
    print("="*80)
    print("\n📊 Strategy Features:")
    print("  • Continuously scans 20 stocks for best opportunities")
    print("  • Maintains portfolio of top 10 stocks")
    print("  • Rotates out weak performers automatically")
    print("  • Dynamic position sizing based on score (8-15%)")
    print("\n📈 Scoring System:")
    print("  • 30% Momentum (recent price performance)")
    print("  • 25% Trend (ADX and directional indicators)")
    print("  • 20% Volatility (lower is better)")
    print("  • 25% Technical Indicators (signals)")
    print("\n⚡ Rotation Rules:")
    print("  • Sells stocks no longer in top 10")
    print("  • Takes profit on saturated positions (12%+)")
    print("  • Stops loss at -5%")
    print("  • Max 3 trades per rotation")
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
    trader = PortfolioRotationTrader(initial_capital=initial_capital)
    
    print("\n🔄 Starting dynamic rotation trading...")
    print("📌 Press Ctrl+C to stop\n")
    
    try:
        # Start trading
        await trader.start_trading()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping rotation trading...")
        trader.stop_trading()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        trader.stop_trading()
    
    # Show final report
    print("\n📊 Final Portfolio Status:")
    trader.print_portfolio_status()
    
    # Performance report
    report = trader.get_performance_report()
    print(f"\n📈 Performance Summary:")
    print(f"  • Total Rotations: {report['rotation_metrics']['total_rotations']}")
    print(f"  • Average Holding Period: {report['rotation_metrics']['avg_holding_period']:.1f} days")
    print(f"  • Rotation Win Rate: {report['rotation_metrics']['rotation_win_rate']:.1f}%")
    
    print("\n✅ Rotation trading session ended")
    print("📁 Check logs for detailed history")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)