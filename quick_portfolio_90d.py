#!/usr/bin/env python3
"""Quick 90-day portfolio backtest"""
import asyncio
from portfolio_backtest_runner import PortfolioBacktester

async def main():
    print("\n🚀 Running 90-day Portfolio Backtest...")
    print("💰 Capital: $100,000")
    print("📊 Strategy: 60% Realistic + 30% Hierarchical 4H + 10% Cash\n")
    
    backtester = PortfolioBacktester(100000)
    await backtester.run_multiple_periods([90])

if __name__ == "__main__":
    asyncio.run(main())