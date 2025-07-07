#!/usr/bin/env python3
"""
Quick Rotation Test
Simple comparison of rotation improvements
"""
import asyncio
from datetime import datetime

# Import backtests
from backtest.rotation_backtest import run_rotation_backtest


async def main():
    print("\n" + "="*60)
    print("QUICK ROTATION ANALYSIS")
    print("="*60)
    
    # Test only 30 and 60 days for speed
    periods = [30, 60]
    
    print("\nRunning original rotation backtest...")
    for days in periods:
        print(f"\n{days}-day test:")
        result = await run_rotation_backtest(days)
        
        print(f"  Return: {result['total_return']:+.2f}%")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Max DD: {result['max_drawdown']:.2f}%")
        print(f"  Rotations: {result['total_rotations']}")
    
    print("\n" + "="*60)
    print("SUGGESTED IMPROVEMENTS:")
    print("="*60)
    
    print("\n1. **Position Sizing**: Use volatility-adjusted sizing")
    print("   - High volatility = smaller positions")
    print("   - Low volatility = larger positions")
    
    print("\n2. **Exit Rules**: Add trailing stops")
    print("   - Lock profits above 10%")
    print("   - Time-based exit for underperformers")
    
    print("\n3. **Entry Filters**: Improve timing")
    print("   - Check market breadth")
    print("   - Avoid entries in downtrends")
    
    print("\n4. **Scoring**: Add relative strength")
    print("   - Compare vs sector average")
    print("   - Momentum quality filter")
    
    print("\n5. **Risk Management**: Dynamic stops")
    print("   - ATR-based stops (2x ATR)")
    print("   - Tighten stops in profit")


if __name__ == "__main__":
    asyncio.run(main())