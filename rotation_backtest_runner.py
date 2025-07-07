#!/usr/bin/env python3
"""
Rotation Strategy Backtest Runner
Test the dynamic top 10 rotation strategy over different periods
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.append(str(Path(__file__).parent))

from backtest.rotation_backtest import run_rotation_backtest
from loguru import logger


async def run_multiple_backtests():
    """Run backtests for multiple periods"""
    periods = [30, 60, 90, 120, 180]
    results = []
    
    print("\n" + "="*80)
    print("🔄 DYNAMIC ROTATION STRATEGY BACKTEST")
    print("="*80)
    print("\nStrategy: Top 10 stocks with dynamic rotation")
    print("Features: Score-based selection, profit taking, stop loss")
    print(f"\nTesting periods: {periods} days")
    print("-"*80)
    
    for days in periods:
        print(f"\n📊 Running {days}-day backtest...")
        
        result = await run_rotation_backtest(days)
        
        # Display results
        print(f"\n{days}-Day Results:")
        print(f"  Return: {result['total_return']:+.2f}%")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Max DD: {result['max_drawdown']:.2f}%")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Trades: {result['total_trades']} (Buy: {result['buy_trades']}, Sell: {result['sell_trades']})")
        print(f"  Rotations: {result['total_rotations']}")
        print(f"  Avg Positions: {result['avg_positions']:.1f}")
        
        if result['winning_trades'] > 0:
            print(f"  Avg Win: {result['avg_win']:+.2f}%")
        if result['losing_trades'] > 0:
            print(f"  Avg Loss: {result['avg_loss']:.2f}%")
        
        # Store results
        results.append({
            'period': days,
            'return': result['total_return'],
            'sharpe': result['sharpe_ratio'],
            'max_dd': result['max_drawdown'],
            'win_rate': result['win_rate'],
            'trades': result['total_trades'],
            'rotations': result['total_rotations'],
            'avg_positions': result['avg_positions']
        })
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Best period
    best_return = df.loc[df['return'].idxmax()]
    best_sharpe = df.loc[df['sharpe'].idxmax()]
    
    print(f"\n📈 Best Return: {best_return['period']} days ({best_return['return']:+.2f}%)")
    print(f"📊 Best Sharpe: {best_sharpe['period']} days ({best_sharpe['sharpe']:.2f})")
    
    # Average metrics
    print(f"\n📊 Average Metrics Across All Periods:")
    print(f"  Avg Return: {df['return'].mean():+.2f}%")
    print(f"  Avg Sharpe: {df['sharpe'].mean():.2f}")
    print(f"  Avg Win Rate: {df['win_rate'].mean():.1f}%")
    print(f"  Avg Rotations: {df['rotations'].mean():.1f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"rotation_backtest_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n📁 Results saved to: {filename}")


async def quick_test():
    """Quick 30-day test"""
    print("\n🚀 Quick 30-day rotation backtest...")
    result = await run_rotation_backtest(30)
    
    print(f"\n📊 Results:")
    print(f"Return: {result['total_return']:+.2f}%")
    print(f"Sharpe: {result['sharpe_ratio']:.2f}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Rotations: {result['total_rotations']}")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("ROTATION STRATEGY BACKTESTER")
    print("="*60)
    
    print("\nOptions:")
    print("1. Quick test (30 days)")
    print("2. Full test (30-180 days)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        asyncio.run(quick_test())
    elif choice == '2':
        asyncio.run(run_multiple_backtests())
    elif choice == '3':
        print("Goodbye!")
        return
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBacktest cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()