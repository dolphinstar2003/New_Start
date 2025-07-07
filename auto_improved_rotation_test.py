#!/usr/bin/env python3
"""
Automatic Improved Rotation Strategy Test
Runs comparison without user input
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

# Configure logger for cleaner output
logger.remove()
logger.add(sys.stderr, level="INFO")

sys.path.append(str(Path(__file__).parent))

from backtest.rotation_backtest import run_rotation_backtest
from backtest.improved_rotation_backtest import run_improved_rotation_backtest


async def main():
    """Run automatic comparison"""
    print("\n" + "="*80)
    print("🔄 AUTOMATIC ROTATION STRATEGY COMPARISON")
    print("="*80)
    
    # Test periods
    periods = [30, 60]
    
    print("\nComparing Original vs Improved Rotation Strategies")
    print("-"*80)
    
    results = []
    
    for days in periods:
        print(f"\n📊 Testing {days}-day period...")
        
        # Run original
        print("  Running original rotation...")
        original = await run_rotation_backtest(days)
        
        # Run improved
        print("  Running improved rotation...")
        improved = await run_improved_rotation_backtest(days)
        
        # Display comparison
        print(f"\n{days}-Day Results:")
        print(f"{'Metric':<20} {'Original':>15} {'Improved':>15} {'Change':>15}")
        print("-"*65)
        
        # Compare key metrics
        metrics = [
            ('Total Return %', 'total_return', '.2f'),
            ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
            ('Max Drawdown %', 'max_drawdown', '.2f'),
            ('Win Rate %', 'win_rate', '.1f'),
            ('Total Trades', 'total_trades', 'd')
        ]
        
        for metric_name, key, fmt in metrics:
            orig_val = original.get(key, 0)
            imp_val = improved.get(key, 0)
            
            # Calculate change
            if key == 'max_drawdown':
                change = orig_val - imp_val  # Lower is better
            else:
                change = imp_val - orig_val
            
            if fmt == 'd':
                change_str = f"{int(change):+d}"
            else:
                change_str = f"{change:+{fmt}}"
            
            print(f"{metric_name:<20} {orig_val:>15{fmt}} {imp_val:>15{fmt}} {change_str:>15}")
        
        # Show exit reasons
        if 'exit_reasons' in improved and improved['exit_reasons']:
            print(f"\n  Exit Reasons (Improved):")
            for reason, stats in improved['exit_reasons'].items():
                print(f"    {reason}: {stats['count']} trades, avg: {stats['avg_return']:.1f}%")
        
        results.append({
            'period': days,
            'orig_return': original['total_return'],
            'imp_return': improved['total_return'],
            'orig_sharpe': original['sharpe_ratio'],
            'imp_sharpe': improved['sharpe_ratio'],
            'orig_win_rate': original['win_rate'],
            'imp_win_rate': improved['win_rate']
        })
    
    # Summary
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Calculate improvements
    df['return_improvement'] = df['imp_return'] - df['orig_return']
    df['sharpe_improvement'] = df['imp_sharpe'] - df['orig_sharpe']
    df['win_rate_improvement'] = df['imp_win_rate'] - df['orig_win_rate']
    
    print("\n📈 Average Improvements:")
    print(f"  Return: {df['return_improvement'].mean():+.2f}% improvement")
    print(f"  Sharpe: {df['sharpe_improvement'].mean():+.2f} improvement")
    print(f"  Win Rate: {df['win_rate_improvement'].mean():+.1f}% improvement")
    
    # Key improvements implemented
    print("\n✅ Key Improvements in the New Strategy:")
    print("  1. Volatility-adjusted position sizing")
    print("  2. Trailing stops (10% activation, 5% trail)")
    print("  3. ATR-based dynamic stop losses")
    print("  4. Time-based exits for underperformers")
    print("  5. Market breadth checking")
    print("  6. Relative strength vs market")
    print("  7. Multi-timeframe momentum scoring")
    print("  8. Volume analysis for entry timing")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"rotation_improvement_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n📁 Results saved to: {filename}")


if __name__ == "__main__":
    asyncio.run(main())