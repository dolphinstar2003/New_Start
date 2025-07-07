#!/usr/bin/env python3
"""
Improved Rotation Strategy Runner
Compare original vs improved rotation strategies
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


async def compare_rotation_strategies():
    """Compare original and improved rotation strategies"""
    periods = [30, 60, 90]
    
    print("\n" + "="*80)
    print("🔄 ROTATION STRATEGY COMPARISON")
    print("="*80)
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
            ('Total Return %', 'total_return', '.2f', '+'),
            ('Sharpe Ratio', 'sharpe_ratio', '.2f', '+'),
            ('Max Drawdown %', 'max_drawdown', '.2f', '-'),
            ('Win Rate %', 'win_rate', '.1f', '+'),
            ('Total Trades', 'total_trades', 'd', ''),
            ('Avg Positions', 'avg_positions', '.1f', '')
        ]
        
        result = {'period': days}
        
        for metric_name, key, fmt, direction in metrics:
            orig_val = original.get(key, 0)
            imp_val = improved.get(key, 0)
            
            # Calculate change
            if direction == '+':
                change = imp_val - orig_val
                change_str = f"+{change:{fmt}}" if change > 0 else f"{change:{fmt}}"
            elif direction == '-':
                change = orig_val - imp_val  # Lower is better
                change_str = f"+{change:{fmt}}" if change > 0 else f"{change:{fmt}}"
            else:
                change = imp_val - orig_val
                if fmt == 'd':
                    change_str = f"{int(change):+d}"
                else:
                    change_str = f"{change:+{fmt}}"
            
            print(f"{metric_name:<20} {orig_val:>15{fmt}} {imp_val:>15{fmt}} {change_str:>15}")
            
            # Store for summary
            result[f'orig_{key}'] = orig_val
            result[f'imp_{key}'] = imp_val
            result[f'change_{key}'] = change
        
        # Show exit reasons if available
        if 'exit_reasons' in improved and improved['exit_reasons']:
            print(f"\n  Improved Exit Reasons:")
            for reason, stats in improved['exit_reasons'].items():
                print(f"    {reason}: {stats['count']} trades, avg return: {stats['avg_return']:.1f}%")
        
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("IMPROVEMENT SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    # Calculate average improvements
    print("\n📈 Average Improvements:")
    print(f"  Return: {df['change_total_return'].mean():+.2f}% improvement")
    print(f"  Sharpe: {df['change_sharpe_ratio'].mean():+.2f} improvement")
    print(f"  Win Rate: {df['change_win_rate'].mean():+.1f}% improvement")
    
    # Best improvement
    best_return = df.loc[df['change_total_return'].idxmax()]
    print(f"\n🏆 Best Return Improvement: {best_return['period']}-day period")
    print(f"   Original: {best_return['orig_total_return']:.2f}% → Improved: {best_return['imp_total_return']:.2f}%")
    print(f"   Gain: +{best_return['change_total_return']:.2f}%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"improved_rotation_comparison_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n📁 Results saved to: {filename}")
    
    return df


async def test_improved_only():
    """Test improved strategy in detail"""
    print("\n" + "="*60)
    print("IMPROVED ROTATION STRATEGY TEST")
    print("="*60)
    
    print("\nKey Improvements:")
    print("  ✅ Volatility-adjusted position sizing")
    print("  ✅ Trailing stops (10% activation, 5% trail)")
    print("  ✅ ATR-based stop losses")
    print("  ✅ Time-based exits (30 days < 5% return)")
    print("  ✅ Market breadth checking")
    print("  ✅ Relative strength scoring")
    
    periods = [30, 60, 90, 120]
    
    for days in periods:
        print(f"\n📊 {days}-day backtest...")
        result = await run_improved_rotation_backtest(days)
        
        print(f"\nResults:")
        print(f"  Return: {result['total_return']:+.2f}%")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Max DD: {result['max_drawdown']:.2f}%")
        print(f"  Rotations: {result['total_rotations']}")
        
        if result.get('exit_reasons'):
            print("\n  Exit Breakdown:")
            total_exits = sum(stats['count'] for stats in result['exit_reasons'].values())
            for reason, stats in sorted(result['exit_reasons'].items(), 
                                       key=lambda x: x[1]['count'], reverse=True):
                pct = (stats['count'] / total_exits * 100) if total_exits > 0 else 0
                print(f"    {reason}: {stats['count']} ({pct:.0f}%), avg: {stats['avg_return']:.1f}%")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("IMPROVED ROTATION STRATEGY TESTER")
    print("="*60)
    
    print("\nOptions:")
    print("1. Compare Original vs Improved")
    print("2. Test Improved Only")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        asyncio.run(compare_rotation_strategies())
    elif choice == '2':
        asyncio.run(test_improved_only())
    elif choice == '3':
        print("Goodbye!")
        return
    else:
        print("Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()