#!/usr/bin/env python3
"""
Quick Rotation Comparison
Shows key improvements in the improved rotation strategy
"""
import asyncio
from datetime import datetime
import pandas as pd

# Import backtest runners
from backtest.rotation_backtest import run_rotation_backtest
from backtest.improved_rotation_backtest import run_improved_rotation_backtest


async def quick_test():
    """Run quick 30-day comparison"""
    print("\n" + "="*60)
    print("QUICK ROTATION STRATEGY COMPARISON (30 days)")
    print("="*60)
    
    # Run original
    print("\n🔄 Running original rotation...")
    original = await run_rotation_backtest(30)
    
    # Run improved
    print("\n✨ Running improved rotation...")
    improved = await run_improved_rotation_backtest(30)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<20} {'Original':>12} {'Improved':>12} {'Change':>12}")
    print("-"*56)
    
    # Key metrics
    metrics = [
        ('Total Return', original['total_return'], improved['total_return'], '%'),
        ('Sharpe Ratio', original['sharpe_ratio'], improved['sharpe_ratio'], ''),
        ('Win Rate', original['win_rate'], improved['win_rate'], '%'),
        ('Max Drawdown', original['max_drawdown'], improved['max_drawdown'], '%'),
        ('Total Trades', original['total_trades'], improved['total_trades'], ''),
    ]
    
    for name, orig, imp, unit in metrics:
        change = imp - orig
        if name == 'Max Drawdown':
            change = orig - imp  # Lower is better
        
        if unit == '%':
            print(f"{name:<20} {orig:>11.1f}{unit} {imp:>11.1f}{unit} {change:>+11.1f}{unit}")
        elif name == 'Total Trades':
            print(f"{name:<20} {int(orig):>12} {int(imp):>12} {int(change):>+12}")
        else:
            print(f"{name:<20} {orig:>12.2f} {imp:>12.2f} {change:>+12.2f}")
    
    # Show improvements
    print("\n" + "="*60)
    print("KEY IMPROVEMENTS IN THE NEW STRATEGY")
    print("="*60)
    
    improvements = [
        "✅ Volatility-Adjusted Position Sizing",
        "   - Larger positions in low-volatility stocks",
        "   - Smaller positions in high-volatility stocks",
        "",
        "✅ Trailing Stop Implementation",
        "   - Activates at 10% profit",
        "   - Trails by 5% to lock in gains",
        "",
        "✅ ATR-Based Stop Losses",
        "   - Dynamic stops based on volatility",
        "   - 2x ATR multiplier for breathing room",
        "",
        "✅ Time-Based Exits",
        "   - Exit positions after 30 days if < 5% return",
        "   - Frees up capital for better opportunities",
        "",
        "✅ Market Breadth Checking",
        "   - Reduces positions in weak markets",
        "   - Requires 40% positive momentum",
        "",
        "✅ Relative Strength Scoring",
        "   - Compares performance vs market average",
        "   - Prioritizes outperformers"
    ]
    
    for line in improvements:
        print(line)
    
    # Exit reasons if available
    if 'exit_reasons' in improved and improved['exit_reasons']:
        print("\n" + "="*60)
        print("EXIT REASON ANALYSIS (Improved Strategy)")
        print("="*60)
        
        total_exits = sum(stats['count'] for stats in improved['exit_reasons'].values())
        for reason, stats in sorted(improved['exit_reasons'].items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
            pct = (stats['count'] / total_exits * 100) if total_exits > 0 else 0
            avg_return = stats['avg_return']
            print(f"{reason:.<20} {stats['count']:>3} trades ({pct:>5.1f}%) | Avg Return: {avg_return:>+6.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    return_improvement = improved['total_return'] - original['total_return']
    win_rate_improvement = improved['win_rate'] - original['win_rate']
    
    print(f"\n🎯 The improved strategy achieved:")
    print(f"   • {return_improvement:+.1f}% better returns")
    print(f"   • {win_rate_improvement:+.1f}% higher win rate")
    
    if improved['sharpe_ratio'] > original['sharpe_ratio']:
        print(f"   • Better risk-adjusted returns (Sharpe +{improved['sharpe_ratio'] - original['sharpe_ratio']:.2f})")
    
    if improved['max_drawdown'] < original['max_drawdown']:
        dd_reduction = original['max_drawdown'] - improved['max_drawdown']
        print(f"   • {dd_reduction:.1f}% lower maximum drawdown")


if __name__ == "__main__":
    asyncio.run(quick_test())