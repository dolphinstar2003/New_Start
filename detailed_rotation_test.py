#!/usr/bin/env python3
"""
Detailed Rotation Test
Shows detailed comparison with 60-day period
"""
import asyncio
from datetime import datetime

from backtest.rotation_backtest import run_rotation_backtest
from backtest.improved_rotation_backtest import run_improved_rotation_backtest


async def main():
    print("\n" + "="*70)
    print("DETAILED ROTATION STRATEGY COMPARISON")
    print("="*70)
    
    # Test 60 days for more meaningful results
    days = 60
    
    print(f"\nTesting {days}-day period for more complete results...")
    print("-"*70)
    
    # Run original
    print("\n📊 Original Rotation Strategy:")
    original = await run_rotation_backtest(days)
    
    print(f"  • Return: {original['total_return']:+.2f}%")
    print(f"  • Sharpe: {original['sharpe_ratio']:.2f}")
    print(f"  • Win Rate: {original['win_rate']:.1f}%")
    print(f"  • Max DD: {original['max_drawdown']:.2f}%")
    print(f"  • Trades: {original['total_trades']} (Buy: {original['buy_trades']}, Sell: {original['sell_trades']})")
    print(f"  • Rotations: {original['total_rotations']}")
    
    # Run improved
    print("\n✨ Improved Rotation Strategy:")
    improved = await run_improved_rotation_backtest(days)
    
    print(f"  • Return: {improved['total_return']:+.2f}%")
    print(f"  • Sharpe: {improved['sharpe_ratio']:.2f}")
    print(f"  • Win Rate: {improved['win_rate']:.1f}%")
    print(f"  • Max DD: {improved['max_drawdown']:.2f}%")
    print(f"  • Trades: {improved['total_trades']} (Buy: {improved['buy_trades']}, Sell: {improved['sell_trades']})")
    print(f"  • Rotations: {improved['total_rotations']}")
    
    # Detailed comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Original':>15} {'Improved':>15} {'Difference':>15}")
    print("-"*70)
    
    # Calculate differences
    metrics = [
        ('Total Return (%)', original['total_return'], improved['total_return']),
        ('Sharpe Ratio', original['sharpe_ratio'], improved['sharpe_ratio']),
        ('Max Drawdown (%)', original['max_drawdown'], improved['max_drawdown']),
        ('Win Rate (%)', original['win_rate'], improved['win_rate']),
        ('Winning Trades', original.get('winning_trades', 0), improved.get('winning_trades', 0)),
        ('Losing Trades', original.get('losing_trades', 0), improved.get('losing_trades', 0)),
        ('Avg Win (%)', original.get('avg_win', 0), improved.get('avg_win', 0)),
        ('Avg Loss (%)', original.get('avg_loss', 0), improved.get('avg_loss', 0)),
    ]
    
    for name, orig, imp in metrics:
        if 'Drawdown' in name:
            diff = orig - imp  # Lower is better
            sign = '+' if diff > 0 else ''
        else:
            diff = imp - orig
            sign = '+' if diff > 0 else ''
        
        if '%' in name:
            print(f"{name:<25} {orig:>15.2f} {imp:>15.2f} {sign}{diff:>14.2f}")
        elif 'Trades' in name:
            print(f"{name:<25} {int(orig):>15} {int(imp):>15} {sign}{int(diff):>14}")
        else:
            print(f"{name:<25} {orig:>15.2f} {imp:>15.2f} {sign}{diff:>14.2f}")
    
    # Exit reasons analysis
    if 'exit_reasons' in improved and improved['exit_reasons']:
        print("\n" + "="*70)
        print("EXIT STRATEGY ANALYSIS (Improved)")
        print("="*70)
        
        print(f"\n{'Exit Reason':<20} {'Count':>10} {'Avg Return':>15}")
        print("-"*45)
        
        for reason, stats in sorted(improved['exit_reasons'].items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
            print(f"{reason:<20} {stats['count']:>10} {stats['avg_return']:>14.1f}%")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    print("\n🔍 Analysis:")
    
    # Return analysis
    if improved['total_return'] > original['total_return']:
        print(f"  ✅ Improved strategy generated {improved['total_return'] - original['total_return']:.1f}% higher returns")
    else:
        print(f"  ⚠️  Original strategy had {original['total_return'] - improved['total_return']:.1f}% better returns")
    
    # Risk analysis
    if improved['sharpe_ratio'] > original['sharpe_ratio']:
        print(f"  ✅ Better risk-adjusted returns (Sharpe +{improved['sharpe_ratio'] - original['sharpe_ratio']:.2f})")
    
    if improved['max_drawdown'] < original['max_drawdown']:
        print(f"  ✅ Lower risk with {original['max_drawdown'] - improved['max_drawdown']:.1f}% smaller drawdowns")
    
    # Trading efficiency
    trade_reduction = ((original['total_trades'] - improved['total_trades']) / original['total_trades'] * 100)
    print(f"  📊 {trade_reduction:.0f}% fewer trades (lower costs)")
    
    print("\n💡 Improvements implemented:")
    print("  1. Volatility-based position sizing")
    print("  2. Trailing stops to protect profits")
    print("  3. Dynamic ATR-based stop losses")
    print("  4. Time-based exits for underperformers")
    print("  5. Market breadth filtering")
    print("  6. Relative strength vs market")


if __name__ == "__main__":
    asyncio.run(main())