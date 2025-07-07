#!/usr/bin/env python3
"""
Enhanced Rotation Strategy Runner
Compare original vs enhanced rotation strategies
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

# Disable debug logs for cleaner output
logger.remove()
logger.add(sys.stderr, level="INFO")

sys.path.append(str(Path(__file__).parent))

from backtest.rotation_backtest import run_rotation_backtest
from backtest.enhanced_rotation_backtest import run_enhanced_rotation_backtest


async def compare_strategies():
    """Compare original and enhanced rotation strategies"""
    periods = [30, 60, 90, 120]
    
    print("\n" + "="*80)
    print("🔄 ROTATION STRATEGY COMPARISON")
    print("="*80)
    print("\nComparing Original vs Enhanced Rotation Strategies")
    print("-"*80)
    
    comparison_results = []
    
    for days in periods:
        print(f"\n📊 Testing {days}-day period...")
        
        # Run original
        print("  Running original rotation...")
        original = await run_rotation_backtest(days)
        
        # Run enhanced
        print("  Running enhanced rotation...")
        enhanced = await run_enhanced_rotation_backtest(days)
        
        # Display comparison
        print(f"\n{days}-Day Results:")
        print(f"{'Metric':<20} {'Original':>15} {'Enhanced':>15} {'Improvement':>15}")
        print("-"*65)
        
        metrics = [
            ('Total Return %', 'total_return', '+.2f'),
            ('Sharpe Ratio', 'sharpe_ratio', '.2f'),
            ('Max Drawdown %', 'max_drawdown', '.2f'),
            ('Win Rate %', 'win_rate', '.1f'),
            ('Total Trades', 'total_trades', 'd'),
            ('Avg Positions', 'avg_positions', '.1f')
        ]
        
        comparison = {'period': days}
        
        for metric_name, key, fmt in metrics:
            orig_val = original.get(key, 0)
            enh_val = enhanced.get(key, 0)
            
            # Calculate improvement
            if key in ['total_return', 'sharpe_ratio', 'win_rate']:
                improvement = enh_val - orig_val
                if fmt.startswith('+'):
                    imp_str = f"{improvement:{fmt}}"
                else:
                    imp_str = f"{improvement:+{fmt}}"
            elif key == 'max_drawdown':
                improvement = orig_val - enh_val  # Lower is better
                if fmt.startswith('+'):
                    imp_str = f"{improvement:{fmt}}"
                else:
                    imp_str = f"{improvement:+{fmt}}"
            else:
                improvement = enh_val - orig_val
                if fmt == 'd':
                    imp_str = f"{int(improvement):+d}"
                else:
                    imp_str = f"{improvement:+.1f}"
            
            print(f"{metric_name:<20} {orig_val:>15{fmt}} {enh_val:>15{fmt}} {imp_str:>15}")
            
            # Store for summary
            comparison[f'orig_{key}'] = orig_val
            comparison[f'enh_{key}'] = enh_val
            comparison[f'imp_{key}'] = improvement
        
        # Show enhanced-specific features
        if 'partial_exits' in enhanced:
            print(f"\n  Enhanced Features:")
            print(f"    Partial Exits: {enhanced['partial_exits']}")
        
        if 'exit_reasons' in enhanced and enhanced['exit_reasons']:
            print(f"    Exit Reasons: {', '.join(enhanced['exit_reasons'].keys())}")
        
        comparison_results.append(comparison)
    
    # Summary
    print("\n" + "="*80)
    print("OVERALL COMPARISON SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(comparison_results)
    
    # Average improvements
    print("\n📈 Average Improvements (Enhanced vs Original):")
    print(f"  Return: {df['imp_total_return'].mean():+.2f}%")
    print(f"  Sharpe: {df['imp_sharpe_ratio'].mean():+.2f}")
    print(f"  Win Rate: {df['imp_win_rate'].mean():+.1f}%")
    print(f"  Max DD Reduction: {df['imp_max_drawdown'].mean():+.2f}%")
    
    # Best improvements
    best_return_imp = df.loc[df['imp_total_return'].idxmax()]
    best_sharpe_imp = df.loc[df['imp_sharpe_ratio'].idxmax()]
    
    print(f"\n🏆 Best Improvements:")
    print(f"  Return: {best_return_imp['period']} days (+{best_return_imp['imp_total_return']:.2f}%)")
    print(f"  Sharpe: {best_sharpe_imp['period']} days (+{best_sharpe_imp['imp_sharpe_ratio']:.2f})")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"rotation_comparison_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\n📁 Detailed comparison saved to: {filename}")


async def test_enhanced_only():
    """Test enhanced strategy only"""
    print("\n🚀 Testing Enhanced Rotation Strategy...")
    
    periods = [30, 60, 90, 120, 180]
    results = []
    
    for days in periods:
        print(f"\n📊 {days}-day backtest...")
        result = await run_enhanced_rotation_backtest(days)
        
        print(f"  Return: {result['total_return']:+.2f}%")
        print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        
        results.append({
            'period': days,
            'return': result['total_return'],
            'sharpe': result['sharpe_ratio'],
            'win_rate': result['win_rate'],
            'max_dd': result['max_drawdown']
        })
    
    # Summary
    df = pd.DataFrame(results)
    print("\n📊 Enhanced Strategy Summary:")
    print(df.to_string(index=False))
    
    print(f"\n  Avg Return: {df['return'].mean():+.2f}%")
    print(f"  Avg Sharpe: {df['sharpe'].mean():.2f}")
    print(f"  Avg Win Rate: {df['win_rate'].mean():.1f}%")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("ENHANCED ROTATION STRATEGY TESTER")
    print("="*60)
    
    print("\nEnhancements include:")
    print("  ✅ Dynamic rotation intervals based on volatility")
    print("  ✅ Enhanced scoring with relative strength")
    print("  ✅ Kelly Criterion position sizing")
    print("  ✅ Market regime detection")
    print("  ✅ Partial profit taking")
    print("  ✅ ATR-based dynamic stops")
    print("  ✅ Time-based exits")
    
    print("\nOptions:")
    print("1. Compare Original vs Enhanced")
    print("2. Test Enhanced Only")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        asyncio.run(compare_strategies())
    elif choice == '2':
        asyncio.run(test_enhanced_only())
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