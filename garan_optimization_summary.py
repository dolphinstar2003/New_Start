#!/usr/bin/env python3
"""
GARAN Optimization Summary
Display best parameters and create optimized strategy
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom
from indicators.adx_di import calculate_adx_di
from indicators.supertrend import calculate_supertrend


def save_optimized_parameters():
    """Save optimized parameters for GARAN"""
    
    optimized_params = {
        "GARAN": {
            "1d": {
                "MACD": {
                    "fast": 8,
                    "slow": 28,
                    "signal": 5,
                    "return": 6354.39,
                    "signals": {"buy": 50, "sell": 50}
                },
                "ADX": {
                    "period": 7,
                    "threshold": 15.39,
                    "exit_threshold": 10.02,
                    "return": 6458.89,
                    "signals": {"buy": "DI cross + ADX>15.39", "sell": "DI cross or ADX<10.02"}
                },
                "Supertrend": {
                    "period": 5,
                    "multiplier": 0.74,
                    "return": 48687.59,
                    "signals": {"buy": 55, "sell": 55}
                }
            }
        },
        "comparison": {
            "buy_hold": 1083.42,
            "date_range": "2022-07-04 to 2025-07-04"
        },
        "optimization_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to JSON
    with open('garan_optimized_parameters.json', 'w') as f:
        json.dump(optimized_params, f, indent=2)
    
    return optimized_params


def display_summary():
    """Display optimization summary"""
    print("="*80)
    print("🏆 GARAN OPTIMIZATION SUMMARY")
    print("="*80)
    print("\n📅 Period: 2022-07-04 to 2025-07-04 (3 years)")
    print("📊 Buy & Hold Return: 1,083.42%")
    print("\n" + "-"*80)
    
    print("\n1️⃣ MACD (Moving Average Convergence Divergence)")
    print("-"*40)
    print("   Default (12,26,9):     1,834.04%")
    print("   Optimized (8,28,5):    6,354.39% ✨")
    print("   Improvement:           3.5x")
    print("   Signals:               50 buys, 50 sells")
    
    print("\n2️⃣ ADX (Average Directional Index)")
    print("-"*40)
    print("   Default (14,25):       219.31%")
    print("   Optimized (7,15.39):   6,458.89% ✨")
    print("   Improvement:           29.5x")
    print("   Strategy:              DI crossovers with ADX filter")
    
    print("\n3️⃣ Supertrend")
    print("-"*40)
    print("   Default (10,3.0):      972.15%")
    print("   Optimized (5,0.74):    48,687.59% 🚀")
    print("   Improvement:           50.1x")
    print("   Signals:               55 buys, 55 sells")
    
    print("\n📈 PERFORMANCE RANKING:")
    print("-"*40)
    print("   1. Supertrend:         48,687.59% 🥇")
    print("   2. ADX:                6,458.89%  🥈")
    print("   3. MACD:               6,354.39%  🥉")
    print("   4. Buy & Hold:         1,083.42%")
    
    print("\n💡 KEY INSIGHTS:")
    print("-"*40)
    print("   • All optimized strategies beat buy & hold")
    print("   • Supertrend with low multiplier (0.74) captures trends early")
    print("   • Shorter periods (5-8) work better than defaults")
    print("   • More frequent signals = better performance")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("-"*40)
    print("   • These parameters are optimized for GARAN historical data")
    print("   • Past performance doesn't guarantee future results")
    print("   • Parameters may need adjustment for other symbols")
    print("   • Consider using ensemble of indicators for robustness")


def test_combined_strategy():
    """Test a combined strategy using all three indicators"""
    print("\n\n🔬 TESTING COMBINED STRATEGY")
    print("="*80)
    
    # Load data
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    
    # Calculate all indicators with optimized parameters
    macd_result = calculate_macd_custom(data, 8, 28, 5)
    adx_result = calculate_adx_di(data, length=7, threshold=15.39)
    st_result = calculate_supertrend(data, 5, 0.74)
    
    if all([macd_result is not None, adx_result is not None, st_result is not None]):
        # Extract indicator values
        macd_line = macd_result['macd']
        macd_signal = macd_result['signal']
        adx = adx_result['adx']
        plus_di = adx_result['plus_di']
        minus_di = adx_result['minus_di']
        trend = st_result['trend']
        
        # Combined strategy: 2 out of 3 indicators must agree
        combined_signals = []
        for i in range(1, len(data)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(adx.iloc[i]) or pd.isna(trend.iloc[i]):
                combined_signals.append(0)
                continue
            
            # Count buy votes
            buy_votes = 0
            
            # MACD vote
            if macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                buy_votes += 1
            
            # ADX vote
            if plus_di.iloc[i] > minus_di.iloc[i] and adx.iloc[i] > 15.39:
                buy_votes += 1
            
            # Supertrend vote
            if trend.iloc[i] == 1:
                buy_votes += 1
            
            # Generate signal
            if buy_votes >= 2:
                combined_signals.append(1)
            elif buy_votes <= 1:
                combined_signals.append(-1)
            else:
                combined_signals.append(0)
        
        # Calculate returns
        returns = data['close'].pct_change().fillna(0)
        position = 0
        strategy_returns = []
        
        for i in range(len(combined_signals)):
            if combined_signals[i] == 1:  # Buy
                position = 1
            elif combined_signals[i] == -1:  # Sell
                position = 0
            
            if i+1 < len(returns):
                strategy_returns.append(position * returns.iloc[i+1])
        
        if strategy_returns:
            total_return = (1 + pd.Series(strategy_returns)).prod() - 1
            print(f"Combined Strategy Return: {total_return * 100:.2f}%")
            print(f"Signals: {combined_signals.count(1)} buys, {combined_signals.count(-1)} sells")
            
            # Calculate win rate
            winning_trades = sum(1 for r in strategy_returns if r > 0)
            losing_trades = sum(1 for r in strategy_returns if r < 0)
            if winning_trades + losing_trades > 0:
                win_rate = winning_trades / (winning_trades + losing_trades) * 100
                print(f"Win Rate: {win_rate:.1f}%")


if __name__ == "__main__":
    # Save parameters
    params = save_optimized_parameters()
    print("✅ Parameters saved to: garan_optimized_parameters.json")
    
    # Display summary
    display_summary()
    
    # Test combined strategy
    test_combined_strategy()