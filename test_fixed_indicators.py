#!/usr/bin/env python3
"""Test the fixed WaveTrend and VixFix strategies"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.wavetrend import calculate_wavetrend
from indicators.vixfix import calculate_vixfix

def test_fixed_wavetrend():
    """Test the fixed WaveTrend strategy"""
    print("=== TESTING FIXED WAVETREND STRATEGY ===")
    
    calc = IndicatorCalculator(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]
    data = calc.load_raw_data(symbol, '1d')
    
    if data is None:
        print("ERROR: Could not load data")
        return
    
    # Test with different parameter combinations
    test_params = [
        {'channel': 10, 'average': 21, 'ob': 50, 'os': -30},
        {'channel': 10, 'average': 21, 'ob': 60, 'os': -40},
        {'channel': 8, 'average': 20, 'ob': 45, 'os': -35},
    ]
    
    for params in test_params:
        print(f"\nTesting with params: {params}")
        
        result = calculate_wavetrend(data, params['channel'], params['average'])
        wt1 = result['wt1']
        wt2 = result['wt2']
        
        overbought = params['ob']
        oversold = params['os']
        
        # New strategy logic
        signals = []
        for i in range(2, len(wt1)):  # Start from 2 to check 2 bars back
            if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                signals.append(0)
                continue
            
            # Buy on bullish crossover in oversold territory or momentum reversal
            if (wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i-1] <= wt2.iloc[i-1] and wt1.iloc[i] < oversold) or \
               (wt1.iloc[i] < oversold and wt1.iloc[i] > wt1.iloc[i-1] and wt1.iloc[i-1] < wt1.iloc[i-2]):
                signals.append(1)
            # Sell on bearish crossover in overbought territory or momentum reversal
            elif (wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[i-1] >= wt2.iloc[i-1] and wt1.iloc[i] > overbought) or \
                 (wt1.iloc[i] > overbought and wt1.iloc[i] < wt1.iloc[i-1] and wt1.iloc[i-1] > wt1.iloc[i-2]):
                signals.append(-1)
            else:
                signals.append(0)
        
        returns = data['close'].pct_change().fillna(0)
        position = 0
        strategy_returns = []
        
        for i, signal in enumerate(signals):
            if signal == 1:
                position = 1
            elif signal == -1:
                position = 0
            
            returns_idx = i + 3  # Adjust for starting at index 2
            if returns_idx < len(returns):
                strategy_returns.append(position * returns.iloc[returns_idx])
        
        if strategy_returns:
            total_return = (1 + pd.Series(strategy_returns)).prod() - 1
            buy_signals = sum(1 for s in signals if s == 1)
            sell_signals = sum(1 for s in signals if s == -1)
            
            print(f"  Total return: {total_return * 100:.2f}%")
            print(f"  Buy signals: {buy_signals}")
            print(f"  Sell signals: {sell_signals}")
            print(f"  Total trades: {buy_signals + sell_signals}")


def test_fixed_vixfix():
    """Test the fixed VixFix strategy"""
    print("\n\n=== TESTING FIXED VIXFIX STRATEGY ===")
    
    calc = IndicatorCalculator(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]
    data = calc.load_raw_data(symbol, '1d')
    
    if data is None:
        print("ERROR: Could not load data")
        return
    
    # Test with different parameter combinations
    test_params = [
        {'length': 22, 'buy_threshold': 20, 'sell_threshold': 5},
        {'length': 22, 'buy_threshold': 25, 'sell_threshold': 7},
        {'length': 20, 'buy_threshold': 15, 'sell_threshold': 6},
    ]
    
    for params in test_params:
        print(f"\nTesting with params: {params}")
        
        result = calculate_vixfix(data, params['length'])
        vixfix = result['vixfix']
        bb_upper = result['bb_upper']
        
        buy_threshold = params['buy_threshold']
        sell_threshold = params['sell_threshold']
        
        # New strategy logic
        signals = []
        for i in range(1, len(vixfix)):
            if pd.isna(vixfix.iloc[i]) or pd.isna(bb_upper.iloc[i]):
                signals.append(0)
                continue
            
            # Buy on extreme fear spike and reversal
            buy_condition = (
                (vixfix.iloc[i] > buy_threshold and vixfix.iloc[i] < vixfix.iloc[i-1]) or
                (vixfix.iloc[i] > bb_upper.iloc[i] and vixfix.iloc[i] < vixfix.iloc[i-1])
            )
            
            # Sell when fear is extremely low and starting to rise
            sell_condition = (
                vixfix.iloc[i] < sell_threshold and vixfix.iloc[i] > vixfix.iloc[i-1]
            )
            
            if buy_condition:
                signals.append(1)
            elif sell_condition:
                signals.append(-1)
            else:
                signals.append(0)
        
        returns = data['close'].pct_change().fillna(0)
        position = 0
        strategy_returns = []
        
        for i, signal in enumerate(signals):
            if signal == 1:
                position = 1
            elif signal == -1:
                position = 0
            
            if i+1 < len(returns):
                strategy_returns.append(position * returns.iloc[i+1])
        
        if strategy_returns:
            total_return = (1 + pd.Series(strategy_returns)).prod() - 1
            buy_signals = sum(1 for s in signals if s == 1)
            sell_signals = sum(1 for s in signals if s == -1)
            
            print(f"  Total return: {total_return * 100:.2f}%")
            print(f"  Buy signals: {buy_signals}")
            print(f"  Sell signals: {sell_signals}")
            print(f"  Total trades: {buy_signals + sell_signals}")


if __name__ == "__main__":
    test_fixed_wavetrend()
    test_fixed_vixfix()