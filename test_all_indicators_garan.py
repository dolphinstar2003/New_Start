#!/usr/bin/env python3
"""
Test All Indicators for GARAN - Step by Step
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom
from indicators.adx_di import calculate_adx_di
from indicators.supertrend import calculate_supertrend


def test_strategy_performance(data, signals):
    """Calculate strategy performance from signals"""
    returns = data['close'].pct_change().fillna(0)
    position = 0
    strategy_returns = []
    
    for i in range(len(signals)):
        if signals[i] == 1:  # Buy
            position = 1
        elif signals[i] == -1:  # Sell
            position = 0
        
        if i+1 < len(returns):
            strategy_returns.append(position * returns.iloc[i+1])
    
    if strategy_returns:
        total_return = (1 + pd.Series(strategy_returns)).prod() - 1
        return total_return * 100
    return 0


def test_indicators():
    """Test all indicators for GARAN"""
    print("🧪 Testing All Indicators for GARAN (1d)")
    print("="*80)
    
    # Load data
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    print(f"Data loaded: {data.shape}")
    
    # 1. TEST MACD
    print("\n1️⃣ MACD Test:")
    print("-"*40)
    
    # Default parameters
    macd_result = calculate_macd_custom(data, 12, 26, 9)
    if macd_result is not None:
        macd_line = macd_result['macd']
        macd_signal = macd_result['signal']
        
        # Generate signals
        signals = []
        for i in range(1, len(macd_line)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                signals.append(0)
            elif macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                signals.append(1)
            elif macd_line.iloc[i] < macd_signal.iloc[i] and macd_line.iloc[i-1] >= macd_signal.iloc[i-1]:
                signals.append(-1)
            else:
                signals.append(0)
        
        perf = test_strategy_performance(data, signals)
        print(f"Default (12,26,9): {perf:.2f}%")
        print(f"Signals: {signals.count(1)} buys, {signals.count(-1)} sells")
        
        # Optimized parameters
        macd_result2 = calculate_macd_custom(data, 8, 28, 5)
        if macd_result2 is not None:
            macd_line2 = macd_result2['macd']
            macd_signal2 = macd_result2['signal']
            
            signals2 = []
            for i in range(1, len(macd_line2)):
                if pd.isna(macd_line2.iloc[i]) or pd.isna(macd_signal2.iloc[i]):
                    signals2.append(0)
                elif macd_line2.iloc[i] > macd_signal2.iloc[i] and macd_line2.iloc[i-1] <= macd_signal2.iloc[i-1]:
                    signals2.append(1)
                elif macd_line2.iloc[i] < macd_signal2.iloc[i] and macd_line2.iloc[i-1] >= macd_signal2.iloc[i-1]:
                    signals2.append(-1)
                else:
                    signals2.append(0)
            
            perf2 = test_strategy_performance(data, signals2)
            print(f"Optimized (8,28,5): {perf2:.2f}%")
            print(f"Signals: {signals2.count(1)} buys, {signals2.count(-1)} sells")
    
    # 2. TEST ADX
    print("\n2️⃣ ADX Test:")
    print("-"*40)
    
    # Default ADX
    adx_result = calculate_adx_di(data, length=14, threshold=25)
    if adx_result is not None and not adx_result.empty:
        adx = adx_result['adx']
        
        # Generate signals
        signals = []
        threshold = 25
        for i in range(1, len(adx)):
            if pd.isna(adx.iloc[i]):
                signals.append(0)
            elif adx.iloc[i] > threshold and adx.iloc[i] > adx.iloc[i-1]:
                signals.append(1)
            elif adx.iloc[i] < threshold * 0.8:
                signals.append(-1)
            else:
                signals.append(0)
        
        perf = test_strategy_performance(data, signals)
        print(f"Default (14, 25): {perf:.2f}%")
        print(f"Signals: {signals.count(1)} buys, {signals.count(-1)} sells")
        print(f"ADX range: {adx.min():.2f} - {adx.max():.2f}")
    
    # 3. TEST SUPERTREND
    print("\n3️⃣ Supertrend Test:")
    print("-"*40)
    
    # Default Supertrend
    st_result = calculate_supertrend(data, 10, 3.0)
    if st_result is not None and not st_result.empty:
        trend = st_result['trend']
        
        # Generate signals
        signals = []
        for i in range(1, len(trend)):
            if pd.isna(trend.iloc[i]):
                signals.append(0)
            elif trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                signals.append(1)
            elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
                signals.append(-1)
            else:
                signals.append(0)
        
        perf = test_strategy_performance(data, signals)
        print(f"Default (10, 3.0): {perf:.2f}%")
        print(f"Signals: {signals.count(1)} buys, {signals.count(-1)} sells")
        
        # Optimized Supertrend
        st_result2 = calculate_supertrend(data, 5, 0.74)
        if st_result2 is not None and not st_result2.empty:
            trend2 = st_result2['trend']
            
            signals2 = []
            for i in range(1, len(trend2)):
                if pd.isna(trend2.iloc[i]):
                    signals2.append(0)
                elif trend2.iloc[i] == 1 and trend2.iloc[i-1] == -1:
                    signals2.append(1)
                elif trend2.iloc[i] == -1 and trend2.iloc[i-1] == 1:
                    signals2.append(-1)
                else:
                    signals2.append(0)
            
            perf2 = test_strategy_performance(data, signals2)
            print(f"Optimized (5, 0.74): {perf2:.2f}%")
            print(f"Signals: {signals2.count(1)} buys, {signals2.count(-1)} sells")
    
    # Buy & Hold comparison
    buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
    print(f"\n📊 Buy & Hold Return: {buy_hold:.2f}%")


if __name__ == "__main__":
    test_indicators()