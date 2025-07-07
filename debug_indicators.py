#!/usr/bin/env python3
"""Debug WaveTrend and VixFix indicators"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.wavetrend import calculate_wavetrend
from indicators.vixfix import calculate_vixfix

def debug_wavetrend():
    """Debug WaveTrend indicator"""
    print("\n=== DEBUGGING WAVETREND ===")
    
    # Load sample data
    calc = IndicatorCalculator(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]
    data = calc.load_raw_data(symbol, '1d')
    
    if data is None:
        print("ERROR: Could not load data")
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: {data.index[0]} to {data.index[-1]}")
    
    # Calculate WaveTrend with default parameters
    result = calculate_wavetrend(data, 10, 21)
    
    print(f"\nWaveTrend result shape: {result.shape}")
    print(f"WaveTrend columns: {list(result.columns)}")
    
    # Check for NaN values
    wt1 = result['wt1']
    wt2 = result['wt2']
    print(f"\nWT1 NaN count: {wt1.isna().sum()} / {len(wt1)}")
    print(f"WT2 NaN count: {wt2.isna().sum()} / {len(wt2)}")
    
    # Print some non-NaN values
    valid_idx = wt1.notna() & wt2.notna()
    valid_wt1 = wt1[valid_idx]
    valid_wt2 = wt2[valid_idx]
    
    if len(valid_wt1) > 0:
        print(f"\nWT1 range: {valid_wt1.min():.2f} to {valid_wt1.max():.2f}")
        print(f"WT2 range: {valid_wt2.min():.2f} to {valid_wt2.max():.2f}")
        print(f"WT1 mean: {valid_wt1.mean():.2f}")
        print(f"WT2 mean: {valid_wt2.mean():.2f}")
    
    # Test strategy logic
    overbought = 60
    oversold = -60
    
    buy_signals = 0
    sell_signals = 0
    
    for i in range(1, len(wt1)):
        if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
            continue
        
        # Buy condition
        if wt1.iloc[i] < oversold and wt1.iloc[i] > wt1.iloc[i-1] and wt1.iloc[i] > wt2.iloc[i]:
            buy_signals += 1
        
        # Sell condition
        elif wt1.iloc[i] > overbought and wt1.iloc[i] < wt1.iloc[i-1] and wt1.iloc[i] < wt2.iloc[i]:
            sell_signals += 1
    
    print(f"\nBuy signals: {buy_signals}")
    print(f"Sell signals: {sell_signals}")
    
    # Check how often conditions are met
    oversold_count = (wt1 < oversold).sum()
    overbought_count = (wt1 > overbought).sum()
    print(f"\nTimes oversold (<{oversold}): {oversold_count}")
    print(f"Times overbought (>{overbought}): {overbought_count}")
    
    # Test with different thresholds
    for ob in [40, 50, 60, 70]:
        for os in [-70, -60, -50, -40]:
            buy_count = 0
            sell_count = 0
            
            for i in range(1, len(wt1)):
                if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                    continue
                
                if wt1.iloc[i] < os and wt1.iloc[i] > wt1.iloc[i-1] and wt1.iloc[i] > wt2.iloc[i]:
                    buy_count += 1
                elif wt1.iloc[i] > ob and wt1.iloc[i] < wt1.iloc[i-1] and wt1.iloc[i] < wt2.iloc[i]:
                    sell_count += 1
            
            if buy_count > 0 or sell_count > 0:
                print(f"OB={ob}, OS={os}: Buy={buy_count}, Sell={sell_count}")


def debug_vixfix():
    """Debug VixFix indicator"""
    print("\n\n=== DEBUGGING VIXFIX ===")
    
    # Load sample data
    calc = IndicatorCalculator(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]
    data = calc.load_raw_data(symbol, '1d')
    
    if data is None:
        print("ERROR: Could not load data")
        return
    
    print(f"Data shape: {data.shape}")
    
    # Calculate VixFix with default parameters
    result = calculate_vixfix(data, 22)
    
    print(f"\nVixFix result shape: {result.shape}")
    print(f"VixFix columns: {list(result.columns)}")
    
    # Check for NaN values
    vixfix = result['vixfix']
    print(f"\nVixFix NaN count: {vixfix.isna().sum()} / {len(vixfix)}")
    
    # Print value ranges
    valid_vix = vixfix[vixfix.notna()]
    if len(valid_vix) > 0:
        print(f"\nVixFix range: {valid_vix.min():.2f} to {valid_vix.max():.2f}")
        print(f"VixFix mean: {valid_vix.mean():.2f}")
        print(f"VixFix std: {valid_vix.std():.2f}")
    
    # Test strategy logic with different thresholds
    for threshold in [10, 15, 20, 25, 30]:
        buy_signals = 0
        sell_signals = 0
        
        for i in range(1, len(vixfix)):
            if pd.isna(vixfix.iloc[i]):
                continue
            
            # Buy when VixFix is high and declining (fear peaking)
            if vixfix.iloc[i] > threshold and vixfix.iloc[i] < vixfix.iloc[i-1]:
                buy_signals += 1
            
            # Sell when VixFix is very low (complacency)
            elif vixfix.iloc[i] < 5:
                sell_signals += 1
        
        print(f"\nThreshold={threshold}: Buy={buy_signals}, Sell={sell_signals}")
    
    # Check how often VixFix exceeds different thresholds
    for threshold in [10, 15, 20, 25, 30]:
        count = (vixfix > threshold).sum()
        pct = count / len(valid_vix) * 100
        print(f"VixFix > {threshold}: {count} times ({pct:.1f}%)")


def test_strategies():
    """Test the actual strategy implementation"""
    print("\n\n=== TESTING STRATEGIES ===")
    
    calc = IndicatorCalculator(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]
    data = calc.load_raw_data(symbol, '1d')
    
    if data is None:
        print("ERROR: Could not load data")
        return
    
    # Test WaveTrend strategy
    print("\n--- WaveTrend Strategy ---")
    wt_result = calculate_wavetrend(data, 10, 21)
    wt1 = wt_result['wt1']
    wt2 = wt_result['wt2']
    
    # Try the exact logic from the optimizer
    overbought = 60
    oversold = -60
    signals = []
    
    for i in range(1, len(wt1)):
        if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
            signals.append(0)
            continue
        
        if wt1.iloc[i] < oversold and wt1.iloc[i] > wt1.iloc[i-1] and wt1.iloc[i] > wt2.iloc[i]:
            signals.append(1)
        elif wt1.iloc[i] > overbought and wt1.iloc[i] < wt1.iloc[i-1] and wt1.iloc[i] < wt2.iloc[i]:
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
        print(f"Total return: {total_return * 100:.2f}%")
        print(f"Number of trades: {sum(1 for s in signals if s != 0)}")
        print(f"Buy signals: {sum(1 for s in signals if s == 1)}")
        print(f"Sell signals: {sum(1 for s in signals if s == -1)}")
    else:
        print("No strategy returns generated!")
    
    # Test VixFix strategy
    print("\n--- VixFix Strategy ---")
    vix_result = calculate_vixfix(data, 22)
    vixfix = vix_result['vixfix']
    
    threshold = 20
    signals = []
    
    for i in range(1, len(vixfix)):
        if pd.isna(vixfix.iloc[i]):
            signals.append(0)
            continue
        
        if vixfix.iloc[i] > threshold and vixfix.iloc[i] < vixfix.iloc[i-1]:
            signals.append(1)
        elif vixfix.iloc[i] < 5:
            signals.append(-1)
        else:
            signals.append(0)
    
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
        print(f"Total return: {total_return * 100:.2f}%")
        print(f"Number of trades: {sum(1 for s in signals if s != 0)}")
        print(f"Buy signals: {sum(1 for s in signals if s == 1)}")
        print(f"Sell signals: {sum(1 for s in signals if s == -1)}")
    else:
        print("No strategy returns generated!")


if __name__ == "__main__":
    debug_wavetrend()
    debug_vixfix()
    test_strategies()