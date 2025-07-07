#!/usr/bin/env python3
"""
Test Single Optimization: GARAN + MACD + 1d
Debug why we're getting 0% returns
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom


def test_garan_macd_1d():
    """Test GARAN MACD optimization on 1d timeframe"""
    print("🧪 Testing GARAN + MACD + 1d Optimization")
    print("="*60)
    
    # Load data
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    
    print(f"\n1️⃣ Data loaded:")
    print(f"   Shape: {data.shape}")
    print(f"   Date range: {data.index[0]} to {data.index[-1]}")
    print(f"   Columns: {data.columns.tolist()}")
    
    # Check data quality
    print(f"\n2️⃣ Data quality:")
    print(f"   NaN values: {data.isnull().sum().sum()}")
    print(f"   Close price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    # Test MACD calculation
    print(f"\n3️⃣ Testing MACD calculation (12, 26, 9):")
    macd_result = calculate_macd_custom(data, 12, 26, 9)
    
    if macd_result is None:
        print("   ❌ MACD calculation failed!")
        return
    
    print(f"   ✅ MACD calculated")
    print(f"   Result shape: {macd_result.shape}")
    print(f"   MACD columns: {macd_result.columns.tolist()}")
    
    # Check MACD values
    macd_line = macd_result['macd']
    signal_line = macd_result['signal']
    
    print(f"\n4️⃣ MACD values check:")
    print(f"   Valid MACD values: {(~macd_line.isna()).sum()}")
    print(f"   Valid Signal values: {(~signal_line.isna()).sum()}")
    print(f"   MACD range: {macd_line.min():.4f} to {macd_line.max():.4f}")
    
    # Test signal generation
    print(f"\n5️⃣ Testing signal generation:")
    signals = []
    for i in range(1, len(macd_line)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            signals.append(0)
            continue
        
        # Buy signal: MACD crosses above signal
        if macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
            signals.append(1)
        # Sell signal: MACD crosses below signal
        elif macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
            signals.append(-1)
        else:
            signals.append(0)
    
    buy_signals = signals.count(1)
    sell_signals = signals.count(-1)
    
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    print(f"   Total signals: {buy_signals + sell_signals}")
    
    # Test returns calculation
    print(f"\n6️⃣ Testing returns calculation:")
    returns = data['close'].pct_change().fillna(0)
    print(f"   Daily returns calculated: {len(returns)}")
    print(f"   Returns range: {returns.min():.4f} to {returns.max():.4f}")
    print(f"   Mean daily return: {returns.mean():.4f}")
    
    # Test strategy
    print(f"\n7️⃣ Testing strategy performance:")
    position = 0
    strategy_returns = []
    
    for i, signal in enumerate(signals):
        if signal == 1:  # Buy
            position = 1
        elif signal == -1:  # Sell
            position = 0
        
        # Apply return for next period
        if i+1 < len(returns):
            strategy_returns.append(position * returns.iloc[i+1])
    
    print(f"   Strategy periods: {len(strategy_returns)}")
    print(f"   Periods in market: {sum(1 for r in strategy_returns if r != 0)}")
    print(f"   Periods out of market: {sum(1 for r in strategy_returns if r == 0)}")
    
    # Calculate total return
    if strategy_returns:
        total_return = (1 + pd.Series(strategy_returns)).prod() - 1
        print(f"\n8️⃣ Strategy Performance:")
        print(f"   Total return: {total_return * 100:.2f}%")
        print(f"   Buy & Hold return: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
    else:
        print(f"\n   ❌ No strategy returns calculated!")
    
    # Show sample signals
    print(f"\n9️⃣ Sample signals (first 10 crossovers):")
    signal_count = 0
    for i in range(1, len(macd_line)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        
        if (macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]) or \
           (macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]):
            signal_type = "BUY" if macd_line.iloc[i] > signal_line.iloc[i] else "SELL"
            print(f"   {data.index[i]}: {signal_type} - MACD={macd_line.iloc[i]:.4f}, Signal={signal_line.iloc[i]:.4f}")
            signal_count += 1
            if signal_count >= 10:
                break


if __name__ == "__main__":
    test_garan_macd_1d()