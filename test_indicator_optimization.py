#!/usr/bin/env python3
"""
Test Indicator Optimization - Simple MACD Test
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom


def test_macd_optimization():
    """Test MACD optimization with one symbol"""
    print("🧪 Testing MACD Optimization...")
    
    # Initialize
    calc = IndicatorCalculator(DATA_DIR)
    symbol = 'GARAN'
    
    # Load data
    data = calc.load_raw_data(symbol, '1d')
    print(f"✅ Data loaded: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Test MACD calculation
    macd_result = calculate_macd_custom(data, 12, 26, 9)
    print(f"✅ MACD calculated: {macd_result.shape if macd_result is not None else 'None'}")
    
    if macd_result is not None:
        print(f"MACD columns: {macd_result.columns.tolist()}")
        print("MACD sample:")
        print(macd_result.tail())
        
        # Test optimization objective
        def objective(trial):
            fast_period = trial.suggest_int('macd_fast', 10, 15)
            slow_period = trial.suggest_int('macd_slow', 20, 26)
            signal_period = trial.suggest_int('macd_signal', 8, 10)
            
            if fast_period >= slow_period:
                return -100
            
            try:
                # Calculate MACD
                result = calculate_macd_custom(data, fast_period, slow_period, signal_period)
                if result is None or result.empty:
                    return -100
                
                macd_line = result.get('macd', pd.Series())
                macd_signal = result.get('signal', pd.Series())
                
                if len(macd_line) == 0 or len(macd_signal) == 0:
                    return -100
                
                # Simple strategy simulation
                signals = []
                for i in range(1, min(len(macd_line), 100)):  # Test with 100 periods
                    if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy signal: MACD crosses above signal
                    if (macd_line.iloc[i] > macd_signal.iloc[i] and 
                        macd_line.iloc[i-1] <= macd_signal.iloc[i-1]):
                        signals.append(1)
                    # Sell signal: MACD crosses below signal  
                    elif (macd_line.iloc[i] < macd_signal.iloc[i] and 
                          macd_line.iloc[i-1] >= macd_signal.iloc[i-1]):
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate simple return
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = []
                position = 0
                
                for i, signal in enumerate(signals):
                    if signal == 1:  # Buy
                        position = 1
                    elif signal == -1:  # Sell
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                    else:
                        strategy_returns.append(0)
                
                # Calculate total return
                if len(strategy_returns) > 0:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100  # Return as percentage
                else:
                    return -100
                
            except Exception as e:
                print(f"   Error in trial: {e}")
                return -100
        
        # Run more trials
        print("\n🎯 Running optimization trials...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=111, show_progress_bar=True)
        
        print(f"\n📊 Best result:")
        print(f"   Score: {study.best_value:.2f}%")
        print(f"   Parameters: {study.best_params}")
        
        return True
    else:
        print("❌ MACD calculation failed")
        return False


if __name__ == "__main__":
    test_macd_optimization()