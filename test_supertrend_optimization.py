#!/usr/bin/env python3
"""
Test Supertrend Optimization with Optuna
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.supertrend import calculate_supertrend


def test_supertrend_optimization():
    """Test Supertrend optimization with GARAN"""
    print("🧪 Testing Supertrend Optimization...")
    
    # Initialize
    calc = IndicatorCalculator(DATA_DIR)
    symbol = 'GARAN'
    
    # Load data
    data = calc.load_raw_data(symbol, '1d')
    print(f"✅ Data loaded: {data.shape}")
    
    # Test Supertrend calculation
    st_result = calculate_supertrend(data, 10, 3.0)
    print(f"✅ Supertrend calculated: {st_result.shape if st_result is not None else 'None'}")
    
    if st_result is not None:
        print(f"Supertrend columns: {st_result.columns.tolist()}")
        print("Supertrend sample:")
        print(st_result.tail())
        
        # Optimization objective
        def objective(trial):
            # Suggest Supertrend parameters
            period = trial.suggest_int('st_period', 5, 20)
            multiplier = trial.suggest_float('st_multiplier', 1.0, 5.0)
            
            try:
                # Calculate Supertrend
                result = calculate_supertrend(data, period, multiplier)
                if result is None or result.empty:
                    return -100
                
                st_trend = result.get('supertrend', pd.Series())
                st_direction = result.get('trend', pd.Series())  # 'trend' not 'direction'
                
                if len(st_direction) == 0:
                    return -100
                
                # Supertrend strategy
                signals = []
                for i in range(1, min(len(st_direction), 500)):  # Test with more data
                    if pd.isna(st_direction.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy when trend changes to up
                    if st_direction.iloc[i] == 1 and st_direction.iloc[i-1] == -1:
                        signals.append(1)
                    # Sell when trend changes to down
                    elif st_direction.iloc[i] == -1 and st_direction.iloc[i-1] == 1:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate return
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
        
        # Run optimization
        print("\n🎯 Running optimization trials...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=111, show_progress_bar=True)
        
        print(f"\n📊 Best result:")
        print(f"   Score: {study.best_value:.2f}%")
        print(f"   Parameters: {study.best_params}")
        
        # Show top 5 trials
        print(f"\n🏆 Top 5 trials:")
        trials_df = study.trials_dataframe()
        top_trials = trials_df.nlargest(5, 'value')[['value', 'params_st_period', 'params_st_multiplier']]
        for idx, row in top_trials.iterrows():
            print(f"   {idx+1}. Return: {row['value']:.2f}%, Period: {row['params_st_period']}, Multiplier: {row['params_st_multiplier']:.2f}")
        
        return True
    else:
        print("❌ Supertrend calculation failed")
        return False


if __name__ == "__main__":
    test_supertrend_optimization()