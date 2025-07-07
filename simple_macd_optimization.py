#!/usr/bin/env python3
"""
Simple MACD Optimization for GARAN
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_macd_garan():
    """Optimize MACD parameters for GARAN"""
    print("🎯 Optimizing MACD for GARAN (1d)")
    print("="*60)
    
    # Load data
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    
    print(f"Data loaded: {data.shape}")
    
    def objective(trial):
        # Suggest MACD parameters
        fast = trial.suggest_int('macd_fast', 8, 20)
        slow = trial.suggest_int('macd_slow', 21, 35)
        signal = trial.suggest_int('macd_signal', 5, 15)
        
        if fast >= slow:
            return -100
        
        try:
            # Calculate MACD
            result = calculate_macd_custom(data, fast, slow, signal)
            if result is None or result.empty:
                return -100
            
            macd_line = result['macd']
            macd_signal = result['signal']
            
            # Generate signals (fixed logic)
            signals = []
            for i in range(1, len(macd_line)):
                if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                    signals.append(0)
                    continue
                
                # Buy signal: MACD crosses above signal
                if macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                    signals.append(1)
                # Sell signal: MACD crosses below signal
                elif macd_line.iloc[i] < macd_signal.iloc[i] and macd_line.iloc[i-1] >= macd_signal.iloc[i-1]:
                    signals.append(-1)
                else:
                    signals.append(0)
            
            # Calculate returns (fixed indexing)
            returns = data['close'].pct_change().fillna(0)
            position = 0
            strategy_returns = []
            
            # Start from where we have signals
            for i in range(len(signals)):
                if signals[i] == 1:  # Buy
                    position = 1
                elif signals[i] == -1:  # Sell
                    position = 0
                
                # Apply return for next period
                if i+1 < len(returns):
                    strategy_returns.append(position * returns.iloc[i+1])
            
            # Calculate total return
            if strategy_returns:
                total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                return total_return * 100
            else:
                return -100
                
        except Exception as e:
            return -100
    
    # Create study and optimize
    study = optuna.create_study(direction='maximize')
    
    print("\nRunning optimization...")
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    
    print(f"\n✅ Optimization complete!")
    print(f"\n🏆 Best parameters:")
    print(f"   MACD Fast: {study.best_params['macd_fast']}")
    print(f"   MACD Slow: {study.best_params['macd_slow']}")
    print(f"   MACD Signal: {study.best_params['macd_signal']}")
    print(f"   Best return: {study.best_value:.2f}%")
    
    # Test with default parameters
    default_result = objective(optuna.trial.FixedTrial({
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    }))
    print(f"\n📊 Default parameters (12,26,9) return: {default_result:.2f}%")
    
    # Show top 5 trials
    print(f"\n🔝 Top 5 parameter combinations:")
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nlargest(5, 'value')[['value', 'params_macd_fast', 'params_macd_slow', 'params_macd_signal']]
    for idx, row in top_trials.iterrows():
        print(f"   {idx+1}. Return: {row['value']:.2f}%, Fast: {int(row['params_macd_fast'])}, "
              f"Slow: {int(row['params_macd_slow'])}, Signal: {int(row['params_macd_signal'])}")
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    best_params, best_return = optimize_macd_garan()