#!/usr/bin/env python3
"""
Optimize ADX for GARAN
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.adx_di import calculate_adx_di

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_adx_garan():
    """Optimize ADX parameters for GARAN"""
    print("🎯 Optimizing ADX for GARAN (1d)")
    print("="*60)
    
    # Load data
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    
    print(f"Data loaded: {data.shape}")
    
    def objective(trial):
        # Suggest ADX parameters
        period = trial.suggest_int('adx_period', 7, 21)
        threshold = trial.suggest_float('adx_threshold', 15, 40)
        exit_threshold = trial.suggest_float('adx_exit_threshold', 10, 30)
        
        if exit_threshold >= threshold:
            return -100
        
        try:
            # Calculate ADX
            result = calculate_adx_di(data, length=period, threshold=threshold)
            if result is None or result.empty:
                return -100
            
            adx = result['adx']
            plus_di = result['plus_di']
            minus_di = result['minus_di']
            
            # Generate signals with DI cross strategy
            signals = []
            for i in range(1, len(adx)):
                if pd.isna(adx.iloc[i]) or pd.isna(plus_di.iloc[i]) or pd.isna(minus_di.iloc[i]):
                    signals.append(0)
                    continue
                
                # Buy signal: +DI crosses above -DI and ADX > threshold
                if (plus_di.iloc[i] > minus_di.iloc[i] and plus_di.iloc[i-1] <= minus_di.iloc[i-1] 
                    and adx.iloc[i] > threshold):
                    signals.append(1)
                # Sell signal: -DI crosses above +DI or ADX drops below exit threshold
                elif (minus_di.iloc[i] > plus_di.iloc[i] and minus_di.iloc[i-1] <= plus_di.iloc[i-1]) or \
                     (adx.iloc[i] < exit_threshold):
                    signals.append(-1)
                else:
                    signals.append(0)
            
            # Calculate returns
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
    print(f"   ADX Period: {study.best_params['adx_period']}")
    print(f"   ADX Threshold: {study.best_params['adx_threshold']:.2f}")
    print(f"   Exit Threshold: {study.best_params['adx_exit_threshold']:.2f}")
    print(f"   Best return: {study.best_value:.2f}%")
    
    # Test with default parameters
    default_result = objective(optuna.trial.FixedTrial({
        'adx_period': 14,
        'adx_threshold': 25.0,
        'adx_exit_threshold': 20.0
    }))
    print(f"\n📊 Default parameters (14, 25, 20) return: {default_result:.2f}%")
    
    # Show top 5 trials
    print(f"\n🔝 Top 5 parameter combinations:")
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nlargest(5, 'value')[['value', 'params_adx_period', 'params_adx_threshold', 'params_adx_exit_threshold']]
    for idx, row in top_trials.iterrows():
        print(f"   {idx+1}. Return: {row['value']:.2f}%, Period: {int(row['params_adx_period'])}, "
              f"Threshold: {row['params_adx_threshold']:.2f}, Exit: {row['params_adx_exit_threshold']:.2f}")
    
    return study.best_params, study.best_value


if __name__ == "__main__":
    best_params, best_return = optimize_adx_garan()