#!/usr/bin/env python3
"""
Simple Squeeze Momentum Optimization
Fix signal generation issues
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.squeeze_momentum import calculate_squeeze_momentum

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_squeeze_simple():
    """Simple optimization for Squeeze Momentum"""
    print("🎯 Testing Squeeze Momentum calculation first...")
    
    # Load GARAN data for testing
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    
    # Test with default parameters
    print("\n1️⃣ Testing with default parameters (20, 2.0, 20, 1.5, 12)...")
    result = calculate_squeeze_momentum(data, 20, 2.0, 20, 1.5, 12)
    
    if result is None:
        print("❌ ERROR: calculate_squeeze_momentum returned None")
        return
    
    print(f"✅ Result shape: {result.shape}")
    print(f"✅ Columns: {result.columns.tolist()}")
    
    # Check the data
    squeeze_on = result['squeeze_on']
    momentum = result['momentum']
    
    print(f"\n2️⃣ Squeeze ON values: {squeeze_on.value_counts().to_dict()}")
    print(f"   Momentum range: [{momentum.min():.2f}, {momentum.max():.2f}]")
    
    # Generate simple signals
    signals = []
    for i in range(1, len(squeeze_on)):
        if pd.isna(squeeze_on.iloc[i]) or pd.isna(momentum.iloc[i]):
            signals.append(0)
            continue
        
        # Simple strategy: Buy when momentum positive, sell when negative
        if momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0:
            signals.append(1)
        elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
            signals.append(-1)
        else:
            signals.append(0)
    
    buy_signals = signals.count(1)
    sell_signals = signals.count(-1)
    
    print(f"\n3️⃣ Simple momentum strategy signals:")
    print(f"   Buy signals: {buy_signals}")
    print(f"   Sell signals: {sell_signals}")
    
    # Calculate returns
    returns = data['close'].pct_change().fillna(0)
    position = 0
    strategy_returns = []
    
    for i in range(len(signals)):
        if signals[i] == 1:
            position = 1
        elif signals[i] == -1:
            position = 0
        
        if i+1 < len(returns):
            strategy_returns.append(position * returns.iloc[i+1])
    
    if strategy_returns:
        total_return = (1 + pd.Series(strategy_returns)).prod() - 1
        print(f"\n4️⃣ Strategy return: {total_return * 100:.2f}%")
        
        buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
        print(f"   Buy & Hold return: {buy_hold:.2f}%")
    
    # Now optimize
    print("\n" + "="*60)
    print("5️⃣ Starting optimization for all symbols...")
    
    # Load all symbols
    symbols = ['GARAN', 'AKBNK', 'ISCTR', 'YKBNK', 'SAHOL', 'KCHOL', 'SISE', 
               'EREGL', 'KRDMD', 'TUPRS', 'ASELS', 'THYAO', 'TCELL', 'BIMAS', 
               'MGROS', 'ULKER', 'AKSEN', 'ENKAI', 'PETKM', 'KOZAL']
    
    all_data = {}
    for symbol in symbols:
        d = calc.load_raw_data(symbol, '1d')
        if d is not None:
            all_data[symbol] = d
    
    print(f"Loaded {len(all_data)} symbols")
    
    def objective(trial):
        # Simplified parameters
        bb_length = trial.suggest_int('bb_length', 15, 25)
        bb_mult = 2.0  # Fixed
        kc_length = trial.suggest_int('kc_length', 15, 25)
        kc_mult = 1.5  # Fixed
        mom_length = trial.suggest_int('mom_length', 10, 15)
        
        total_returns = []
        
        for symbol, data in all_data.items():
            try:
                result = calculate_squeeze_momentum(data, bb_length, bb_mult, kc_length, kc_mult, mom_length)
                if result is None:
                    continue
                
                momentum = result['momentum']
                
                # Simple momentum crossover strategy
                signals = []
                for i in range(1, len(momentum)):
                    if pd.isna(momentum.iloc[i]):
                        signals.append(0)
                    elif momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0:
                        signals.append(1)
                    elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Skip if no signals
                if signals.count(1) == 0 or signals.count(-1) == 0:
                    continue
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i in range(len(signals)):
                    if signals[i] == 1:
                        position = 1
                    elif signals[i] == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    if total_return > 0:
                        total_returns.append(total_return * 100)
                
            except Exception as e:
                continue
        
        if len(total_returns) >= 10:
            # Use geometric mean
            avg_return = np.exp(np.mean(np.log1p(np.array(total_returns) / 100))) - 1
            return avg_return * 100
        return -100
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print(f"\n✅ Optimization complete!")
    print(f"\n🏆 Best parameters:")
    print(f"   BB Length: {study.best_params['bb_length']}")
    print(f"   KC Length: {study.best_params['kc_length']}")
    print(f"   Mom Length: {study.best_params['mom_length']}")
    print(f"   Best avg return: {study.best_value:.2f}%")
    
    # Test best parameters on each symbol
    print(f"\n📊 Testing best parameters on each symbol:")
    print("-" * 50)
    
    best_bb = study.best_params['bb_length']
    best_kc = study.best_params['kc_length']
    best_mom = study.best_params['mom_length']
    
    results = []
    for symbol, data in all_data.items():
        try:
            result = calculate_squeeze_momentum(data, best_bb, 2.0, best_kc, 1.5, best_mom)
            if result is None:
                continue
            
            momentum = result['momentum']
            
            signals = []
            for i in range(1, len(momentum)):
                if pd.isna(momentum.iloc[i]):
                    signals.append(0)
                elif momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0:
                    signals.append(1)
                elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
                    signals.append(-1)
                else:
                    signals.append(0)
            
            # Calculate returns
            returns = data['close'].pct_change().fillna(0)
            position = 0
            strategy_returns = []
            
            for i in range(len(signals)):
                if signals[i] == 1:
                    position = 1
                elif signals[i] == -1:
                    position = 0
                
                if i+1 < len(returns):
                    strategy_returns.append(position * returns.iloc[i+1])
            
            if strategy_returns:
                total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                
                results.append({
                    'symbol': symbol,
                    'return': total_return * 100,
                    'buy_hold': buy_hold,
                    'signals': f"{signals.count(1)}B/{signals.count(-1)}S"
                })
        
        except Exception as e:
            continue
    
    # Sort and display
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print(f"{'Symbol':<8} {'Return':<12} {'B&H':<12} {'Signals':<10}")
    print("-" * 50)
    
    for r in results[:10]:
        print(f"{r['symbol']:<8} {r['return']:>10.1f}% {r['buy_hold']:>10.1f}% {r['signals']:>10}")
    
    # Save results
    universal_params = {
        "Squeeze": {
            "params": {
                "sq_bb_length": best_bb,
                "sq_bb_mult": 2.0,
                "sq_kc_length": best_kc,
                "sq_kc_mult": 1.5,
                "sq_mom_length": best_mom
            },
            "avg_return": study.best_value,
            "strategy": "momentum_crossover"
        }
    }
    
    with open('squeeze_optimal_parameters.json', 'w') as f:
        json.dump(universal_params, f, indent=2)
    
    print(f"\n✅ Parameters saved to: squeeze_optimal_parameters.json")


if __name__ == "__main__":
    optimize_squeeze_simple()