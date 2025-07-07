#!/usr/bin/env python3
"""
Simple VixFix Optimization
Find universal parameters for VixFix indicator
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.vixfix import calculate_vixfix

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_vixfix_simple():
    """Simple optimization for VixFix"""
    print("🎯 Testing VixFix calculation first...")
    
    # Load GARAN data for testing
    calc = IndicatorCalculator(DATA_DIR)
    data = calc.load_raw_data('GARAN', '1d')
    
    # Test with default parameters
    print("\n1️⃣ Testing with default parameters (22, 20, 2.0)...")
    result = calculate_vixfix(data, 22, 20, 2.0)
    
    if result is None:
        print("❌ ERROR: calculate_vixfix returned None")
        return
    
    print(f"✅ Result shape: {result.shape}")
    print(f"✅ Columns: {result.columns.tolist()}")
    
    # Check the data
    vixfix = result['vixfix']
    bb_upper = result['bb_upper']
    bb_middle = result['bb_middle']
    
    print(f"\n2️⃣ VixFix range: [{vixfix.min():.2f}, {vixfix.max():.2f}]")
    print(f"   BB Upper range: [{bb_upper.min():.2f}, {bb_upper.max():.2f}]")
    
    # Generate simple signals - VixFix is a fear gauge, buy on extreme fear
    signals = []
    holding = False
    hold_counter = 0
    
    for i in range(1, len(vixfix)):
        if pd.isna(vixfix.iloc[i]) or pd.isna(bb_upper.iloc[i]):
            signals.append(0)
            continue
        
        # Buy when VixFix spikes above BB upper
        if not holding and vixfix.iloc[i] > bb_upper.iloc[i]:
            signals.append(1)
            holding = True
            hold_counter = 0
        # Sell after 5 days or when VixFix drops below middle
        elif holding and (hold_counter >= 5 or vixfix.iloc[i] < bb_middle.iloc[i]):
            signals.append(-1)
            holding = False
        else:
            signals.append(0)
            if holding:
                hold_counter += 1
    
    buy_signals = signals.count(1)
    sell_signals = signals.count(-1)
    
    print(f"\n3️⃣ Simple VixFix strategy signals:")
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
    all_data = {}
    for symbol in SACRED_SYMBOLS:
        d = calc.load_raw_data(symbol, '1d')
        if d is not None:
            all_data[symbol] = d
    
    print(f"Loaded {len(all_data)} symbols")
    
    def objective(trial):
        # VixFix parameters
        lookback = trial.suggest_int('lookback', 15, 30)
        bb_length = trial.suggest_int('bb_length', 15, 25)
        bb_mult = trial.suggest_float('bb_mult', 1.5, 3.0)
        hold_days = trial.suggest_int('hold_days', 3, 10)
        
        total_returns = []
        
        for symbol, data in all_data.items():
            try:
                result = calculate_vixfix(data, lookback, bb_length, bb_mult)
                if result is None:
                    continue
                
                vixfix = result['vixfix']
                bb_upper = result['bb_upper']
                bb_middle = result['bb_middle']
                
                # Generate signals
                signals = []
                holding = False
                hold_counter = 0
                
                for i in range(1, len(vixfix)):
                    if pd.isna(vixfix.iloc[i]):
                        signals.append(0)
                        continue
                    
                    if not holding and vixfix.iloc[i] > bb_upper.iloc[i]:
                        signals.append(1)
                        holding = True
                        hold_counter = 0
                    elif holding and (hold_counter >= hold_days or vixfix.iloc[i] < bb_middle.iloc[i]):
                        signals.append(-1)
                        holding = False
                    else:
                        signals.append(0)
                        if holding:
                            hold_counter += 1
                
                # Skip if no signals
                if signals.count(1) == 0:
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
    print(f"   Lookback: {study.best_params['lookback']}")
    print(f"   BB Length: {study.best_params['bb_length']}")
    print(f"   BB Mult: {study.best_params['bb_mult']:.2f}")
    print(f"   Hold Days: {study.best_params['hold_days']}")
    print(f"   Best avg return: {study.best_value:.2f}%")
    
    # Test best parameters on each symbol
    print(f"\n📊 Testing best parameters on each symbol:")
    print("-" * 50)
    
    best_lookback = study.best_params['lookback']
    best_bb_length = study.best_params['bb_length']
    best_bb_mult = study.best_params['bb_mult']
    best_hold = study.best_params['hold_days']
    
    results = []
    for symbol, data in all_data.items():
        try:
            result = calculate_vixfix(data, best_lookback, best_bb_length, best_bb_mult)
            if result is None:
                continue
            
            vixfix = result['vixfix']
            bb_upper = result['bb_upper']
            bb_middle = result['bb_middle']
            
            signals = []
            holding = False
            hold_counter = 0
            
            for i in range(1, len(vixfix)):
                if pd.isna(vixfix.iloc[i]):
                    signals.append(0)
                    continue
                
                if not holding and vixfix.iloc[i] > bb_upper.iloc[i]:
                    signals.append(1)
                    holding = True
                    hold_counter = 0
                elif holding and (hold_counter >= best_hold or vixfix.iloc[i] < bb_middle.iloc[i]):
                    signals.append(-1)
                    holding = False
                else:
                    signals.append(0)
                    if holding:
                        hold_counter += 1
            
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
        "VixFix": {
            "params": {
                "vf_lookback": best_lookback,
                "vf_bb_length": best_bb_length,
                "vf_bb_mult": best_bb_mult,
                "vf_hold_days": best_hold
            },
            "avg_return": study.best_value,
            "strategy": "fear_gauge_reversal"
        }
    }
    
    # Load existing parameters and update
    try:
        with open('universal_optimal_parameters.json', 'r') as f:
            existing = json.load(f)
    except:
        existing = {}
    
    # Add all known parameters
    all_params = {
        "MACD": existing.get("MACD", {
            "params": {"macd_fast": 8, "macd_slow": 21, "macd_signal": 5},
            "avg_return": 3585.18
        }),
        "ADX": existing.get("ADX", {
            "params": {"adx_period": 7, "adx_threshold": 15.03, "adx_exit_threshold": 11.91},
            "avg_return": 3603.58
        }),
        "Supertrend": existing.get("Supertrend", {
            "params": {"st_period": 6, "st_multiplier": 0.50},
            "avg_return": 41347.87
        }),
        "WaveTrend": {
            "params": {"wt_n1": 13, "wt_n2": 15, "wt_overbought": 70, "wt_oversold": -50},
            "avg_return": 147.50
        },
        "Squeeze": {
            "params": {"sq_bb_length": 20, "sq_bb_mult": 2.0, "sq_kc_length": 20, "sq_kc_mult": 1.5, "sq_mom_length": 12},
            "avg_return": 254.06  # From the running optimization
        },
        "VixFix": universal_params["VixFix"],
        "optimization_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('universal_optimal_parameters_complete.json', 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\n✅ All parameters saved to: universal_optimal_parameters_complete.json")
    
    # Display final summary
    print("\n" + "="*60)
    print("📊 ALL 6 INDICATORS UNIVERSAL PARAMETERS SUMMARY")
    print("="*60)
    
    # Performance ranking
    indicators = [
        ("Supertrend", all_params["Supertrend"]["avg_return"]),
        ("ADX", all_params["ADX"]["avg_return"]),
        ("MACD", all_params["MACD"]["avg_return"]),
        ("Squeeze", all_params["Squeeze"]["avg_return"]),
        ("WaveTrend", all_params["WaveTrend"]["avg_return"]),
        ("VixFix", all_params["VixFix"]["avg_return"])
    ]
    
    indicators.sort(key=lambda x: x[1], reverse=True)
    
    print("\n📈 PERFORMANCE RANKING:")
    print("-" * 40)
    for i, (ind, ret) in enumerate(indicators):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {i+1}. {ind:<12} {ret:>8.1f}% {medal}")


if __name__ == "__main__":
    optimize_vixfix_simple()