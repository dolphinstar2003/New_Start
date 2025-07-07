#!/usr/bin/env python3
"""
Optimize Squeeze Momentum and VixFix Universal Parameters
Continue from WaveTrend optimization
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
from indicators.squeeze_momentum import calculate_squeeze_momentum
from indicators.vixfix import calculate_vixfix
from loguru import logger

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SqueezeVixOptimizer:
    """Find universal parameters for Squeeze and VixFix"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        self.symbols = SACRED_SYMBOLS
        self.timeframe = '1d'
        self.n_trials = 100
        
        # Pre-load all data
        self.all_data = {}
        print("Loading data for all symbols...")
        for symbol in self.symbols:
            data = self.calc.load_raw_data(symbol, self.timeframe)
            if data is not None:
                self.all_data[symbol] = data
        
        print(f"Loaded data for {len(self.all_data)} symbols\n")
    
    def calculate_strategy_return(self, data, signals):
        """Calculate strategy return from signals"""
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
    
    def optimize_squeeze_universal(self):
        """Find Squeeze Momentum parameters that work for all symbols"""
        print("🎯 Optimizing Squeeze Momentum for all symbols...")
        
        def objective(trial):
            # Suggest Squeeze parameters
            bb_length = trial.suggest_int('sq_bb_length', 15, 25)
            bb_mult = trial.suggest_float('sq_bb_mult', 1.5, 2.5)
            kc_length = trial.suggest_int('sq_kc_length', 15, 25)
            kc_mult = trial.suggest_float('sq_kc_mult', 1.0, 2.0)
            mom_length = trial.suggest_int('sq_mom_length', 10, 20)
            
            # Test on all symbols
            returns = []
            
            for symbol, data in self.all_data.items():
                try:
                    # Calculate Squeeze Momentum
                    result = calculate_squeeze_momentum(
                        data, bb_length, bb_mult, kc_length, kc_mult, mom_length
                    )
                    if result is None or result.empty:
                        continue
                    
                    squeeze = result['squeeze']
                    momentum = result['momentum']
                    
                    # Generate signals - improved strategy
                    signals = []
                    squeeze_fired = False
                    
                    for i in range(1, len(squeeze)):
                        if pd.isna(squeeze.iloc[i]) or pd.isna(momentum.iloc[i]):
                            signals.append(0)
                            continue
                        
                        # Track squeeze firing
                        if squeeze.iloc[i] == 0 and squeeze.iloc[i-1] == 1:
                            squeeze_fired = True
                        
                        # Buy: Squeeze fires or positive momentum after squeeze
                        if (squeeze.iloc[i] == 0 and squeeze.iloc[i-1] == 1 and momentum.iloc[i] > 0) or \
                           (squeeze_fired and momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0):
                            signals.append(1)
                            squeeze_fired = False
                        # Sell: Momentum turns negative or decreases significantly
                        elif (momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0) or \
                             (momentum.iloc[i] < momentum.iloc[i-1] * 0.5):
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    if ret > 0:
                        returns.append(ret)
                    
                except Exception as e:
                    continue
            
            if len(returns) >= 10:  # Need at least 10 positive results
                avg_return = np.exp(np.mean(np.log1p(np.array(returns) / 100))) - 1
                return avg_return * 100
            return -1000
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\n✅ Squeeze Momentum Optimization complete!")
        print(f"Best universal parameters:")
        print(f"  BB Length: {study.best_params['sq_bb_length']}")
        print(f"  BB Mult: {study.best_params['sq_bb_mult']:.2f}")
        print(f"  KC Length: {study.best_params['sq_kc_length']}")
        print(f"  KC Mult: {study.best_params['sq_kc_mult']:.2f}")
        print(f"  Mom Length: {study.best_params['sq_mom_length']}")
        print(f"  Avg return: {study.best_value:.2f}%")
        
        self.test_parameters_on_all_symbols('Squeeze', study.best_params)
        
        return study.best_params, study.best_value
    
    def optimize_vixfix_universal(self):
        """Find VixFix parameters that work for all symbols"""
        print("\n🎯 Optimizing VixFix for all symbols...")
        
        def objective(trial):
            # Suggest VixFix parameters
            lookback = trial.suggest_int('vf_lookback', 15, 30)
            bb_length = trial.suggest_int('vf_bb_length', 15, 25)
            bb_mult = trial.suggest_float('vf_bb_mult', 1.5, 3.0)
            threshold = trial.suggest_float('vf_threshold', 0.5, 0.9)
            hold_days = trial.suggest_int('vf_hold_days', 3, 10)
            
            # Test on all symbols
            returns = []
            
            for symbol, data in self.all_data.items():
                try:
                    # Calculate VixFix
                    result = calculate_vixfix(data, lookback, bb_length, bb_mult)
                    if result is None or result.empty:
                        continue
                    
                    vixfix = result['vixfix']
                    bb_upper = result['bb_upper']
                    
                    # Generate signals - improved strategy
                    signals = []
                    buy_day = -999
                    
                    for i in range(1, len(vixfix)):
                        if pd.isna(vixfix.iloc[i]) or pd.isna(bb_upper.iloc[i]):
                            signals.append(0)
                            continue
                        
                        # Buy: VixFix spike above threshold
                        if vixfix.iloc[i] > bb_upper.iloc[i] * threshold and vixfix.iloc[i] > vixfix.iloc[i-1]:
                            signals.append(1)
                            buy_day = i
                        # Sell: After hold period or VixFix drops below mean
                        elif (i - buy_day >= hold_days) or \
                             (vixfix.iloc[i] < result['bb_middle'].iloc[i]):
                            signals.append(-1)
                            buy_day = -999
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    if ret > 0:
                        returns.append(ret)
                    
                except Exception as e:
                    continue
            
            if len(returns) >= 10:
                avg_return = np.exp(np.mean(np.log1p(np.array(returns) / 100))) - 1
                return avg_return * 100
            return -1000
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\n✅ VixFix Optimization complete!")
        print(f"Best universal parameters:")
        print(f"  Lookback: {study.best_params['vf_lookback']}")
        print(f"  BB Length: {study.best_params['vf_bb_length']}")
        print(f"  BB Mult: {study.best_params['vf_bb_mult']:.2f}")
        print(f"  Threshold: {study.best_params['vf_threshold']:.2f}")
        print(f"  Hold Days: {study.best_params['vf_hold_days']}")
        print(f"  Avg return: {study.best_value:.2f}%")
        
        self.test_parameters_on_all_symbols('VixFix', study.best_params)
        
        return study.best_params, study.best_value
    
    def test_parameters_on_all_symbols(self, indicator_name, params):
        """Test optimized parameters on each symbol"""
        print(f"\n📊 Testing {indicator_name} on each symbol:")
        print("-" * 60)
        
        results = []
        
        for symbol, data in self.all_data.items():
            try:
                if indicator_name == 'Squeeze':
                    result = calculate_squeeze_momentum(
                        data, params['sq_bb_length'], params['sq_bb_mult'],
                        params['sq_kc_length'], params['sq_kc_mult'], params['sq_mom_length']
                    )
                    if result is not None:
                        squeeze = result['squeeze']
                        momentum = result['momentum']
                        
                        signals = []
                        squeeze_fired = False
                        
                        for i in range(1, len(squeeze)):
                            if pd.isna(squeeze.iloc[i]):
                                signals.append(0)
                            elif squeeze.iloc[i] == 0 and squeeze.iloc[i-1] == 1:
                                squeeze_fired = True
                                if momentum.iloc[i] > 0:
                                    signals.append(1)
                                else:
                                    signals.append(0)
                            elif squeeze_fired and momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0:
                                signals.append(1)
                                squeeze_fired = False
                            elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
                                signals.append(-1)
                            else:
                                signals.append(0)
                
                elif indicator_name == 'VixFix':
                    result = calculate_vixfix(
                        data, params['vf_lookback'], params['vf_bb_length'], params['vf_bb_mult']
                    )
                    if result is not None:
                        vixfix = result['vixfix']
                        bb_upper = result['bb_upper']
                        bb_middle = result['bb_middle']
                        
                        signals = []
                        buy_day = -999
                        
                        for i in range(1, len(vixfix)):
                            if pd.isna(vixfix.iloc[i]):
                                signals.append(0)
                            elif vixfix.iloc[i] > bb_upper.iloc[i] * params['vf_threshold'] and vixfix.iloc[i] > vixfix.iloc[i-1]:
                                signals.append(1)
                                buy_day = i
                            elif (i - buy_day >= params['vf_hold_days']) or \
                                 (vixfix.iloc[i] < bb_middle.iloc[i]):
                                signals.append(-1)
                                buy_day = -999
                            else:
                                signals.append(0)
                
                # Calculate return
                ret = self.calculate_strategy_return(data, signals)
                buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                
                results.append({
                    'symbol': symbol,
                    'strategy_return': ret,
                    'buy_hold': buy_hold,
                    'outperformance': ret - buy_hold
                })
                
            except Exception as e:
                continue
        
        # Sort by return
        results.sort(key=lambda x: x['strategy_return'], reverse=True)
        
        # Display results
        print(f"{'Symbol':<8} {'Strategy':<12} {'Buy&Hold':<12} {'Outperform':<12}")
        print("-" * 60)
        
        positive_count = 0
        for r in results[:10]:  # Show top 10
            if r['strategy_return'] > 0:
                positive_count += 1
            print(f"{r['symbol']:<8} {r['strategy_return']:>10.1f}% {r['buy_hold']:>10.1f}% {r['outperformance']:>10.1f}%")
        
        # Calculate statistics
        all_returns = [r['strategy_return'] for r in results]
        avg_return = np.mean(all_returns)
        median_return = np.median(all_returns)
        
        print(f"\n📈 Statistics:")
        print(f"  Symbols with positive returns: {positive_count}/{len(results)}")
        print(f"  Average return: {avg_return:.1f}%")
        print(f"  Median return: {median_return:.1f}%")


def main():
    """Main function"""
    print("🔍 Finding Universal Optimal Parameters for Squeeze and VixFix")
    print("="*80)
    
    optimizer = SqueezeVixOptimizer()
    
    # Load existing parameters
    try:
        with open('universal_optimal_parameters.json', 'r') as f:
            universal_params = json.load(f)
    except:
        universal_params = {}
    
    # Add WaveTrend parameters from previous run
    universal_params["WaveTrend"] = {
        "params": {
            "wt_n1": 13,
            "wt_n2": 15,
            "wt_overbought": 70,
            "wt_oversold": -50
        },
        "avg_return": 147.50
    }
    
    # Optimize remaining indicators
    sq_params, sq_return = optimizer.optimize_squeeze_universal()
    vf_params, vf_return = optimizer.optimize_vixfix_universal()
    
    # Update universal parameters
    universal_params.update({
        "Squeeze": {
            "params": sq_params,
            "avg_return": sq_return
        },
        "VixFix": {
            "params": vf_params,
            "avg_return": vf_return
        },
        "optimization_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Save complete results
    with open('universal_optimal_parameters_complete.json', 'w') as f:
        json.dump(universal_params, f, indent=2)
    
    print("\n✅ All universal parameters saved to: universal_optimal_parameters_complete.json")
    
    # Display complete summary
    print("\n" + "="*80)
    print("📊 ALL 6 INDICATORS UNIVERSAL PARAMETERS SUMMARY")
    print("="*80)
    
    indicators_order = ['Supertrend', 'ADX', 'Squeeze', 'WaveTrend', 'MACD', 'VixFix']
    
    for ind in indicators_order:
        if ind in universal_params:
            print(f"\n{ind}:")
            params = universal_params[ind]['params']
            
            # Format parameters nicely
            if ind == 'MACD':
                print(f"  Fast={params['macd_fast']}, Slow={params['macd_slow']}, Signal={params['macd_signal']}")
            elif ind == 'ADX':
                print(f"  Period={params['adx_period']}, Threshold={params['adx_threshold']:.2f}, Exit={params['adx_exit_threshold']:.2f}")
            elif ind == 'Supertrend':
                print(f"  Period={params['st_period']}, Multiplier={params['st_multiplier']:.2f}")
            elif ind == 'WaveTrend':
                print(f"  N1={params['wt_n1']}, N2={params['wt_n2']}, OB={params['wt_overbought']}, OS={params['wt_oversold']}")
            elif ind == 'Squeeze':
                print(f"  BB={params['sq_bb_length']}/{params['sq_bb_mult']:.1f}, KC={params['sq_kc_length']}/{params['sq_kc_mult']:.1f}, Mom={params['sq_mom_length']}")
            elif ind == 'VixFix':
                print(f"  Lookback={params['vf_lookback']}, BB={params['vf_bb_length']}/{params['vf_bb_mult']:.1f}, Threshold={params['vf_threshold']:.1f}, Hold={params['vf_hold_days']}d")
            
            print(f"  Avg Return: {universal_params[ind]['avg_return']:.1f}%")
    
    # Performance ranking
    print("\n📈 PERFORMANCE RANKING:")
    print("-" * 40)
    sorted_indicators = sorted(
        [(ind, universal_params[ind]['avg_return']) for ind in universal_params if ind != 'optimization_date'],
        key=lambda x: x[1],
        reverse=True
    )
    
    for i, (ind, ret) in enumerate(sorted_indicators):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"  {i+1}. {ind:<12} {ret:>8.1f}% {medal}")


if __name__ == "__main__":
    main()