#!/usr/bin/env python3
"""
Optimize Remaining Universal Parameters
Find optimal parameters for WaveTrend, Squeeze Momentum, and VixFix
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
from indicators.wavetrend import calculate_wavetrend
from indicators.squeeze_momentum import calculate_squeeze_momentum
from indicators.vixfix import calculate_vixfix
from loguru import logger

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class RemainingOptimizer:
    """Find universal parameters for remaining indicators"""
    
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
                print(f"  ✓ {symbol}: {len(data)} bars")
            else:
                print(f"  ✗ {symbol}: No data")
        
        print(f"\nLoaded data for {len(self.all_data)} symbols")
    
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
    
    def optimize_wavetrend_universal(self):
        """Find WaveTrend parameters that work for all symbols"""
        print("\n🎯 Optimizing WaveTrend for all symbols...")
        
        def objective(trial):
            # Suggest WaveTrend parameters
            n1 = trial.suggest_int('wt_n1', 5, 15)
            n2 = trial.suggest_int('wt_n2', 15, 30)
            overbought = trial.suggest_int('wt_overbought', 50, 70)
            oversold = trial.suggest_int('wt_oversold', -70, -50)
            
            if n1 >= n2:
                return -1000
            
            # Test on all symbols
            returns = []
            
            for symbol, data in self.all_data.items():
                try:
                    # Calculate WaveTrend
                    result = calculate_wavetrend(data, n1, n2)
                    if result is None or result.empty:
                        continue
                    
                    wt1 = result['wt1']
                    wt2 = result['wt2']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(wt1)):
                        if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                            signals.append(0)
                            continue
                        
                        # Buy: WT1 crosses above WT2 in oversold
                        if (wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i-1] <= wt2.iloc[i-1] 
                            and wt1.iloc[i] < oversold):
                            signals.append(1)
                        # Sell: WT1 crosses below WT2 in overbought
                        elif (wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[i-1] >= wt2.iloc[i-1] 
                              and wt1.iloc[i] > overbought):
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    if ret > 0:
                        returns.append(ret)
                    
                except Exception as e:
                    continue
            
            if returns:
                avg_return = np.exp(np.mean(np.log1p(np.array(returns) / 100))) - 1
                return avg_return * 100
            return -1000
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\n✅ WaveTrend Optimization complete!")
        print(f"Best universal parameters:")
        print(f"  N1: {study.best_params['wt_n1']}")
        print(f"  N2: {study.best_params['wt_n2']}")
        print(f"  Overbought: {study.best_params['wt_overbought']}")
        print(f"  Oversold: {study.best_params['wt_oversold']}")
        print(f"  Avg return: {study.best_value:.2f}%")
        
        self.test_parameters_on_all_symbols('WaveTrend', study.best_params)
        
        return study.best_params, study.best_value
    
    def optimize_squeeze_universal(self):
        """Find Squeeze Momentum parameters that work for all symbols"""
        print("\n🎯 Optimizing Squeeze Momentum for all symbols...")
        
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
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(squeeze)):
                        if pd.isna(squeeze.iloc[i]) or pd.isna(momentum.iloc[i]):
                            signals.append(0)
                            continue
                        
                        # Buy: Squeeze fires (transitions from 1 to 0) with positive momentum
                        if (squeeze.iloc[i] == 0 and squeeze.iloc[i-1] == 1 
                            and momentum.iloc[i] > 0):
                            signals.append(1)
                        # Sell: Momentum turns negative
                        elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    if ret > 0:
                        returns.append(ret)
                    
                except Exception as e:
                    continue
            
            if returns:
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
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(vixfix)):
                        if pd.isna(vixfix.iloc[i]) or pd.isna(bb_upper.iloc[i]):
                            signals.append(0)
                            continue
                        
                        # Buy: VixFix exceeds threshold of BB upper
                        if vixfix.iloc[i] > bb_upper.iloc[i] * threshold:
                            signals.append(1)
                        # Sell: After fixed holding period or when VixFix drops significantly
                        elif i > 5 and signals[i-5] == 1:  # Hold for 5 days
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    if ret > 0:
                        returns.append(ret)
                    
                except Exception as e:
                    continue
            
            if returns:
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
                if indicator_name == 'WaveTrend':
                    result = calculate_wavetrend(data, params['wt_n1'], params['wt_n2'])
                    if result is not None:
                        wt1 = result['wt1']
                        wt2 = result['wt2']
                        
                        signals = []
                        for i in range(1, len(wt1)):
                            if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                                signals.append(0)
                            elif (wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i-1] <= wt2.iloc[i-1] 
                                and wt1.iloc[i] < params['wt_oversold']):
                                signals.append(1)
                            elif (wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[i-1] >= wt2.iloc[i-1] 
                                  and wt1.iloc[i] > params['wt_overbought']):
                                signals.append(-1)
                            else:
                                signals.append(0)
                
                elif indicator_name == 'Squeeze':
                    result = calculate_squeeze_momentum(
                        data, params['sq_bb_length'], params['sq_bb_mult'],
                        params['sq_kc_length'], params['sq_kc_mult'], params['sq_mom_length']
                    )
                    if result is not None:
                        squeeze = result['squeeze']
                        momentum = result['momentum']
                        
                        signals = []
                        for i in range(1, len(squeeze)):
                            if pd.isna(squeeze.iloc[i]):
                                signals.append(0)
                            elif (squeeze.iloc[i] == 0 and squeeze.iloc[i-1] == 1 
                                and momentum.iloc[i] > 0):
                                signals.append(1)
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
                        
                        signals = []
                        for i in range(1, len(vixfix)):
                            if pd.isna(vixfix.iloc[i]):
                                signals.append(0)
                            elif vixfix.iloc[i] > bb_upper.iloc[i] * params['vf_threshold']:
                                signals.append(1)
                            elif i > 5 and signals[i-5] == 1:
                                signals.append(-1)
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
    print("🔍 Finding Universal Optimal Parameters for Remaining Indicators")
    print("="*80)
    
    optimizer = RemainingOptimizer()
    
    # Load existing parameters
    try:
        with open('universal_optimal_parameters.json', 'r') as f:
            universal_params = json.load(f)
    except:
        universal_params = {}
    
    # Optimize remaining indicators
    wt_params, wt_return = optimizer.optimize_wavetrend_universal()
    sq_params, sq_return = optimizer.optimize_squeeze_universal()
    vf_params, vf_return = optimizer.optimize_vixfix_universal()
    
    # Update universal parameters
    universal_params.update({
        "WaveTrend": {
            "params": wt_params,
            "avg_return": wt_return
        },
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
    
    # Save updated results
    with open('universal_optimal_parameters_complete.json', 'w') as f:
        json.dump(universal_params, f, indent=2)
    
    print("\n✅ All universal parameters saved to: universal_optimal_parameters_complete.json")
    
    # Display summary
    print("\n" + "="*80)
    print("📊 ALL UNIVERSAL PARAMETERS SUMMARY")
    print("="*80)
    
    indicators = ['MACD', 'ADX', 'Supertrend', 'WaveTrend', 'Squeeze', 'VixFix']
    for ind in indicators:
        if ind in universal_params:
            print(f"\n{ind}:")
            print(f"  Parameters: {universal_params[ind]['params']}")
            print(f"  Avg Return: {universal_params[ind]['avg_return']:.1f}%")


if __name__ == "__main__":
    main()