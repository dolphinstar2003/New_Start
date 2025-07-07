#!/usr/bin/env python3
"""
Find Universal Optimal Parameters for All Symbols
Goal: Find parameters that work well across all 20 symbols
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
from indicators.adx_di import calculate_adx_di
from indicators.supertrend import calculate_supertrend
from loguru import logger

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class UniversalOptimizer:
    """Find universal parameters that work for all symbols"""
    
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
    
    def optimize_macd_universal(self):
        """Find MACD parameters that work for all symbols"""
        print("\n🎯 Optimizing MACD for all symbols...")
        
        def objective(trial):
            # Suggest MACD parameters
            fast = trial.suggest_int('macd_fast', 8, 20)
            slow = trial.suggest_int('macd_slow', 21, 35)
            signal = trial.suggest_int('macd_signal', 5, 15)
            
            if fast >= slow:
                return -1000
            
            # Test on all symbols
            returns = []
            
            for symbol, data in self.all_data.items():
                try:
                    # Calculate MACD
                    result = calculate_macd_custom(data, fast, slow, signal)
                    if result is None or result.empty:
                        continue
                    
                    macd_line = result['macd']
                    macd_signal = result['signal']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(macd_line)):
                        if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                            signals.append(0)
                            continue
                        
                        if macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                            signals.append(1)
                        elif macd_line.iloc[i] < macd_signal.iloc[i] and macd_line.iloc[i-1] >= macd_signal.iloc[i-1]:
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    if ret > 0:  # Only count positive returns
                        returns.append(ret)
                    
                except Exception as e:
                    continue
            
            # Return average return across all symbols
            if returns:
                # Use geometric mean for better representation
                avg_return = np.exp(np.mean(np.log1p(np.array(returns) / 100))) - 1
                return avg_return * 100
            return -1000
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        print(f"\n✅ MACD Optimization complete!")
        print(f"Best universal parameters:")
        print(f"  Fast: {study.best_params['macd_fast']}")
        print(f"  Slow: {study.best_params['macd_slow']}")
        print(f"  Signal: {study.best_params['macd_signal']}")
        print(f"  Avg return: {study.best_value:.2f}%")
        
        # Test on each symbol
        self.test_parameters_on_all_symbols('MACD', study.best_params)
        
        return study.best_params, study.best_value
    
    def optimize_adx_universal(self):
        """Find ADX parameters that work for all symbols"""
        print("\n🎯 Optimizing ADX for all symbols...")
        
        def objective(trial):
            # Suggest ADX parameters
            period = trial.suggest_int('adx_period', 7, 21)
            threshold = trial.suggest_float('adx_threshold', 15, 40)
            exit_threshold = trial.suggest_float('adx_exit_threshold', 10, 30)
            
            if exit_threshold >= threshold:
                return -1000
            
            # Test on all symbols
            returns = []
            
            for symbol, data in self.all_data.items():
                try:
                    # Calculate ADX
                    result = calculate_adx_di(data, length=period, threshold=threshold)
                    if result is None or result.empty:
                        continue
                    
                    adx = result['adx']
                    plus_di = result['plus_di']
                    minus_di = result['minus_di']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(adx)):
                        if pd.isna(adx.iloc[i]) or pd.isna(plus_di.iloc[i]) or pd.isna(minus_di.iloc[i]):
                            signals.append(0)
                            continue
                        
                        if (plus_di.iloc[i] > minus_di.iloc[i] and plus_di.iloc[i-1] <= minus_di.iloc[i-1] 
                            and adx.iloc[i] > threshold):
                            signals.append(1)
                        elif (minus_di.iloc[i] > plus_di.iloc[i] and minus_di.iloc[i-1] <= plus_di.iloc[i-1]) or \
                             (adx.iloc[i] < exit_threshold):
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
        
        print(f"\n✅ ADX Optimization complete!")
        print(f"Best universal parameters:")
        print(f"  Period: {study.best_params['adx_period']}")
        print(f"  Threshold: {study.best_params['adx_threshold']:.2f}")
        print(f"  Exit: {study.best_params['adx_exit_threshold']:.2f}")
        print(f"  Avg return: {study.best_value:.2f}%")
        
        self.test_parameters_on_all_symbols('ADX', study.best_params)
        
        return study.best_params, study.best_value
    
    def optimize_supertrend_universal(self):
        """Find Supertrend parameters that work for all symbols"""
        print("\n🎯 Optimizing Supertrend for all symbols...")
        
        def objective(trial):
            # Suggest Supertrend parameters
            period = trial.suggest_int('st_period', 5, 25)
            multiplier = trial.suggest_float('st_multiplier', 0.5, 5.0)
            
            # Test on all symbols
            returns = []
            
            for symbol, data in self.all_data.items():
                try:
                    # Calculate Supertrend
                    result = calculate_supertrend(data, period, multiplier)
                    if result is None or result.empty:
                        continue
                    
                    trend = result['trend']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(trend)):
                        if pd.isna(trend.iloc[i]):
                            signals.append(0)
                            continue
                        
                        if trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                            signals.append(1)
                        elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
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
        
        print(f"\n✅ Supertrend Optimization complete!")
        print(f"Best universal parameters:")
        print(f"  Period: {study.best_params['st_period']}")
        print(f"  Multiplier: {study.best_params['st_multiplier']:.2f}")
        print(f"  Avg return: {study.best_value:.2f}%")
        
        self.test_parameters_on_all_symbols('Supertrend', study.best_params)
        
        return study.best_params, study.best_value
    
    def test_parameters_on_all_symbols(self, indicator_name, params):
        """Test optimized parameters on each symbol"""
        print(f"\n📊 Testing {indicator_name} on each symbol:")
        print("-" * 60)
        
        results = []
        
        for symbol, data in self.all_data.items():
            try:
                if indicator_name == 'MACD':
                    result = calculate_macd_custom(data, params['macd_fast'], params['macd_slow'], params['macd_signal'])
                    if result is not None:
                        macd_line = result['macd']
                        macd_signal = result['signal']
                        
                        signals = []
                        for i in range(1, len(macd_line)):
                            if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                                signals.append(0)
                            elif macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                                signals.append(1)
                            elif macd_line.iloc[i] < macd_signal.iloc[i] and macd_line.iloc[i-1] >= macd_signal.iloc[i-1]:
                                signals.append(-1)
                            else:
                                signals.append(0)
                
                elif indicator_name == 'ADX':
                    result = calculate_adx_di(data, length=params['adx_period'], threshold=params['adx_threshold'])
                    if result is not None:
                        adx = result['adx']
                        plus_di = result['plus_di']
                        minus_di = result['minus_di']
                        
                        signals = []
                        for i in range(1, len(adx)):
                            if pd.isna(adx.iloc[i]):
                                signals.append(0)
                            elif (plus_di.iloc[i] > minus_di.iloc[i] and plus_di.iloc[i-1] <= minus_di.iloc[i-1] 
                                and adx.iloc[i] > params['adx_threshold']):
                                signals.append(1)
                            elif (minus_di.iloc[i] > plus_di.iloc[i] and minus_di.iloc[i-1] <= plus_di.iloc[i-1]) or \
                                 (adx.iloc[i] < params['adx_exit_threshold']):
                                signals.append(-1)
                            else:
                                signals.append(0)
                
                elif indicator_name == 'Supertrend':
                    result = calculate_supertrend(data, params['st_period'], params['st_multiplier'])
                    if result is not None:
                        trend = result['trend']
                        
                        signals = []
                        for i in range(1, len(trend)):
                            if pd.isna(trend.iloc[i]):
                                signals.append(0)
                            elif trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                                signals.append(1)
                            elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
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
    print("🔍 Finding Universal Optimal Parameters")
    print("="*80)
    
    optimizer = UniversalOptimizer()
    
    # Optimize each indicator
    macd_params, macd_return = optimizer.optimize_macd_universal()
    adx_params, adx_return = optimizer.optimize_adx_universal()
    st_params, st_return = optimizer.optimize_supertrend_universal()
    
    # Save results
    import json
    universal_params = {
        "MACD": {
            "params": macd_params,
            "avg_return": macd_return
        },
        "ADX": {
            "params": adx_params,
            "avg_return": adx_return
        },
        "Supertrend": {
            "params": st_params,
            "avg_return": st_return
        },
        "optimization_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('universal_optimal_parameters.json', 'w') as f:
        json.dump(universal_params, f, indent=2)
    
    print("\n✅ Universal parameters saved to: universal_optimal_parameters.json")


if __name__ == "__main__":
    main()