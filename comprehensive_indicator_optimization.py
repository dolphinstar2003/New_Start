#!/usr/bin/env python3
"""
Comprehensive Indicator Optimization
Test all indicators for all symbols across different time periods
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.supertrend import calculate_supertrend
from indicators.adx_di import calculate_adx_di
from indicators.wavetrend import calculate_wavetrend
from indicators.macd_custom import calculate_macd_custom
from indicators.squeeze_momentum import calculate_squeeze_momentum
from indicators.vixfix import calculate_vixfix
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ComprehensiveOptimizer:
    """Optimize indicators for all symbols and time periods"""
    
    def __init__(self):
        self.indicator_calc = IndicatorCalculator(DATA_DIR)
        self.results = {}
        
        # Define time periods to test
        self.time_periods = {
            '30d': 30,    # 1 month
            '60d': 60,    # 2 months  
            '90d': 90,    # 3 months
            '180d': 180,  # 6 months
            '365d': 365,  # 1 year
            'all': None   # All available data
        }
        
        # Optimization settings
        self.n_trials = 50  # Reduced for faster execution across all symbols
        
        logger.info("Comprehensive Optimizer initialized")
    
    def optimize_macd(self, data: pd.DataFrame, period_days: int = None) -> dict:
        """Optimize MACD for given data and period"""
        # Limit data to specified period
        if period_days:
            data = data.tail(period_days + 50)  # Extra buffer for indicators
        
        def objective(trial):
            fast = trial.suggest_int('macd_fast', 8, 20)
            slow = trial.suggest_int('macd_slow', 21, 35)
            signal = trial.suggest_int('macd_signal', 5, 15)
            
            if fast >= slow:
                return -100
            
            try:
                result = calculate_macd_custom(data, fast, slow, signal)
                if result is None or result.empty:
                    return -100
                
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
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_adx(self, data: pd.DataFrame, period_days: int = None) -> dict:
        """Optimize ADX for given data and period"""
        if period_days:
            data = data.tail(period_days + 50)
        
        def objective(trial):
            period = trial.suggest_int('adx_period', 7, 21)
            threshold = trial.suggest_float('adx_threshold', 15, 40)
            
            try:
                result = calculate_adx_di(data, length=period, threshold=threshold)
                if result is None or result.empty:
                    return -100
                
                adx = result['adx']
                
                # ADX strategy
                signals = []
                for i in range(1, len(adx)):
                    if pd.isna(adx.iloc[i]):
                        signals.append(0)
                        continue
                    
                    if adx.iloc[i] > threshold and adx.iloc[i] > adx.iloc[i-1]:
                        signals.append(1)
                    elif adx.iloc[i] < threshold:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_supertrend(self, data: pd.DataFrame, period_days: int = None) -> dict:
        """Optimize Supertrend for given data and period"""
        if period_days:
            data = data.tail(period_days + 50)
        
        def objective(trial):
            period = trial.suggest_int('st_period', 5, 25)
            multiplier = trial.suggest_float('st_multiplier', 0.5, 5.0)
            
            try:
                result = calculate_supertrend(data, period, multiplier)
                if result is None or result.empty:
                    return -100
                
                trend = result['trend']
                
                # Supertrend strategy
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
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_wavetrend(self, data: pd.DataFrame, period_days: int = None) -> dict:
        """Optimize WaveTrend for given data and period"""
        if period_days:
            data = data.tail(period_days + 50)
        
        def objective(trial):
            channel_length = trial.suggest_int('wt_channel_length', 5, 15)
            average_length = trial.suggest_int('wt_average_length', 15, 30)
            overbought = trial.suggest_float('wt_overbought', 40, 70)
            oversold = trial.suggest_float('wt_oversold', -70, -40)
            
            try:
                result = calculate_wavetrend(data, channel_length, average_length)
                if result is None or result.empty:
                    return -100
                
                wt1 = result['wt1']
                wt2 = result['wt2']
                
                # WaveTrend strategy
                signals = []
                for i in range(1, len(wt1)):
                    if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy signal: oversold bounce
                    if wt1.iloc[i] < oversold and wt1.iloc[i] > wt1.iloc[i-1] and wt1.iloc[i] > wt2.iloc[i]:
                        signals.append(1)
                    # Sell signal: overbought decline
                    elif wt1.iloc[i] > overbought and wt1.iloc[i] < wt1.iloc[i-1] and wt1.iloc[i] < wt2.iloc[i]:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_squeeze_momentum(self, data: pd.DataFrame, period_days: int = None) -> dict:
        """Optimize Squeeze Momentum for given data and period"""
        if period_days:
            data = data.tail(period_days + 50)
        
        def objective(trial):
            bb_length = trial.suggest_int('sq_bb_length', 15, 25)
            bb_mult = trial.suggest_float('sq_bb_mult', 1.5, 2.5)
            kc_length = trial.suggest_int('sq_kc_length', 15, 25)
            kc_mult = trial.suggest_float('sq_kc_mult', 1.0, 2.0)
            
            try:
                result = calculate_squeeze_momentum(data, bb_length, bb_mult, kc_length, kc_mult)
                if result is None or result.empty:
                    return -100
                
                momentum = result['momentum']
                squeeze_on = result['squeeze_on']
                
                # Squeeze Momentum strategy
                signals = []
                for i in range(1, len(momentum)):
                    if pd.isna(momentum.iloc[i]) or pd.isna(squeeze_on.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy when momentum turns positive and no squeeze
                    if momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0 and not squeeze_on.iloc[i]:
                        signals.append(1)
                    # Sell when momentum turns negative
                    elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_vixfix(self, data: pd.DataFrame, period_days: int = None) -> dict:
        """Optimize VixFix for given data and period"""
        if period_days:
            data = data.tail(period_days + 50)
        
        def objective(trial):
            length = trial.suggest_int('vix_length', 10, 30)
            threshold = trial.suggest_float('vix_threshold', 10, 30)
            
            try:
                result = calculate_vixfix(data, length)
                if result is None or result.empty:
                    return -100
                
                vixfix = result['vixfix']
                
                # VixFix strategy (buy on extreme fear)
                signals = []
                for i in range(1, len(vixfix)):
                    if pd.isna(vixfix.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy when VixFix is high (extreme fear) and declining
                    if vixfix.iloc[i] > threshold and vixfix.iloc[i] < vixfix.iloc[i-1]:
                        signals.append(1)
                    # Sell when VixFix is very low (complacency)
                    elif vixfix.iloc[i] < 5:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    if i+1 < len(returns):
                        strategy_returns.append(position * returns.iloc[i+1])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def optimize_all_indicators_for_symbol(self, symbol: str) -> dict:
        """Optimize all indicators for a single symbol across all time periods"""
        print(f"\n📊 Optimizing {symbol}...")
        
        # Load data
        data = self.indicator_calc.load_raw_data(symbol, '1d')
        if data is None:
            print(f"❌ No data for {symbol}")
            return {}
        
        symbol_results = {}
        
        # Test each time period
        for period_name, period_days in self.time_periods.items():
            print(f"\n  ⏱️  Period: {period_name}")
            period_results = {}
            
            # MACD
            try:
                print(f"    • MACD...", end='', flush=True)
                macd_result = self.optimize_macd(data, period_days)
                period_results['MACD'] = macd_result
                print(f" ✓ {macd_result['best_score']:.1f}%")
            except Exception as e:
                print(f" ✗ Failed")
                period_results['MACD'] = {'error': str(e)}
            
            # ADX
            try:
                print(f"    • ADX...", end='', flush=True)
                adx_result = self.optimize_adx(data, period_days)
                period_results['ADX'] = adx_result
                print(f" ✓ {adx_result['best_score']:.1f}%")
            except Exception as e:
                print(f" ✗ Failed")
                period_results['ADX'] = {'error': str(e)}
            
            # Supertrend
            try:
                print(f"    • Supertrend...", end='', flush=True)
                st_result = self.optimize_supertrend(data, period_days)
                period_results['Supertrend'] = st_result
                print(f" ✓ {st_result['best_score']:.1f}%")
            except Exception as e:
                print(f" ✗ Failed")
                period_results['Supertrend'] = {'error': str(e)}
            
            # WaveTrend
            try:
                print(f"    • WaveTrend...", end='', flush=True)
                wt_result = self.optimize_wavetrend(data, period_days)
                period_results['WaveTrend'] = wt_result
                print(f" ✓ {wt_result['best_score']:.1f}%")
            except Exception as e:
                print(f" ✗ Failed")
                period_results['WaveTrend'] = {'error': str(e)}
            
            # Squeeze Momentum
            try:
                print(f"    • Squeeze Momentum...", end='', flush=True)
                sq_result = self.optimize_squeeze_momentum(data, period_days)
                period_results['SqueezeMomentum'] = sq_result
                print(f" ✓ {sq_result['best_score']:.1f}%")
            except Exception as e:
                print(f" ✗ Failed")
                period_results['SqueezeMomentum'] = {'error': str(e)}
            
            # VixFix
            try:
                print(f"    • VixFix...", end='', flush=True)
                vix_result = self.optimize_vixfix(data, period_days)
                period_results['VixFix'] = vix_result
                print(f" ✓ {vix_result['best_score']:.1f}%")
            except Exception as e:
                print(f" ✗ Failed")
                period_results['VixFix'] = {'error': str(e)}
            
            symbol_results[period_name] = period_results
        
        return symbol_results
    
    def run_comprehensive_optimization(self, symbols: list = None):
        """Run optimization for all symbols"""
        if symbols is None:
            symbols = SACRED_SYMBOLS
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE INDICATOR OPTIMIZATION")
        print("="*80)
        print(f"\n📊 Symbols: {len(symbols)}")
        print(f"⏱️  Time periods: {list(self.time_periods.keys())}")
        print(f"🔧 Indicators: MACD, ADX, Supertrend, WaveTrend, Squeeze Momentum, VixFix")
        print(f"🎲 Trials per optimization: {self.n_trials}")
        
        # Run optimization for each symbol
        for i, symbol in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
            self.results[symbol] = self.optimize_all_indicators_for_symbol(symbol)
        
        # Save results
        self.save_results()
        
        # Display summary
        self.display_summary()
    
    def save_results(self):
        """Save optimization results to file"""
        filename = f"comprehensive_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return obj
        
        clean_results = convert_types(self.results)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {filename}")
    
    def display_summary(self):
        """Display summary of optimization results"""
        print("\n" + "="*80)
        print("📈 OPTIMIZATION SUMMARY")
        print("="*80)
        
        # Find best parameters across all symbols and periods
        best_by_indicator = {
            'MACD': {'score': -100, 'symbol': '', 'period': '', 'params': {}},
            'ADX': {'score': -100, 'symbol': '', 'period': '', 'params': {}},
            'Supertrend': {'score': -100, 'symbol': '', 'period': '', 'params': {}},
            'WaveTrend': {'score': -100, 'symbol': '', 'period': '', 'params': {}},
            'SqueezeMomentum': {'score': -100, 'symbol': '', 'period': '', 'params': {}},
            'VixFix': {'score': -100, 'symbol': '', 'period': '', 'params': {}}
        }
        
        # Collect all results
        for symbol, symbol_results in self.results.items():
            for period, period_results in symbol_results.items():
                for indicator, result in period_results.items():
                    if 'best_score' in result and result['best_score'] > best_by_indicator[indicator]['score']:
                        best_by_indicator[indicator] = {
                            'score': result['best_score'],
                            'symbol': symbol,
                            'period': period,
                            'params': result['best_params']
                        }
        
        print("\n🏆 BEST PARAMETERS BY INDICATOR:")
        print("-" * 80)
        
        for indicator, best in best_by_indicator.items():
            if best['score'] > -100:
                print(f"\n{indicator}:")
                print(f"  Symbol: {best['symbol']}")
                print(f"  Period: {best['period']}")
                print(f"  Return: {best['score']:.2f}%")
                print(f"  Parameters: {best['params']}")
        
        # Average performance by period
        print("\n📊 AVERAGE PERFORMANCE BY PERIOD:")
        print("-" * 80)
        
        period_stats = {}
        for period in self.time_periods.keys():
            scores = []
            for symbol_results in self.results.values():
                if period in symbol_results:
                    for indicator_result in symbol_results[period].values():
                        if 'best_score' in indicator_result:
                            scores.append(indicator_result['best_score'])
            
            if scores:
                period_stats[period] = {
                    'avg': np.mean(scores),
                    'max': np.max(scores),
                    'count': len(scores)
                }
        
        for period, stats in period_stats.items():
            print(f"{period}: Avg={stats['avg']:.1f}%, Max={stats['max']:.1f}%, Count={stats['count']}")


def main():
    """Main function"""
    optimizer = ComprehensiveOptimizer()
    
    # You can test with a subset first
    # test_symbols = SACRED_SYMBOLS[:3]  # First 3 symbols
    # optimizer.run_comprehensive_optimization(test_symbols)
    
    # Or run for all symbols
    optimizer.run_comprehensive_optimization(SACRED_SYMBOLS)


if __name__ == "__main__":
    main()