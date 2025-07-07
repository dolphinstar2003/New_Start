#!/usr/bin/env python3
"""
Quick Optimization for Key Indicators and Timeframes
Focus on most important combinations only
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.supertrend import calculate_supertrend
from indicators.adx_di import calculate_adx_di
from indicators.macd_custom import calculate_macd_custom
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class QuickOptimizer:
    """Quick optimization for key indicators"""
    
    def __init__(self):
        self.indicator_calc = IndicatorCalculator(DATA_DIR)
        self.results = {}
        
        # Focus on most important timeframes
        self.key_timeframes = ['1d', '4h']  # Daily and 4-hour
        
        # Focus on most important periods
        self.key_periods = {
            '90d': 90,    # 3 months
            '365d': 365,  # 1 year
            'all': None   # All data
        }
        
        # More trials for accuracy
        self.n_trials = 100
        
        logger.info("Quick Optimizer initialized")
    
    def optimize_macd(self, data: pd.DataFrame) -> dict:
        """Optimize MACD"""
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
                        continue
                    
                    if macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                        signals.append(1)
                    elif macd_line.iloc[i] < macd_signal.iloc[i] and macd_line.iloc[i-1] >= macd_signal.iloc[i-1]:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                if not signals:
                    return -100
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                signal_idx = 0
                for i in range(len(macd_line) - 1, len(returns)):
                    if signal_idx < len(signals):
                        if signals[signal_idx] == 1:
                            position = 1
                        elif signals[signal_idx] == -1:
                            position = 0
                        signal_idx += 1
                    
                    strategy_returns.append(position * returns.iloc[i])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value
        }
    
    def optimize_adx(self, data: pd.DataFrame) -> dict:
        """Optimize ADX"""
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
                        continue
                    
                    if adx.iloc[i] > threshold and adx.iloc[i] > adx.iloc[i-1]:
                        signals.append(1)
                    elif adx.iloc[i] < threshold * 0.8:  # Exit when trend weakens
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                if not signals:
                    return -100
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                signal_idx = 0
                for i in range(len(adx) - 1, len(returns)):
                    if signal_idx < len(signals):
                        if signals[signal_idx] == 1:
                            position = 1
                        elif signals[signal_idx] == -1:
                            position = 0
                        signal_idx += 1
                    
                    strategy_returns.append(position * returns.iloc[i])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value
        }
    
    def optimize_supertrend(self, data: pd.DataFrame) -> dict:
        """Optimize Supertrend"""
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
                        continue
                    
                    if trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                        signals.append(1)
                    elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                if not signals:
                    return -100
                
                # Calculate returns
                returns = data['close'].pct_change().fillna(0)
                position = 0
                strategy_returns = []
                
                signal_idx = 0
                for i in range(len(trend) - 1, len(returns)):
                    if signal_idx < len(signals):
                        if signals[signal_idx] == 1:
                            position = 1
                        elif signals[signal_idx] == -1:
                            position = 0
                        signal_idx += 1
                    
                    strategy_returns.append(position * returns.iloc[i])
                
                if strategy_returns:
                    total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                    return total_return * 100
                
                return -100
                
            except Exception as e:
                return -100
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value
        }
    
    def run_quick_optimization(self, symbols: list = None):
        """Run quick optimization for key combinations"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:5]  # Top 5 symbols
        
        print("\n" + "="*80)
        print("🚀 QUICK INDICATOR OPTIMIZATION")
        print("="*80)
        print(f"\n📊 Symbols: {', '.join(symbols)}")
        print(f"⏱️  Timeframes: {', '.join(self.key_timeframes)}")
        print(f"📅 Periods: {', '.join(self.key_periods.keys())}")
        print(f"🔧 Indicators: MACD, ADX, Supertrend")
        print(f"🎲 Trials per optimization: {self.n_trials}")
        
        best_overall = {
            'MACD': {'score': -100},
            'ADX': {'score': -100},
            'Supertrend': {'score': -100}
        }
        
        for symbol in symbols:
            print(f"\n[{symbol}] Processing...")
            self.results[symbol] = {}
            
            for timeframe in self.key_timeframes:
                print(f"  📊 {timeframe}:")
                self.results[symbol][timeframe] = {}
                
                # Load data
                data = self.indicator_calc.load_raw_data(symbol, timeframe)
                if data is None:
                    print(f"    ❌ No data available")
                    continue
                
                for period_name, period_days in self.key_periods.items():
                    # Limit data to period
                    period_data = data
                    if period_days:
                        period_data = data.tail(period_days + 50)
                    
                    print(f"    • {period_name}: ", end='', flush=True)
                    
                    # MACD
                    macd_result = self.optimize_macd(period_data)
                    print(f"MACD={macd_result['best_score']:.1f}%", end='', flush=True)
                    
                    # ADX
                    adx_result = self.optimize_adx(period_data)
                    print(f", ADX={adx_result['best_score']:.1f}%", end='', flush=True)
                    
                    # Supertrend
                    st_result = self.optimize_supertrend(period_data)
                    print(f", ST={st_result['best_score']:.1f}%")
                    
                    # Store results
                    self.results[symbol][timeframe][period_name] = {
                        'MACD': macd_result,
                        'ADX': adx_result,
                        'Supertrend': st_result
                    }
                    
                    # Track best overall
                    if macd_result['best_score'] > best_overall['MACD']['score']:
                        best_overall['MACD'] = {
                            'score': macd_result['best_score'],
                            'params': macd_result['best_params'],
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'period': period_name
                        }
                    
                    if adx_result['best_score'] > best_overall['ADX']['score']:
                        best_overall['ADX'] = {
                            'score': adx_result['best_score'],
                            'params': adx_result['best_params'],
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'period': period_name
                        }
                    
                    if st_result['best_score'] > best_overall['Supertrend']['score']:
                        best_overall['Supertrend'] = {
                            'score': st_result['best_score'],
                            'params': st_result['best_params'],
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'period': period_name
                        }
        
        # Display best results
        print("\n" + "="*80)
        print("🏆 BEST PARAMETERS FOUND")
        print("="*80)
        
        for indicator, best in best_overall.items():
            if best['score'] > -100:
                print(f"\n{indicator}:")
                print(f"  Symbol: {best['symbol']}")
                print(f"  Timeframe: {best['timeframe']}")
                print(f"  Period: {best['period']}")
                print(f"  Return: {best['score']:.2f}%")
                print(f"  Parameters: {best['params']}")
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save optimization results"""
        filename = f"quick_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types
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


def main():
    """Main function"""
    optimizer = QuickOptimizer()
    
    # Top 5 symbols for quick test
    top_symbols = ['GARAN', 'THYAO', 'AKBNK', 'ISCTR', 'SAHOL']
    
    optimizer.run_quick_optimization(top_symbols)


if __name__ == "__main__":
    main()