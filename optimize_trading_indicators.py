#!/usr/bin/env python3
"""
Trading Indicator Optimization with Optuna
Optimize indicator parameters for better trading performance
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import asyncio
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.supertrend import calculate_supertrend
from indicators.adx_di import calculate_adx_di
from indicators.wavetrend import calculate_wavetrend
from indicators.macd_custom import calculate_macd_custom
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class TradingIndicatorOptimizer:
    """Optimize trading indicator parameters"""
    
    def __init__(self):
        self.indicator_calc = IndicatorCalculator(DATA_DIR)
        
        # Time periods to test
        self.time_periods = {
            '30d': 30,
            '60d': 60,
            '90d': 90,
            '180d': 180,
            '365d': 365,
            'all': None
        }
        
        # Timeframes to test
        self.timeframes = ['1h', '4h', '1d', '1wk']
        
        # Optimization settings
        self.n_trials = 60  # Reduced for faster execution across all combinations
        
        logger.info("Trading Indicator Optimizer initialized")
    
    def optimize_macd_parameters(self, symbol: str) -> dict:
        """Optimize MACD parameters"""
        logger.info(f"Optimizing MACD parameters for {symbol}")
        
        def objective(trial):
            # Suggest MACD parameters
            fast_period = trial.suggest_int('macd_fast', 8, 16)
            slow_period = trial.suggest_int('macd_slow', 20, 30)
            signal_period = trial.suggest_int('macd_signal', 7, 12)
            
            # Ensure fast < slow
            if fast_period >= slow_period:
                return -100  # Invalid configuration
            
            try:
                # Get data and calculate MACD
                data = self.indicator_calc.load_raw_data(symbol, '1d')
                if data is None or len(data) < 100:
                    return -100
                
                # Calculate MACD with trial parameters
                macd_result = calculate_macd_custom(data, fast_period, slow_period, signal_period)
                if macd_result is None:
                    return -100
                
                macd_line = macd_result.get('macd', pd.Series())
                macd_signal = macd_result.get('signal', pd.Series())
                macd_histogram = macd_result.get('histogram', pd.Series())
                
                # Simple trading simulation
                signals = []
                for i in range(1, len(macd_line)):
                    if pd.isna(macd_line[i]) or pd.isna(macd_signal[i]):
                        signals.append(0)
                        continue
                    
                    # Buy signal: MACD crosses above signal
                    if macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1]:
                        signals.append(1)
                    # Sell signal: MACD crosses below signal
                    elif macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1]:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate simple return
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = []
                position = 0
                
                for i, signal in enumerate(signals):
                    if signal == 1:  # Buy
                        position = 1
                    elif signal == -1:  # Sell
                        position = 0
                    
                    strategy_returns.append(position * returns.iloc[i+1] if i+1 < len(returns) else 0)
                
                # Calculate total return
                total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                
                return total_return * 100  # Return as percentage
                
            except Exception as e:
                logger.warning(f"MACD optimization trial failed: {e}")
                return -100
        
        # Run optimization
        study = optuna.create_study(direction='maximize', study_name=f'macd_{symbol}')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        return {
            'indicator': 'MACD',
            'symbol': symbol,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }
    
    def optimize_adx_parameters(self, symbol: str) -> dict:
        """Optimize ADX parameters"""
        logger.info(f"Optimizing ADX parameters for {symbol}")
        
        def objective(trial):
            # Suggest ADX parameters
            period = trial.suggest_int('adx_period', 10, 20)
            threshold = trial.suggest_float('adx_threshold', 20, 35)
            
            try:
                # Get data
                data = self.indicator_calc.load_raw_data(symbol, '1d')
                if data is None or len(data) < 100:
                    return -100
                
                # Calculate ADX
                adx_result = calculate_adx_di(data, length=period, threshold=threshold)
                if adx_result is None or adx_result.empty:
                    return -100
                
                adx = adx_result.get('adx', pd.Series())
                
                # Simple trend following strategy
                signals = []
                for i in range(1, len(adx)):
                    if pd.isna(adx[i]):
                        signals.append(0)
                        continue
                    
                    # Strong trend signal
                    if adx[i] > threshold and adx[i] > adx[i-1]:
                        signals.append(1)  # Buy on strong uptrend
                    elif adx[i] < threshold:
                        signals.append(-1)  # Sell on weak trend
                    else:
                        signals.append(0)
                
                # Calculate return
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = []
                position = 0
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    strategy_returns.append(position * returns.iloc[i+1] if i+1 < len(returns) else 0)
                
                total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                return total_return * 100
                
            except Exception as e:
                logger.warning(f"ADX optimization trial failed: {e}")
                return -100
        
        study = optuna.create_study(direction='maximize', study_name=f'adx_{symbol}')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        return {
            'indicator': 'ADX',
            'symbol': symbol,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }
    
    def optimize_supertrend_parameters(self, symbol: str) -> dict:
        """Optimize Supertrend parameters"""
        logger.info(f"Optimizing Supertrend parameters for {symbol}")
        
        def objective(trial):
            # Suggest Supertrend parameters
            period = trial.suggest_int('st_period', 8, 15)
            multiplier = trial.suggest_float('st_multiplier', 2.5, 4.0)
            
            try:
                # Get data
                data = self.indicator_calc.load_raw_data(symbol, '1d')
                if data is None or len(data) < 100:
                    return -100
                
                # Calculate Supertrend
                st_result = calculate_supertrend(data, period, multiplier)
                if st_result is None or st_result.empty:
                    return -100
                
                st_trend = st_result.get('supertrend', pd.Series())
                st_direction = st_result.get('direction', pd.Series())
                
                # Supertrend strategy
                signals = []
                for i in range(1, len(st_direction)):
                    if pd.isna(st_direction[i]):
                        signals.append(0)
                        continue
                    
                    # Buy when trend changes to up
                    if st_direction[i] == 1 and st_direction[i-1] == -1:
                        signals.append(1)
                    # Sell when trend changes to down
                    elif st_direction[i] == -1 and st_direction[i-1] == 1:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate return
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = []
                position = 0
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    strategy_returns.append(position * returns.iloc[i+1] if i+1 < len(returns) else 0)
                
                total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                return total_return * 100
                
            except Exception as e:
                logger.warning(f"Supertrend optimization trial failed: {e}")
                return -100
        
        study = optuna.create_study(direction='maximize', study_name=f'supertrend_{symbol}')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        return {
            'indicator': 'Supertrend',
            'symbol': symbol,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }
    
    def optimize_wavetrend_parameters(self, symbol: str) -> dict:
        """Optimize WaveTrend parameters"""
        logger.info(f"Optimizing WaveTrend parameters for {symbol}")
        
        def objective(trial):
            # Suggest WaveTrend parameters
            channel_length = trial.suggest_int('wt_channel_length', 8, 12)
            average_length = trial.suggest_int('wt_average_length', 19, 25)
            overbought = trial.suggest_float('wt_overbought', 40, 65)
            oversold = trial.suggest_float('wt_oversold', -50, -25)
            
            try:
                # Get data
                data = self.indicator_calc.load_raw_data(symbol, '1d')
                if data is None or len(data) < 100:
                    return -100
                
                # Calculate WaveTrend
                wt_result = calculate_wavetrend(data, channel_length, average_length)
                if wt_result is None or wt_result.empty:
                    return -100
                
                wt1 = wt_result.get('wt1', pd.Series())
                wt2 = wt_result.get('wt2', pd.Series())
                
                # WaveTrend strategy - using crossovers
                signals = []
                for i in range(1, len(wt1)):
                    if pd.isna(wt1[i]) or pd.isna(wt2[i]):
                        signals.append(0)
                        continue
                    
                    # Buy on bullish crossover in oversold or momentum reversal
                    if (wt1[i] > wt2[i] and wt1[i-1] <= wt2[i-1] and wt1[i] < oversold) or \
                       (wt1[i] < oversold and wt1[i] > wt1[i-1]):
                        signals.append(1)
                    # Sell on bearish crossover in overbought or momentum reversal
                    elif (wt1[i] < wt2[i] and wt1[i-1] >= wt2[i-1] and wt1[i] > overbought) or \
                         (wt1[i] > overbought and wt1[i] < wt1[i-1]):
                        signals.append(-1)
                    else:
                        signals.append(0)
                
                # Calculate return
                returns = data['close'].pct_change().fillna(0)
                strategy_returns = []
                position = 0
                
                for i, signal in enumerate(signals):
                    if signal == 1:
                        position = 1
                    elif signal == -1:
                        position = 0
                    
                    strategy_returns.append(position * returns.iloc[i+1] if i+1 < len(returns) else 0)
                
                total_return = (1 + pd.Series(strategy_returns)).prod() - 1
                return total_return * 100
                
            except Exception as e:
                logger.warning(f"WaveTrend optimization trial failed: {e}")
                return -100
        
        study = optuna.create_study(direction='maximize', study_name=f'wavetrend_{symbol}')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        return {
            'indicator': 'WaveTrend',
            'symbol': symbol,
            'best_params': study.best_params,
            'best_score': study.best_value,
            'trials': len(study.trials)
        }
    
    def optimize_all_comprehensive(self, symbols: list = None) -> dict:
        """Comprehensive optimization for all indicators, timeframes, and periods"""
        if symbols is None:
            symbols = SACRED_SYMBOLS
        
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE TRADING INDICATOR OPTIMIZATION")
        print("="*80)
        print(f"\n📊 Configuration:")
        print(f"   • Symbols: {len(symbols)}")
        print(f"   • Timeframes: {self.timeframes}")
        print(f"   • Time periods: {list(self.time_periods.keys())}")
        print(f"   • Indicators: MACD, ADX, Supertrend, WaveTrend, Squeeze Momentum, VixFix")
        print(f"   • Trials per optimization: {self.n_trials}")
        print(f"   • Total optimizations: {len(symbols) * len(self.timeframes) * len(self.time_periods) * 6}")
        
        comprehensive_results = {}
        
        for symbol_idx, symbol in enumerate(symbols):
            print(f"\n[{symbol_idx+1}/{len(symbols)}] Processing {symbol}...")
            comprehensive_results[symbol] = {}
            
            for tf_idx, timeframe in enumerate(self.timeframes):
                print(f"\n  [{tf_idx+1}/{len(self.timeframes)}] Timeframe: {timeframe}")
                comprehensive_results[symbol][timeframe] = {}
                
                # Load data for this timeframe
                data = self.indicator_calc.load_raw_data(symbol, timeframe)
                if data is None:
                    print(f"    ❌ No data available for {symbol} {timeframe}")
                    continue
                
                for period_name, period_days in self.time_periods.items():
                    print(f"\n    Period: {period_name}")
                    period_results = {}
                    
                    # Limit data to period if specified
                    period_data = data
                    if period_days:
                        period_data = data.tail(period_days + 50)  # Extra buffer
                    
                    # Optimize each indicator
                    try:
                        print(f"      • MACD...", end='', flush=True)
                        macd_result = self._optimize_macd_for_data(period_data)
                        period_results['MACD'] = macd_result
                        print(f" ✓ {macd_result['best_score']:.1f}%")
                    except Exception as e:
                        print(f" ✗ Failed: {str(e)[:30]}")
                        period_results['MACD'] = {'error': str(e)}
                    
                    try:
                        print(f"      • ADX...", end='', flush=True)
                        adx_result = self._optimize_adx_for_data(period_data)
                        period_results['ADX'] = adx_result
                        print(f" ✓ {adx_result['best_score']:.1f}%")
                    except Exception as e:
                        print(f" ✗ Failed: {str(e)[:30]}")
                        period_results['ADX'] = {'error': str(e)}
                    
                    try:
                        print(f"      • Supertrend...", end='', flush=True)
                        st_result = self._optimize_supertrend_for_data(period_data)
                        period_results['Supertrend'] = st_result
                        print(f" ✓ {st_result['best_score']:.1f}%")
                    except Exception as e:
                        print(f" ✗ Failed: {str(e)[:30]}")
                        period_results['Supertrend'] = {'error': str(e)}
                    
                    try:
                        print(f"      • WaveTrend...", end='', flush=True)
                        wt_result = self._optimize_wavetrend_for_data(period_data)
                        period_results['WaveTrend'] = wt_result
                        print(f" ✓ {wt_result['best_score']:.1f}%")
                    except Exception as e:
                        print(f" ✗ Failed: {str(e)[:30]}")
                        period_results['WaveTrend'] = {'error': str(e)}
                    
                    try:
                        print(f"      • Squeeze Momentum...", end='', flush=True)
                        sq_result = self._optimize_squeeze_momentum_for_data(period_data)
                        period_results['SqueezeMomentum'] = sq_result
                        print(f" ✓ {sq_result['best_score']:.1f}%")
                    except Exception as e:
                        print(f" ✗ Failed: {str(e)[:30]}")
                        period_results['SqueezeMomentum'] = {'error': str(e)}
                    
                    try:
                        print(f"      • VixFix...", end='', flush=True)
                        vix_result = self._optimize_vixfix_for_data(period_data)
                        period_results['VixFix'] = vix_result
                        print(f" ✓ {vix_result['best_score']:.1f}%")
                    except Exception as e:
                        print(f" ✗ Failed: {str(e)[:30]}")
                        period_results['VixFix'] = {'error': str(e)}
                    
                    comprehensive_results[symbol][timeframe][period_name] = period_results
        
        # Save results
        self._save_comprehensive_results(comprehensive_results)
        
        # Display summary
        self._display_comprehensive_summary(comprehensive_results)
        
        return comprehensive_results
    
    def optimize_all_indicators(self, symbols: list = None) -> dict:
        """Optimize all indicators for given symbols"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:3]  # Test with first 3 symbols
        
        print("\n" + "="*70)
        print("🎯 TRADING INDICATOR OPTIMIZATION")
        print("="*70)
        
        results = {}
        
        for symbol in symbols:
            print(f"\n📊 Optimizing indicators for {symbol}...")
            results[symbol] = {}
            
            # Optimize each indicator
            indicators = ['MACD', 'ADX', 'Supertrend', 'WaveTrend']
            
            for indicator in indicators:
                try:
                    if indicator == 'MACD':
                        result = self.optimize_macd_parameters(symbol)
                    elif indicator == 'ADX':
                        result = self.optimize_adx_parameters(symbol)
                    elif indicator == 'Supertrend':
                        result = self.optimize_supertrend_parameters(symbol)
                    elif indicator == 'WaveTrend':
                        result = self.optimize_wavetrend_parameters(symbol)
                    
                    results[symbol][indicator] = result
                    print(f"   ✅ {indicator}: {result['best_score']:.2f}% return")
                    
                except Exception as e:
                    print(f"   ❌ {indicator}: Failed - {e}")
                    results[symbol][indicator] = {'error': str(e)}
        
        # Display summary
        self.display_optimization_results(results)
        
        return results
    
    def display_optimization_results(self, results: dict):
        """Display optimization results summary"""
        print("\n" + "="*70)
        print("📈 INDICATOR OPTIMIZATION RESULTS")
        print("="*70)
        
        print(f"\n{'Symbol':<8} {'Indicator':<12} {'Best Return':<12} {'Best Parameters'}")
        print("-" * 70)
        
        for symbol, symbol_results in results.items():
            for indicator, result in symbol_results.items():
                if 'error' not in result:
                    best_return = f"{result['best_score']:.2f}%"
                    params_str = ", ".join([f"{k}={v}" for k, v in result['best_params'].items()])
                    if len(params_str) > 25:
                        params_str = params_str[:22] + "..."
                    
                    print(f"{symbol:<8} {indicator:<12} {best_return:<12} {params_str}")
                else:
                    print(f"{symbol:<8} {indicator:<12} {'ERROR':<12} {result['error'][:25]}")
        
        # Best performing combinations
        print(f"\n🏆 Best Performing Indicators:")
        all_results = []
        for symbol, symbol_results in results.items():
            for indicator, result in symbol_results.items():
                if 'error' not in result:
                    all_results.append({
                        'symbol': symbol,
                        'indicator': indicator,
                        'return': result['best_score'],
                        'params': result['best_params']
                    })
        
        # Sort by return
        all_results.sort(key=lambda x: x['return'], reverse=True)
        
        for i, result in enumerate(all_results[:5]):
            print(f"   {i+1}. {result['symbol']} {result['indicator']}: {result['return']:.2f}%")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Update core_indicators.py with optimal parameters")
        print(f"   2. Test optimized indicators in rotation strategy")
        print(f"   3. Run full backtest with optimized parameters")


    def _optimize_macd_for_data(self, data: pd.DataFrame) -> dict:
        """Optimize MACD for given data"""
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
                
                # Generate signals and calculate returns
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _optimize_adx_for_data(self, data: pd.DataFrame) -> dict:
        """Optimize ADX for given data"""
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _optimize_supertrend_for_data(self, data: pd.DataFrame) -> dict:
        """Optimize Supertrend for given data"""
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _optimize_wavetrend_for_data(self, data: pd.DataFrame) -> dict:
        """Optimize WaveTrend for given data"""
        def objective(trial):
            channel_length = trial.suggest_int('wt_channel_length', 5, 15)
            average_length = trial.suggest_int('wt_average_length', 15, 30)
            overbought = trial.suggest_float('wt_overbought', 40, 70)
            oversold = trial.suggest_float('wt_oversold', -50, -20)  # Adjusted range based on actual data
            
            try:
                result = calculate_wavetrend(data, channel_length, average_length)
                if result is None or result.empty:
                    return -100
                
                wt1 = result['wt1']
                wt2 = result['wt2']
                
                # WaveTrend strategy - using crossovers instead of levels
                signals = []
                for i in range(1, len(wt1)):
                    if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy on bullish crossover in oversold territory or momentum reversal
                    if (wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i-1] <= wt2.iloc[i-1] and wt1.iloc[i] < oversold) or \
                       (wt1.iloc[i] < oversold and wt1.iloc[i] > wt1.iloc[i-1] and wt1.iloc[i-1] < wt1.iloc[i-2]):
                        signals.append(1)
                    # Sell on bearish crossover in overbought territory or momentum reversal
                    elif (wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[i-1] >= wt2.iloc[i-1] and wt1.iloc[i] > overbought) or \
                         (wt1.iloc[i] > overbought and wt1.iloc[i] < wt1.iloc[i-1] and wt1.iloc[i-1] > wt1.iloc[i-2]):
                        signals.append(-1)
                    else:
                        signals.append(0)
                
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _save_comprehensive_results(self, results: dict):
        """Save comprehensive results to JSON"""
        from datetime import datetime
        import json
        
        filename = f"comprehensive_indicator_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
        
        clean_results = convert_types(results)
        
        with open(filename, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n📁 Results saved to: {filename}")
    
    def _optimize_squeeze_momentum_for_data(self, data: pd.DataFrame) -> dict:
        """Optimize Squeeze Momentum for given data"""
        def objective(trial):
            bb_length = trial.suggest_int('sq_bb_length', 15, 25)
            bb_mult = trial.suggest_float('sq_bb_mult', 1.5, 2.5)
            kc_length = trial.suggest_int('sq_kc_length', 15, 25)
            kc_mult = trial.suggest_float('sq_kc_mult', 1.0, 2.0)
            
            try:
                from indicators.squeeze_momentum import calculate_squeeze_momentum
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
                    
                    if momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0 and not squeeze_on.iloc[i]:
                        signals.append(1)
                    elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _optimize_vixfix_for_data(self, data: pd.DataFrame) -> dict:
        """Optimize VixFix for given data"""
        def objective(trial):
            length = trial.suggest_int('vix_length', 10, 30)
            buy_threshold = trial.suggest_float('vix_buy_threshold', 15, 35)
            sell_threshold = trial.suggest_float('vix_sell_threshold', 3, 10)
            
            try:
                from indicators.vixfix import calculate_vixfix
                result = calculate_vixfix(data, length)
                if result is None or result.empty:
                    return -100
                
                vixfix = result['vixfix']
                bb_upper = result['bb_upper']
                
                # VixFix strategy - improved logic
                signals = []
                for i in range(1, len(vixfix)):
                    if pd.isna(vixfix.iloc[i]) or pd.isna(bb_upper.iloc[i]):
                        signals.append(0)
                        continue
                    
                    # Buy on extreme fear spike and reversal
                    buy_condition = (
                        (vixfix.iloc[i] > buy_threshold and vixfix.iloc[i] < vixfix.iloc[i-1]) or  # Fear spike reversal
                        (vixfix.iloc[i] > bb_upper.iloc[i] and vixfix.iloc[i] < vixfix.iloc[i-1])  # Above BB and declining
                    )
                    
                    # Sell when fear is extremely low (complacency) and starting to rise
                    sell_condition = (
                        vixfix.iloc[i] < sell_threshold and vixfix.iloc[i] > vixfix.iloc[i-1]  # Low fear increasing
                    )
                    
                    if buy_condition:
                        signals.append(1)
                    elif sell_condition:
                        signals.append(-1)
                    else:
                        signals.append(0)
                
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
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
    
    def _display_comprehensive_summary(self, results: dict):
        """Display comprehensive optimization summary"""
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE OPTIMIZATION SUMMARY")
        print("="*80)
        
        # Find global best for each indicator
        best_by_indicator = {
            'MACD': {'score': -100, 'symbol': '', 'timeframe': '', 'period': '', 'params': {}},
            'ADX': {'score': -100, 'symbol': '', 'timeframe': '', 'period': '', 'params': {}},
            'Supertrend': {'score': -100, 'symbol': '', 'timeframe': '', 'period': '', 'params': {}},
            'WaveTrend': {'score': -100, 'symbol': '', 'timeframe': '', 'period': '', 'params': {}},
            'SqueezeMomentum': {'score': -100, 'symbol': '', 'timeframe': '', 'period': '', 'params': {}},
            'VixFix': {'score': -100, 'symbol': '', 'timeframe': '', 'period': '', 'params': {}}
        }
        
        # Analyze results
        for symbol, tf_results in results.items():
            for timeframe, period_results in tf_results.items():
                for period, indicator_results in period_results.items():
                    for indicator, result in indicator_results.items():
                        if 'best_score' in result and result['best_score'] > best_by_indicator[indicator]['score']:
                            best_by_indicator[indicator] = {
                                'score': result['best_score'],
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'period': period,
                                'params': result['best_params']
                            }
        
        # Display best results
        print("\n🏆 GLOBAL BEST PARAMETERS BY INDICATOR:")
        print("-" * 80)
        
        for indicator, best in best_by_indicator.items():
            if best['score'] > -100:
                print(f"\n{indicator}:")
                print(f"  Symbol: {best['symbol']}")
                print(f"  Timeframe: {best['timeframe']}")
                print(f"  Period: {best['period']}")
                print(f"  Return: {best['score']:.2f}%")
                print(f"  Parameters: {best['params']}")
        
        # Best by timeframe
        print("\n📊 BEST PERFORMANCE BY TIMEFRAME:")
        print("-" * 80)
        
        tf_stats = {}
        for symbol, tf_results in results.items():
            for timeframe, period_results in tf_results.items():
                if timeframe not in tf_stats:
                    tf_stats[timeframe] = []
                
                for period, indicator_results in period_results.items():
                    for indicator, result in indicator_results.items():
                        if 'best_score' in result:
                            tf_stats[timeframe].append(result['best_score'])
        
        for timeframe, scores in tf_stats.items():
            if scores:
                avg_score = np.mean(scores)
                max_score = np.max(scores)
                print(f"{timeframe}: Avg={avg_score:.1f}%, Max={max_score:.1f}%")


def main():
    """Main optimization function"""
    print("🚀 Starting Comprehensive Trading Indicator Optimization...")
    
    optimizer = TradingIndicatorOptimizer()
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Optimize trading indicators')
    parser.add_argument('--mode', choices=['simple', 'comprehensive'], default='comprehensive',
                        help='Optimization mode: simple (basic) or comprehensive (all timeframes/periods)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to optimize')
    parser.add_argument('--test', action='store_true', help='Test mode with fewer trials')
    args = parser.parse_args()
    
    if args.test:
        optimizer.n_trials = 10  # Reduce trials for testing
    
    symbols = args.symbols if args.symbols else SACRED_SYMBOLS
    
    try:
        if args.mode == 'comprehensive':
            results = optimizer.optimize_all_comprehensive(symbols)
            print(f"\n✅ Comprehensive indicator optimization completed!")
        else:
            results = optimizer.optimize_all_indicators(symbols)
            print(f"\n✅ Simple indicator optimization completed!")
            
            # Save results
            import json
            with open('optimized_indicator_parameters.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = {}
                for symbol, symbol_results in results.items():
                    json_results[symbol] = {}
                    for indicator, result in symbol_results.items():
                        if 'error' not in result:
                            json_results[symbol][indicator] = {
                                'best_params': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                              for k, v in result['best_params'].items()},
                                'best_score': float(result['best_score']),
                                'trials': result['trials']
                            }
                        else:
                            json_results[symbol][indicator] = result
                
                json.dump(json_results, f, indent=2)
            
            print(f"📁 Results saved to: optimized_indicator_parameters.json")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()