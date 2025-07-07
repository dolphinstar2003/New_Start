#!/usr/bin/env python3
"""
Regime-Based Parameter Optimizer
Optimizes indicator parameters for different market regimes using Optuna
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import optuna
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.supertrend import calculate_supertrend
from indicators.macd_custom import calculate_macd_custom
from indicators.adx_di import calculate_adx_di
from indicators.wavetrend import calculate_wavetrend
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeBasedOptimizer:
    """Optimizes parameters for different market regimes"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        self.regimes = self.load_regimes()
        self.optimized_params = {}
        
    def load_regimes(self) -> Dict:
        """Load market regimes from analysis"""
        try:
            with open('market_regimes_analysis.json', 'r') as f:
                regime_data = json.load(f)
            return regime_data['regimes']['manual']  # Use manual regimes for now
        except FileNotFoundError:
            logger.error("❌ Market regimes file not found. Run market_regime_analyzer.py first.")
            return {}
    
    def get_regime_data(self, regime_type: str) -> List[pd.DataFrame]:
        """Get data for specific market regime periods"""
        regime_periods = self.regimes.get(regime_type, [])
        all_regime_data = []
        
        for start_date, end_date in regime_periods:
            period_data = []
            
            for symbol in SACRED_SYMBOLS:
                try:
                    data = self.calc.load_raw_data(symbol, '1d')
                    if data is None:
                        continue
                    
                    # Filter to regime period
                    start = pd.to_datetime(start_date).tz_localize(None)
                    end = pd.to_datetime(end_date).tz_localize(None)
                    data_index = data.index.tz_localize(None) if data.index.tz else data.index
                    
                    mask = (data_index >= start) & (data_index <= end)
                    regime_data = data[mask]
                    
                    if len(regime_data) >= 30:  # Minimum 30 days
                        period_data.append(regime_data)
                        
                except Exception as e:
                    logger.error(f"Error loading {symbol} for {regime_type}: {e}")
                    continue
            
            all_regime_data.extend(period_data)
        
        logger.info(f"📊 {regime_type}: {len(all_regime_data)} datasets loaded")
        return all_regime_data
    
    def calculate_strategy_return(self, data: pd.DataFrame, indicator: str, params: Dict) -> float:
        """Calculate strategy return for given parameters"""
        try:
            if indicator == 'Supertrend':
                result = calculate_supertrend(data, params['st_period'], params['st_multiplier'])
                if result is None:
                    return -100.0
                signals = result['buy_signal'].astype(int) - result['sell_signal'].astype(int)
                
            elif indicator == 'MACD':
                result = calculate_macd_custom(data, params['macd_fast'], params['macd_slow'], params['macd_signal'])
                if result is None:
                    return -100.0
                
                signals = pd.Series(0, index=data.index)
                signals[(result['macd'] > result['signal']) & 
                       (result['macd'].shift(1) <= result['signal'].shift(1))] = 1
                signals[(result['macd'] < result['signal']) & 
                       (result['macd'].shift(1) >= result['signal'].shift(1))] = -1
                
            elif indicator == 'ADX':
                result = calculate_adx_di(data, params['adx_period'], params['adx_threshold'])
                if result is None:
                    return -100.0
                
                signals = pd.Series(0, index=data.index)
                buy_condition = (result['di_bullish_cross'] & 
                               (result['adx'] > params['adx_threshold']))
                sell_condition = (result['di_bearish_cross'] | 
                                (result['adx'] < params.get('adx_exit_threshold', params['adx_threshold'] * 0.8)))
                signals[buy_condition] = 1
                signals[sell_condition] = -1
                
            elif indicator == 'WaveTrend':
                result = calculate_wavetrend(data, params['wt_n1'], params['wt_n2'])
                if result is None:
                    return -100.0
                
                signals = pd.Series(0, index=data.index)
                buy_condition = ((result['wt1'] > result['wt2']) & 
                               (result['wt1'].shift(1) <= result['wt2'].shift(1)) &
                               (result['wt1'] < params['wt_oversold']))
                sell_condition = ((result['wt1'] < result['wt2']) & 
                                (result['wt1'].shift(1) >= result['wt2'].shift(1)) &
                                (result['wt1'] > params['wt_overbought']))
                signals[buy_condition] = 1
                signals[sell_condition] = -1
                
            else:
                return -100.0
            
            # Calculate returns
            returns = data['close'].pct_change().fillna(0)
            position = 0
            strategy_returns = []
            trades = 0
            
            for i in range(len(signals)):
                if signals.iloc[i] == 1 and position == 0:
                    position = 1
                    trades += 1
                elif signals.iloc[i] == -1 and position == 1:
                    position = 0
                
                if i + 1 < len(returns):
                    strategy_returns.append(position * returns.iloc[i + 1])
            
            if len(strategy_returns) == 0 or trades == 0:
                return -100.0
            
            # Calculate total return
            total_return = (1 + pd.Series(strategy_returns)).prod() - 1
            
            # Penalize if too few trades (less than 1 trade per month)
            min_trades = max(1, len(data) // 30)
            if trades < min_trades:
                total_return *= 0.5  # 50% penalty
            
            return total_return * 100
            
        except Exception as e:
            logger.error(f"Error calculating {indicator} return: {e}")
            return -100.0
    
    def objective_supertrend(self, trial, regime_data: List[pd.DataFrame]) -> float:
        """Optuna objective for Supertrend"""
        params = {
            'st_period': trial.suggest_int('st_period', 3, 20),
            'st_multiplier': trial.suggest_float('st_multiplier', 0.5, 5.0)
        }
        
        returns = []
        for data in regime_data:
            ret = self.calculate_strategy_return(data, 'Supertrend', params)
            if ret > -100:  # Valid return
                returns.append(ret)
        
        if len(returns) == 0:
            return -100.0
        
        # Multi-objective: mean return + consistency (negative std)
        mean_return = np.mean(returns)
        consistency = -np.std(returns) / 10  # Normalize std penalty
        
        return mean_return + consistency
    
    def objective_macd(self, trial, regime_data: List[pd.DataFrame]) -> float:
        """Optuna objective for MACD"""
        params = {
            'macd_fast': trial.suggest_int('macd_fast', 5, 20),
            'macd_slow': trial.suggest_int('macd_slow', 15, 50),
            'macd_signal': trial.suggest_int('macd_signal', 3, 15)
        }
        
        # Ensure fast < slow
        if params['macd_fast'] >= params['macd_slow']:
            return -100.0
        
        returns = []
        for data in regime_data:
            ret = self.calculate_strategy_return(data, 'MACD', params)
            if ret > -100:
                returns.append(ret)
        
        if len(returns) == 0:
            return -100.0
        
        mean_return = np.mean(returns)
        consistency = -np.std(returns) / 10
        
        return mean_return + consistency
    
    def objective_adx(self, trial, regime_data: List[pd.DataFrame]) -> float:
        """Optuna objective for ADX"""
        params = {
            'adx_period': trial.suggest_int('adx_period', 5, 25),
            'adx_threshold': trial.suggest_float('adx_threshold', 10.0, 40.0),
            'adx_exit_threshold': trial.suggest_float('adx_exit_threshold', 5.0, 30.0)
        }
        
        # Ensure exit < threshold
        if params['adx_exit_threshold'] >= params['adx_threshold']:
            return -100.0
        
        returns = []
        for data in regime_data:
            ret = self.calculate_strategy_return(data, 'ADX', params)
            if ret > -100:
                returns.append(ret)
        
        if len(returns) == 0:
            return -100.0
        
        mean_return = np.mean(returns)
        consistency = -np.std(returns) / 10
        
        return mean_return + consistency
    
    def objective_wavetrend(self, trial, regime_data: List[pd.DataFrame]) -> float:
        """Optuna objective for WaveTrend"""
        params = {
            'wt_n1': trial.suggest_int('wt_n1', 5, 20),
            'wt_n2': trial.suggest_int('wt_n2', 10, 30),
            'wt_overbought': trial.suggest_int('wt_overbought', 50, 80),
            'wt_oversold': trial.suggest_int('wt_oversold', -80, -30)
        }
        
        returns = []
        for data in regime_data:
            ret = self.calculate_strategy_return(data, 'WaveTrend', params)
            if ret > -100:
                returns.append(ret)
        
        if len(returns) == 0:
            return -100.0
        
        mean_return = np.mean(returns)
        consistency = -np.std(returns) / 10
        
        return mean_return + consistency
    
    def optimize_indicator_for_regime(self, indicator: str, regime_type: str, n_trials: int = 100):
        """Optimize specific indicator for specific regime"""
        logger.info(f"🎯 Optimizing {indicator} for {regime_type} market...")
        
        # Get regime data
        regime_data = self.get_regime_data(regime_type)
        if len(regime_data) == 0:
            logger.warning(f"⚠️  No data available for {regime_type}")
            return None
        
        # Create study
        study_name = f"{indicator}_{regime_type}"
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Choose objective function
        if indicator == 'Supertrend':
            objective = lambda trial: self.objective_supertrend(trial, regime_data)
        elif indicator == 'MACD':
            objective = lambda trial: self.objective_macd(trial, regime_data)
        elif indicator == 'ADX':
            objective = lambda trial: self.objective_adx(trial, regime_data)
        elif indicator == 'WaveTrend':
            objective = lambda trial: self.objective_wavetrend(trial, regime_data)
        else:
            logger.error(f"❌ Unknown indicator: {indicator}")
            return None
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"✅ {indicator} for {regime_type}: Best value = {best_value:.2f}%")
        logger.info(f"   Best params: {best_params}")
        
        return {
            'params': best_params,
            'value': best_value,
            'n_trials': len(study.trials),
            'regime_data_count': len(regime_data)
        }
    
    def optimize_all_regimes(self, n_trials: int = 50):
        """Optimize all indicators for all market regimes"""
        logger.info("🚀 Starting regime-based optimization...")
        
        indicators = ['Supertrend', 'MACD', 'ADX', 'WaveTrend']
        regimes = ['bull_market', 'bear_market', 'sideways_market']
        
        results = {}
        
        for regime in regimes:
            logger.info(f"\n📊 Optimizing for {regime}...")
            results[regime] = {}
            
            for indicator in indicators:
                result = self.optimize_indicator_for_regime(indicator, regime, n_trials)
                if result:
                    results[regime][indicator] = result
        
        self.optimized_params = results
        return results
    
    def save_optimized_params(self):
        """Save optimized parameters"""
        output = {
            'optimization_date': datetime.now().isoformat(),
            'method': 'regime_based_optuna',
            'regimes_used': list(self.regimes.keys()),
            'optimized_parameters': self.optimized_params
        }
        
        with open('regime_optimized_parameters.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info("💾 Optimized parameters saved to: regime_optimized_parameters.json")
    
    def compare_with_universal(self):
        """Compare regime-based params with universal params"""
        logger.info("\n📊 REGIME-BASED vs UNIVERSAL PARAMETERS")
        logger.info("=" * 70)
        
        try:
            with open('universal_optimal_parameters_complete.json', 'r') as f:
                universal_params = json.load(f)
        except FileNotFoundError:
            logger.warning("⚠️  Universal parameters file not found")
            return
        
        for regime, indicators in self.optimized_params.items():
            logger.info(f"\n🎯 {regime.upper()}:")
            
            for indicator, result in indicators.items():
                universal_result = universal_params.get(indicator, {})
                universal_return = universal_result.get('avg_return', 0)
                regime_return = result['value']
                
                improvement = regime_return - universal_return if universal_return != 0 else regime_return
                
                logger.info(f"   {indicator}:")
                logger.info(f"     Universal: {universal_return:.2f}%")
                logger.info(f"     {regime}: {regime_return:.2f}%")
                logger.info(f"     Improvement: {improvement:+.2f}%")
    
    def get_current_params(self, method: str = 'manual') -> Dict:
        """Get parameters for current market regime"""
        # Load current regime from market analysis
        try:
            with open('market_regimes_analysis.json', 'r') as f:
                regime_data = json.load(f)
            current_regime = regime_data['current_regimes'][method]
        except:
            current_regime = 'sideways_market'  # Default
        
        logger.info(f"🎯 Current regime ({method}): {current_regime}")
        
        if current_regime in self.optimized_params:
            return self.optimized_params[current_regime]
        else:
            logger.warning(f"⚠️  No optimized params for {current_regime}, using sideways_market")
            return self.optimized_params.get('sideways_market', {})
    
    def run_full_optimization(self, n_trials: int = 50):
        """Run complete regime-based optimization"""
        logger.info("🚀 Starting complete regime-based optimization...")
        
        # Optimize all regimes
        self.optimize_all_regimes(n_trials)
        
        # Save results
        self.save_optimized_params()
        
        # Compare with universal
        self.compare_with_universal()
        
        # Show current regime recommendations
        current_params = self.get_current_params()
        
        logger.info("\n🎯 CURRENT REGIME RECOMMENDATIONS:")
        logger.info("=" * 50)
        for indicator, result in current_params.items():
            logger.info(f"{indicator}: {result['params']} (Return: {result['value']:.2f}%)")


def main():
    """Main function"""
    optimizer = RegimeBasedOptimizer()
    optimizer.run_full_optimization(n_trials=30)  # Reduced for speed


if __name__ == "__main__":
    main()