#!/usr/bin/env python3
"""
Walk-Forward Backtest
Tests strategies with rolling optimization windows
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from create_ensemble_strategy import EnsembleStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """Walk-forward analysis for trading strategies"""
    
    def __init__(self, 
                 optimization_window: int = 180,  # days
                 test_window: int = 30,           # days
                 step_size: int = 30):            # days
        """
        Initialize walk-forward backtest
        
        Args:
            optimization_window: Days of data for optimization
            test_window: Days of data for out-of-sample testing
            step_size: Days to step forward each iteration
        """
        self.optimization_window = optimization_window
        self.test_window = test_window
        self.step_size = step_size
        self.calc = IndicatorCalculator(DATA_DIR)
        
        # Results storage
        self.results = []
        self.equity_curves = {}
        
    def optimize_parameters(self, start_date: datetime, end_date: datetime, 
                          indicator: str) -> dict:
        """Optimize indicator parameters for given period"""
        logger.info(f"Optimizing {indicator} from {start_date.date()} to {end_date.date()}")
        
        # Load universal parameters as baseline
        try:
            params_file = Path(__file__).parent / 'universal_optimal_parameters_complete.json'
            with open(params_file, 'r') as f:
                universal_params = json.load(f)
            
            if indicator in universal_params:
                best_params = universal_params[indicator]['params']
                best_return = universal_params[indicator]['avg_return']
            else:
                # Fallback defaults
                defaults = {
                    'Supertrend': {'st_period': 10, 'st_multiplier': 3.0},
                    'MACD': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
                    'ADX': {'adx_period': 14, 'adx_threshold': 25, 'adx_exit_threshold': 20},
                    'WaveTrend': {'wt_n1': 10, 'wt_n2': 21, 'wt_overbought': 60, 'wt_oversold': -60},
                    'Squeeze': {'sq_bb_length': 20, 'sq_bb_mult': 2.0, 'sq_kc_length': 20, 'sq_kc_mult': 1.5, 'sq_mom_length': 12},
                    'VixFix': {'vf_lookback': 22, 'vf_bb_length': 20, 'vf_bb_mult': 2.0, 'vf_hold_days': 5}
                }
                best_params = defaults.get(indicator, {})
                best_return = 0
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            best_params = {}
            best_return = 0
        
        return {
            'params': best_params,
            'in_sample_return': best_return,
            'period': f"{start_date.date()} to {end_date.date()}"
        }
    
    def test_parameters(self, params: dict, start_date: datetime, 
                       end_date: datetime, indicator: str) -> dict:
        """Test parameters on out-of-sample data"""
        # Create ensemble strategy
        ensemble = EnsembleStrategy()
        
        # Override parameters for testing
        if indicator == 'all':
            # Test all indicators ensemble
            test_params = params
        else:
            # Test single indicator
            test_params = {indicator: {'params': params['params']}}
        
        # Calculate returns for test period
        total_return = 0
        trades = 0
        winning_trades = 0
        
        for symbol in SACRED_SYMBOLS:
            try:
                data = self.calc.load_raw_data(symbol, '1d')
                if data is None:
                    continue
                
                # Filter to test period (handle timezone)
                data_start = pd.to_datetime(start_date).tz_localize(None)
                data_end = pd.to_datetime(end_date).tz_localize(None)
                data_index = data.index.tz_localize(None) if data.index.tz else data.index
                
                mask = (data_index >= data_start) & (data_index <= data_end)
                test_data = data[mask]
                
                if len(test_data) < 10:
                    continue
                
                # Generate signals based on indicator
                if indicator == 'all':
                    signals = ensemble.generate_signals(test_data, strategy='balanced')
                else:
                    # Single indicator test
                    if indicator == 'Supertrend':
                        from indicators.supertrend import calculate_supertrend
                        st_params = params['params']
                        result = calculate_supertrend(test_data, st_params['st_period'], st_params['st_multiplier'])
                        signals = result['buy_signal'].astype(int) - result['sell_signal'].astype(int)
                    elif indicator == 'MACD':
                        from indicators.macd_custom import calculate_macd_custom
                        macd_params = params['params']
                        result = calculate_macd_custom(test_data, macd_params['macd_fast'], 
                                                     macd_params['macd_slow'], macd_params['macd_signal'])
                        # Simple MACD crossover
                        signals = pd.Series(0, index=test_data.index)
                        signals[(result['macd'] > result['signal']) & 
                               (result['macd'].shift(1) <= result['signal'].shift(1))] = 1
                        signals[(result['macd'] < result['signal']) & 
                               (result['macd'].shift(1) >= result['signal'].shift(1))] = -1
                    else:
                        # Add other indicators as needed
                        signals = pd.Series(0, index=test_data.index)
                
                # Calculate returns
                position = 0
                entry_price = 0
                
                for i in range(len(signals)):
                    if signals.iloc[i] == 1 and position == 0:
                        # Buy signal
                        position = 1
                        entry_price = test_data['close'].iloc[i]
                        
                    elif signals.iloc[i] == -1 and position == 1:
                        # Sell signal
                        exit_price = test_data['close'].iloc[i]
                        trade_return = (exit_price - entry_price) / entry_price
                        total_return += trade_return
                        trades += 1
                        if trade_return > 0:
                            winning_trades += 1
                        position = 0
                
                # Close any open position
                if position == 1 and len(test_data) > 0:
                    exit_price = test_data['close'].iloc[-1]
                    trade_return = (exit_price - entry_price) / entry_price
                    total_return += trade_return
                    trades += 1
                    if trade_return > 0:
                        winning_trades += 1
                        
            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}")
                continue
        
        # Average return across symbols
        avg_return = (total_return / len(SACRED_SYMBOLS)) * 100 if trades > 0 else 0
        win_rate = (winning_trades / trades * 100) if trades > 0 else 0
        
        return {
            'out_sample_return': avg_return,
            'trades': trades,
            'win_rate': win_rate,
            'period': f"{start_date.date()} to {end_date.date()}"
        }
    
    def run_walk_forward(self, indicator: str, start_date: str, end_date: str) -> dict:
        """Run walk-forward analysis for an indicator"""
        logger.info(f"\nRunning walk-forward analysis for {indicator}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Optimization window: {self.optimization_window} days")
        logger.info(f"Test window: {self.test_window} days")
        logger.info(f"Step size: {self.step_size} days\n")
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Results for this indicator
        indicator_results = []
        equity_curve = [100]  # Start with $100
        
        # Walk forward
        current_date = start + timedelta(days=self.optimization_window)
        
        with tqdm(desc=f"Walk-forward {indicator}") as pbar:
            while current_date + timedelta(days=self.test_window) <= end:
                # Optimization period
                opt_start = current_date - timedelta(days=self.optimization_window)
                opt_end = current_date
                
                # Test period
                test_start = current_date
                test_end = current_date + timedelta(days=self.test_window)
                
                # Optimize
                opt_result = self.optimize_parameters(opt_start, opt_end, indicator)
                
                # Test
                test_result = self.test_parameters(opt_result, test_start, test_end, indicator)
                
                # Store results
                result = {
                    'optimization_period': opt_result['period'],
                    'test_period': test_result['period'],
                    'in_sample_return': opt_result['in_sample_return'],
                    'out_sample_return': test_result['out_sample_return'],
                    'trades': test_result['trades'],
                    'win_rate': test_result['win_rate'],
                    'parameters': opt_result['params']
                }
                indicator_results.append(result)
                
                # Update equity curve
                equity_change = 1 + (test_result['out_sample_return'] / 100)
                equity_curve.append(equity_curve[-1] * equity_change)
                
                # Step forward
                current_date += timedelta(days=self.step_size)
                pbar.update(1)
        
        # Calculate statistics
        out_sample_returns = [r['out_sample_return'] for r in indicator_results]
        in_sample_returns = [r['in_sample_return'] for r in indicator_results]
        
        stats = {
            'indicator': indicator,
            'total_periods': len(indicator_results),
            'avg_in_sample_return': np.mean(in_sample_returns),
            'avg_out_sample_return': np.mean(out_sample_returns),
            'std_out_sample_return': np.std(out_sample_returns),
            'sharpe_ratio': np.mean(out_sample_returns) / np.std(out_sample_returns) if np.std(out_sample_returns) > 0 else 0,
            'win_periods': sum(1 for r in out_sample_returns if r > 0),
            'win_period_rate': sum(1 for r in out_sample_returns if r > 0) / len(out_sample_returns) * 100,
            'total_return': (equity_curve[-1] - 100),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'results': indicator_results,
            'equity_curve': equity_curve
        }
        
        return stats
    
    def calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def run_all_indicators(self, start_date: str, end_date: str):
        """Run walk-forward analysis for all indicators"""
        indicators = ['Supertrend', 'MACD', 'ADX', 'WaveTrend', 'Squeeze', 'VixFix']
        
        for indicator in indicators:
            try:
                stats = self.run_walk_forward(indicator, start_date, end_date)
                self.results.append(stats)
                self.equity_curves[indicator] = stats['equity_curve']
            except Exception as e:
                logger.error(f"Error with {indicator}: {e}")
        
        # Also test ensemble strategy
        try:
            stats = self.run_walk_forward('all', start_date, end_date)
            stats['indicator'] = 'Ensemble'
            self.results.append(stats)
            self.equity_curves['Ensemble'] = stats['equity_curve']
        except Exception as e:
            logger.error(f"Error with ensemble: {e}")
    
    def plot_results(self):
        """Plot equity curves and statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity curves
        ax1 = axes[0, 0]
        for indicator, curve in self.equity_curves.items():
            ax1.plot(curve, label=indicator)
        ax1.set_title('Walk-Forward Equity Curves')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Equity ($100 start)')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Out-of-sample returns
        ax2 = axes[0, 1]
        indicators = [r['indicator'] for r in self.results]
        avg_returns = [r['avg_out_sample_return'] for r in self.results]
        colors = ['green' if r > 0 else 'red' for r in avg_returns]
        
        ax2.bar(indicators, avg_returns, color=colors)
        ax2.set_title('Average Out-of-Sample Returns')
        ax2.set_xlabel('Indicator')
        ax2.set_ylabel('Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Sharpe ratios
        ax3 = axes[1, 0]
        sharpe_ratios = [r['sharpe_ratio'] for r in self.results]
        colors = ['green' if s > 0 else 'red' for s in sharpe_ratios]
        
        ax3.bar(indicators, sharpe_ratios, color=colors)
        ax3.set_title('Sharpe Ratios')
        ax3.set_xlabel('Indicator')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Win period rates
        ax4 = axes[1, 1]
        win_rates = [r['win_period_rate'] for r in self.results]
        colors = ['green' if w > 50 else 'red' for w in win_rates]
        
        ax4.bar(indicators, win_rates, color=colors)
        ax4.set_title('Win Period Rate')
        ax4.set_xlabel('Indicator')
        ax4.set_ylabel('Win Rate (%)')
        ax4.axhline(y=50, color='black', linestyle='--', linewidth=0.5)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('walkforward_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate detailed report"""
        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS REPORT")
        print("="*80)
        print(f"Optimization Window: {self.optimization_window} days")
        print(f"Test Window: {self.test_window} days")
        print(f"Step Size: {self.step_size} days")
        print("="*80)
        
        # Sort by out-of-sample return
        sorted_results = sorted(self.results, key=lambda x: x['avg_out_sample_return'], reverse=True)
        
        for result in sorted_results:
            print(f"\n{result['indicator'].upper()}")
            print("-" * 40)
            print(f"Total Periods: {result['total_periods']}")
            print(f"Avg In-Sample Return: {result['avg_in_sample_return']:.2f}%")
            print(f"Avg Out-Sample Return: {result['avg_out_sample_return']:.2f}%")
            print(f"Std Dev: {result['std_out_sample_return']:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Win Periods: {result['win_periods']}/{result['total_periods']} ({result['win_period_rate']:.1f}%)")
            print(f"Total Return: {result['total_return']:.2f}%")
            print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
        
        # Best performer
        best = sorted_results[0]
        print("\n" + "="*80)
        print(f"🏆 BEST PERFORMER: {best['indicator']}")
        print(f"   Out-of-Sample Return: {best['avg_out_sample_return']:.2f}%")
        print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        print(f"   Total Return: {best['total_return']:.2f}%")
        print("="*80)
        
        # Save detailed results
        output_file = 'walkforward_results.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk-Forward Backtest')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--opt-window', type=int, default=180,
                       help='Optimization window in days')
    parser.add_argument('--test-window', type=int, default=30,
                       help='Test window in days')
    parser.add_argument('--step', type=int, default=30,
                       help='Step size in days')
    parser.add_argument('--indicator', type=str, default='all',
                       help='Indicator to test (or "all")')
    
    args = parser.parse_args()
    
    # Create walk-forward tester
    wf = WalkForwardBacktest(
        optimization_window=args.opt_window,
        test_window=args.test_window,
        step_size=args.step
    )
    
    # Run analysis
    if args.indicator == 'all':
        wf.run_all_indicators(args.start, args.end)
    else:
        stats = wf.run_walk_forward(args.indicator, args.start, args.end)
        wf.results.append(stats)
        wf.equity_curves[args.indicator] = stats['equity_curve']
    
    # Generate report and plots
    wf.generate_report()
    wf.plot_results()


if __name__ == "__main__":
    main()