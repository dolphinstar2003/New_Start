#!/usr/bin/env python3
"""
Portfolio Walk-Forward Backtest System
Creates and manages multiple portfolios using trained indicator parameters
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
from indicators.supertrend import calculate_supertrend
from indicators.macd_custom import calculate_macd_custom
from indicators.adx_di import calculate_adx_di
from indicators.wavetrend import calculate_wavetrend
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages multiple trading portfolios with different strategies"""
    
    def __init__(self, initial_capital: float = 50000):
        self.initial_capital = initial_capital
        self.portfolios = {
            'aggressive': {
                'name': 'Aggressive (Supertrend Only)',
                'description': 'High return, high risk - Supertrend only',
                'indicators': ['Supertrend'],
                'capital': initial_capital,
                'positions': {},
                'cash': initial_capital,
                'equity_curve': [initial_capital],
                'trades': [],
                'max_positions': 10
            },
            'balanced': {
                'name': 'Balanced (Multi-Indicator)',
                'description': 'Balanced approach - ADX + WaveTrend confirmation',
                'indicators': ['ADX', 'WaveTrend'],
                'capital': initial_capital,
                'positions': {},
                'cash': initial_capital,
                'equity_curve': [initial_capital],
                'trades': [],
                'max_positions': 8
            },
            'conservative': {
                'name': 'Conservative (MACD + ADX)',
                'description': 'Low risk, steady gains - MACD + ADX',
                'indicators': ['MACD', 'ADX'],
                'capital': initial_capital,
                'positions': {},
                'cash': initial_capital,
                'equity_curve': [initial_capital],
                'trades': [],
                'max_positions': 5
            }
        }
        
        # Load trained parameters
        with open('universal_optimal_parameters_complete.json', 'r') as f:
            self.trained_params = json.load(f)
        
        self.calc = IndicatorCalculator(DATA_DIR)
        self.current_date = None
        
    def calculate_portfolio_value(self, portfolio_name: str, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        portfolio = self.portfolios[portfolio_name]
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in prices:
                total_value += shares * prices[symbol]
                
        return total_value
    
    def get_position_size(self, portfolio_name: str, price: float) -> int:
        """Calculate position size based on portfolio strategy"""
        portfolio = self.portfolios[portfolio_name]
        max_positions = portfolio['max_positions']
        
        # Use equal weight allocation
        target_allocation = portfolio['cash'] / max_positions
        shares = int(target_allocation / price)
        
        return max(shares, 0)
    
    def generate_signals(self, symbol: str, data: pd.DataFrame, portfolio_name: str) -> pd.Series:
        """Generate trading signals for a portfolio strategy"""
        portfolio = self.portfolios[portfolio_name]
        indicators = portfolio['indicators']
        
        signals = pd.Series(0, index=data.index)
        
        if portfolio_name == 'aggressive':
            # Supertrend only
            st_params = self.trained_params['Supertrend']['params']
            result = calculate_supertrend(data, st_params['st_period'], st_params['st_multiplier'])
            if result is not None:
                signals = result['buy_signal'].astype(int) - result['sell_signal'].astype(int)
                
        elif portfolio_name == 'balanced':
            # ADX + WaveTrend confirmation
            adx_signals = pd.Series(0, index=data.index)
            wt_signals = pd.Series(0, index=data.index)
            
            # ADX signals
            adx_params = self.trained_params['ADX']['params']
            adx_result = calculate_adx_di(data, adx_params['adx_period'], adx_params['adx_threshold'])
            if adx_result is not None:
                buy_condition = (adx_result['di_bullish_cross'] & 
                               (adx_result['adx'] > adx_params['adx_threshold']))
                sell_condition = (adx_result['di_bearish_cross'] | 
                                (adx_result['adx'] < adx_params['adx_exit_threshold']))
                adx_signals[buy_condition] = 1
                adx_signals[sell_condition] = -1
            
            # WaveTrend signals
            wt_params = self.trained_params['WaveTrend']['params']
            wt_result = calculate_wavetrend(data, wt_params['wt_n1'], wt_params['wt_n2'])
            if wt_result is not None:
                buy_condition = ((wt_result['wt1'] > wt_result['wt2']) & 
                               (wt_result['wt1'].shift(1) <= wt_result['wt2'].shift(1)) &
                               (wt_result['wt1'] < wt_params['wt_oversold']))
                sell_condition = ((wt_result['wt1'] < wt_result['wt2']) & 
                                (wt_result['wt1'].shift(1) >= wt_result['wt2'].shift(1)) &
                                (wt_result['wt1'] > wt_params['wt_overbought']))
                wt_signals[buy_condition] = 1
                wt_signals[sell_condition] = -1
            
            # Require both signals to agree (conservative approach)
            signals[(adx_signals == 1) & (wt_signals == 1)] = 1
            signals[(adx_signals == -1) | (wt_signals == -1)] = -1
            
        elif portfolio_name == 'conservative':
            # MACD + ADX confirmation
            macd_signals = pd.Series(0, index=data.index)
            adx_signals = pd.Series(0, index=data.index)
            
            # MACD signals
            macd_params = self.trained_params['MACD']['params']
            macd_result = calculate_macd_custom(data, macd_params['macd_fast'], 
                                             macd_params['macd_slow'], macd_params['macd_signal'])
            if macd_result is not None:
                buy_condition = ((macd_result['macd'] > macd_result['signal']) & 
                               (macd_result['macd'].shift(1) <= macd_result['signal'].shift(1)))
                sell_condition = ((macd_result['macd'] < macd_result['signal']) & 
                                (macd_result['macd'].shift(1) >= macd_result['signal'].shift(1)))
                macd_signals[buy_condition] = 1
                macd_signals[sell_condition] = -1
            
            # ADX trend strength filter
            adx_params = self.trained_params['ADX']['params']
            adx_result = calculate_adx_di(data, adx_params['adx_period'], adx_params['adx_threshold'])
            if adx_result is not None:
                # Only trade when ADX shows strong trend
                strong_trend = adx_result['adx'] > adx_params['adx_threshold']
                signals[(macd_signals == 1) & strong_trend] = 1
                signals[(macd_signals == -1)] = -1  # Exit on any MACD sell signal
        
        return signals
    
    def execute_trades(self, portfolio_name: str, symbol: str, signal: int, price: float, date: datetime):
        """Execute trades based on signals"""
        portfolio = self.portfolios[portfolio_name]
        
        if signal == 1:  # Buy signal
            # Check if we already have this position
            if symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                return
            
            # Check if we have room for more positions
            current_positions = len([s for s, shares in portfolio['positions'].items() if shares > 0])
            if current_positions >= portfolio['max_positions']:
                return
            
            # Calculate position size
            shares = self.get_position_size(portfolio_name, price)
            cost = shares * price
            
            if cost <= portfolio['cash'] and shares > 0:
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = shares
                
                # Record trade
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'value': cost
                })
                
        elif signal == -1:  # Sell signal
            if symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                shares = portfolio['positions'][symbol]
                proceeds = shares * price
                
                portfolio['cash'] += proceeds
                portfolio['positions'][symbol] = 0
                
                # Record trade
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares,
                    'price': price,
                    'value': proceeds
                })
    
    def update_portfolios(self, date: datetime, prices: Dict[str, float], signals: Dict[str, Dict[str, int]]):
        """Update all portfolios with new prices and signals"""
        self.current_date = date
        
        for portfolio_name in self.portfolios.keys():
            # Execute trades based on signals
            if portfolio_name in signals:
                for symbol, signal in signals[portfolio_name].items():
                    if symbol in prices:
                        self.execute_trades(portfolio_name, symbol, signal, prices[symbol], date)
            
            # Update equity curve
            portfolio_value = self.calculate_portfolio_value(portfolio_name, prices)
            self.portfolios[portfolio_name]['equity_curve'].append(portfolio_value)
    
    def get_portfolio_stats(self, portfolio_name: str) -> Dict:
        """Calculate portfolio performance statistics"""
        portfolio = self.portfolios[portfolio_name]
        equity_curve = portfolio['equity_curve']
        
        if len(equity_curve) < 2:
            return {}
        
        # Calculate returns
        returns = pd.Series(equity_curve).pct_change().dropna()
        total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Calculate max drawdown
        peak = pd.Series(equity_curve).expanding().max()
        drawdown = (pd.Series(equity_curve) - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        trades = portfolio['trades']
        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell']
        
        # Calculate win rate from completed trades
        completed_trades = []
        for sell_trade in sell_trades:
            # Find corresponding buy trade
            for buy_trade in reversed(buy_trades):
                if (buy_trade['symbol'] == sell_trade['symbol'] and 
                    buy_trade['date'] < sell_trade['date']):
                    trade_return = (sell_trade['price'] - buy_trade['price']) / buy_trade['price']
                    completed_trades.append(trade_return)
                    break
        
        win_rate = sum(1 for r in completed_trades if r > 0) / len(completed_trades) * 100 if completed_trades else 0
        
        current_positions = len([s for s, shares in portfolio['positions'].items() if shares > 0])
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(buy_trades),
            'completed_trades': len(completed_trades),
            'win_rate': win_rate,
            'current_positions': current_positions,
            'current_value': equity_curve[-1],
            'current_cash': portfolio['cash']
        }


class PortfolioWalkForward:
    """Walk-forward analysis for multiple portfolios"""
    
    def __init__(self, optimization_window: int = 180, test_window: int = 30, step_size: int = 30):
        self.optimization_window = optimization_window
        self.test_window = test_window
        self.step_size = step_size
        self.portfolio_manager = PortfolioManager()
        self.results = {}
        
    def run_walk_forward(self, start_date: str, end_date: str):
        """Run walk-forward analysis for all portfolios"""
        logger.info(f"🚀 Starting Portfolio Walk-Forward Analysis")
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info(f"⚙️  Optimization window: {self.optimization_window} days")
        logger.info(f"🧪 Test window: {self.test_window} days")
        logger.info(f"👣 Step size: {self.step_size} days")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Walk through time periods
        current_date = start + timedelta(days=self.optimization_window)
        
        with tqdm(desc="Portfolio Walk-Forward") as pbar:
            while current_date + timedelta(days=self.test_window) <= end:
                test_start = current_date
                test_end = current_date + timedelta(days=self.test_window)
                
                # Get prices and signals for test period
                prices_dict = {}
                signals_dict = {'aggressive': {}, 'balanced': {}, 'conservative': {}}
                
                for symbol in SACRED_SYMBOLS:
                    try:
                        data = self.portfolio_manager.calc.load_raw_data(symbol, '1d')
                        if data is None:
                            continue
                        
                        # Filter to test period
                        data_start = pd.to_datetime(test_start).tz_localize(None)
                        data_end = pd.to_datetime(test_end).tz_localize(None)
                        data_index = data.index.tz_localize(None) if data.index.tz else data.index
                        
                        mask = (data_index >= data_start) & (data_index <= data_end)
                        test_data = data[mask]
                        
                        if len(test_data) < 5:
                            continue
                        
                        # Get current price (last available)
                        current_price = test_data['close'].iloc[-1]
                        prices_dict[symbol] = current_price
                        
                        # Generate signals for each portfolio
                        for portfolio_name in ['aggressive', 'balanced', 'conservative']:
                            signals = self.portfolio_manager.generate_signals(symbol, test_data, portfolio_name)
                            # Use the last signal in the period
                            if len(signals) > 0:
                                signals_dict[portfolio_name][symbol] = signals.iloc[-1]
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update portfolios
                self.portfolio_manager.update_portfolios(test_end, prices_dict, signals_dict)
                
                # Step forward
                current_date += timedelta(days=self.step_size)
                pbar.update(1)
        
        # Calculate final statistics
        self.calculate_final_results()
        
    def calculate_final_results(self):
        """Calculate final performance statistics"""
        logger.info("\n📊 PORTFOLIO PERFORMANCE ANALYSIS")
        logger.info("=" * 80)
        
        for portfolio_name, portfolio in self.portfolio_manager.portfolios.items():
            stats = self.portfolio_manager.get_portfolio_stats(portfolio_name)
            self.results[portfolio_name] = stats
            
            print(f"\n🎯 {portfolio['name']}")
            print(f"   {portfolio['description']}")
            print("-" * 60)
            
            if stats:
                print(f"   💰 Total Return: {stats['total_return']:.2f}%")
                print(f"   📈 Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
                print(f"   📉 Max Drawdown: {stats['max_drawdown']:.2f}%")
                print(f"   🔄 Total Trades: {stats['total_trades']}")
                print(f"   ✅ Win Rate: {stats['win_rate']:.1f}%")
                print(f"   🏠 Current Positions: {stats['current_positions']}")
                print(f"   💵 Current Value: ${stats['current_value']:,.2f}")
                print(f"   💸 Current Cash: ${stats['current_cash']:,.2f}")
            else:
                print("   ⚠️  No data available")
        
        # Find best performer
        best_portfolio = max(self.results.keys(), 
                           key=lambda x: self.results[x].get('total_return', -float('inf')))
        
        if self.results[best_portfolio]:
            print("\n" + "=" * 80)
            print(f"🏆 BEST PERFORMER: {self.portfolio_manager.portfolios[best_portfolio]['name']}")
            print(f"   💰 Return: {self.results[best_portfolio]['total_return']:.2f}%")
            print(f"   📈 Sharpe: {self.results[best_portfolio]['sharpe_ratio']:.2f}")
            print("=" * 80)
    
    def plot_results(self):
        """Plot portfolio equity curves"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. Equity curves
        for portfolio_name, portfolio in self.portfolio_manager.portfolios.items():
            equity_curve = portfolio['equity_curve']
            if len(equity_curve) > 1:
                ax1.plot(equity_curve, label=portfolio['name'])
        
        ax1.set_title('Portfolio Equity Curves')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Performance comparison
        portfolios = list(self.results.keys())
        returns = [self.results[p].get('total_return', 0) for p in portfolios]
        colors = ['green' if r > 0 else 'red' for r in returns]
        
        ax2.bar(portfolios, returns, color=colors)
        ax2.set_title('Portfolio Returns Comparison')
        ax2.set_xlabel('Portfolio')
        ax2.set_ylabel('Total Return (%)')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('portfolio_walkforward_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save detailed results to JSON"""
        output = {
            'analysis_info': {
                'optimization_window': self.optimization_window,
                'test_window': self.test_window,
                'step_size': self.step_size,
                'date_generated': datetime.now().isoformat()
            },
            'portfolios': {}
        }
        
        for portfolio_name, portfolio in self.portfolio_manager.portfolios.items():
            output['portfolios'][portfolio_name] = {
                'name': portfolio['name'],
                'description': portfolio['description'],
                'indicators': portfolio['indicators'],
                'stats': self.results.get(portfolio_name, {}),
                'equity_curve': portfolio['equity_curve'],
                'trades': portfolio['trades'][-20:] if portfolio['trades'] else [],  # Last 20 trades
                'final_positions': {k: v for k, v in portfolio['positions'].items() if v > 0}
            }
        
        with open('portfolio_walkforward_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info("💾 Results saved to: portfolio_walkforward_results.json")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Walk-Forward Backtest')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--opt-window', type=int, default=90,
                       help='Optimization window in days')
    parser.add_argument('--test-window', type=int, default=30,
                       help='Test window in days')
    parser.add_argument('--step', type=int, default=15,
                       help='Step size in days')
    
    args = parser.parse_args()
    
    # Create portfolio walk-forward tester
    pf_walkforward = PortfolioWalkForward(
        optimization_window=args.opt_window,
        test_window=args.test_window,
        step_size=args.step
    )
    
    # Run analysis
    pf_walkforward.run_walk_forward(args.start, args.end)
    
    # Generate plots and save results
    pf_walkforward.plot_results()
    pf_walkforward.save_results()


if __name__ == "__main__":
    main()