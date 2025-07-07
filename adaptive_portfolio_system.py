#!/usr/bin/env python3
"""
Adaptive Portfolio System
Automatically switches parameters based on current market regime
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


class AdaptivePortfolioSystem:
    """Portfolio system that adapts to market regimes"""
    
    def __init__(self, initial_capital: float = 50000):
        self.initial_capital = initial_capital
        self.calc = IndicatorCalculator(DATA_DIR)
        
        # Load regime-optimized parameters
        self.regime_params = self.load_regime_parameters()
        
        # Load market regimes
        self.market_regimes = self.load_market_regimes()
        
        # Portfolio configurations with regime adaptation
        self.portfolios = {
            'adaptive_aggressive': {
                'name': 'Adaptive Aggressive',
                'description': 'Uses best single indicator per regime',
                'strategy': 'single_best',
                'capital': initial_capital,
                'positions': {},
                'cash': initial_capital,
                'equity_curve': [initial_capital],
                'trades': [],
                'max_positions': 10,
                'regime_switches': []
            },
            'adaptive_balanced': {
                'name': 'Adaptive Balanced',
                'description': 'Combines top 2 indicators per regime',
                'strategy': 'dual_best',
                'capital': initial_capital,
                'positions': {},
                'cash': initial_capital,
                'equity_curve': [initial_capital],
                'trades': [],
                'max_positions': 8,
                'regime_switches': []
            },
            'adaptive_conservative': {
                'name': 'Adaptive Conservative',
                'description': 'Uses all indicators with regime weighting',
                'strategy': 'weighted_ensemble',
                'capital': initial_capital,
                'positions': {},
                'cash': initial_capital,
                'equity_curve': [initial_capital],
                'trades': [],
                'max_positions': 6,
                'regime_switches': []
            }
        }
        
        self.current_regime = 'sideways_market'
        self.regime_history = []
        
    def load_regime_parameters(self) -> Dict:
        """Load regime-optimized parameters"""
        try:
            with open('regime_optimized_parameters.json', 'r') as f:
                data = json.load(f)
            return data['optimized_parameters']
        except FileNotFoundError:
            logger.error("❌ Regime parameters not found. Run regime_based_optimizer.py first.")
            return {}
    
    def load_market_regimes(self) -> Dict:
        """Load market regime analysis"""
        try:
            with open('market_regimes_analysis.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error("❌ Market regimes not found. Run market_regime_analyzer.py first.")
            return {}
    
    def detect_current_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime based on recent data"""
        if len(data) < 50:
            return 'sideways_market'
        
        # Calculate recent volatility (VIX proxy)
        recent_returns = data['close'].pct_change().tail(20).dropna()
        if len(recent_returns) == 0:
            return 'sideways_market'
        
        volatility = recent_returns.std() * np.sqrt(252) * 100
        
        # Calculate trend (MA-based)
        price = data['close'].iloc[-1]
        ma_20 = data['close'].rolling(20).mean().iloc[-1]
        ma_50 = data['close'].rolling(50).mean().iloc[-1]
        
        # Regime classification
        if volatility > 35:  # High volatility
            regime = 'bear_market'
        elif volatility < 20 and price > ma_20 and ma_20 > ma_50:  # Low vol + uptrend
            regime = 'bull_market'
        else:  # Medium volatility or sideways
            regime = 'sideways_market'
        
        return regime
    
    def get_regime_indicators(self, regime: str, strategy: str) -> List[str]:
        """Get indicators to use for given regime and strategy"""
        regime_data = self.regime_params.get(regime, {})
        
        if not regime_data:
            return ['Supertrend']  # Fallback
        
        # Sort indicators by performance
        sorted_indicators = sorted(regime_data.items(), 
                                 key=lambda x: x[1]['value'], reverse=True)
        
        if strategy == 'single_best':
            return [sorted_indicators[0][0]]
        elif strategy == 'dual_best':
            return [ind[0] for ind in sorted_indicators[:2]]
        elif strategy == 'weighted_ensemble':
            return [ind[0] for ind in sorted_indicators]
        else:
            return ['Supertrend']
    
    def calculate_regime_signals(self, symbol: str, data: pd.DataFrame, 
                               indicators: List[str], regime: str, strategy: str) -> pd.Series:
        """Calculate signals using regime-specific parameters"""
        signals = pd.Series(0, index=data.index)
        
        if not indicators or regime not in self.regime_params:
            return signals
        
        indicator_signals = {}
        indicator_weights = {}
        
        for indicator in indicators:
            if indicator not in self.regime_params[regime]:
                continue
                
            params = self.regime_params[regime][indicator]['params']
            performance = self.regime_params[regime][indicator]['value']
            
            try:
                if indicator == 'Supertrend':
                    result = calculate_supertrend(data, params['st_period'], params['st_multiplier'])
                    if result is not None:
                        indicator_signals[indicator] = result['buy_signal'].astype(int) - result['sell_signal'].astype(int)
                
                elif indicator == 'MACD':
                    result = calculate_macd_custom(data, params['macd_fast'], 
                                                 params['macd_slow'], params['macd_signal'])
                    if result is not None:
                        macd_signals = pd.Series(0, index=data.index)
                        macd_signals[(result['macd'] > result['signal']) & 
                                   (result['macd'].shift(1) <= result['signal'].shift(1))] = 1
                        macd_signals[(result['macd'] < result['signal']) & 
                                   (result['macd'].shift(1) >= result['signal'].shift(1))] = -1
                        indicator_signals[indicator] = macd_signals
                
                elif indicator == 'ADX':
                    result = calculate_adx_di(data, params['adx_period'], params['adx_threshold'])
                    if result is not None:
                        adx_signals = pd.Series(0, index=data.index)
                        buy_condition = (result['di_bullish_cross'] & 
                                       (result['adx'] > params['adx_threshold']))
                        sell_condition = (result['di_bearish_cross'] | 
                                        (result['adx'] < params['adx_exit_threshold']))
                        adx_signals[buy_condition] = 1
                        adx_signals[sell_condition] = -1
                        indicator_signals[indicator] = adx_signals
                
                elif indicator == 'WaveTrend':
                    result = calculate_wavetrend(data, params['wt_n1'], params['wt_n2'])
                    if result is not None:
                        wt_signals = pd.Series(0, index=data.index)
                        buy_condition = ((result['wt1'] > result['wt2']) & 
                                       (result['wt1'].shift(1) <= result['wt2'].shift(1)) &
                                       (result['wt1'] < params['wt_oversold']))
                        sell_condition = ((result['wt1'] < result['wt2']) & 
                                        (result['wt1'].shift(1) >= result['wt2'].shift(1)) &
                                        (result['wt1'] > params['wt_overbought']))
                        wt_signals[buy_condition] = 1
                        wt_signals[sell_condition] = -1
                        indicator_signals[indicator] = wt_signals
                
                # Weight by performance (positive values only)
                indicator_weights[indicator] = max(performance, 0.1)
                
            except Exception as e:
                logger.error(f"Error calculating {indicator} for {symbol}: {e}")
        
        # Combine signals based on strategy
        if not indicator_signals:
            return signals
        
        if strategy == 'single_best':
            # Use the best performing indicator
            best_indicator = max(indicator_weights.keys(), key=lambda x: indicator_weights[x])
            signals = indicator_signals[best_indicator]
            
        elif strategy == 'dual_best':
            # Require confirmation from both indicators
            indicator_list = list(indicator_signals.keys())
            if len(indicator_list) >= 2:
                sig1 = indicator_signals[indicator_list[0]]
                sig2 = indicator_signals[indicator_list[1]]
                
                # Buy when both agree
                signals[(sig1 == 1) & (sig2 == 1)] = 1
                # Sell when either sells
                signals[(sig1 == -1) | (sig2 == -1)] = -1
            elif len(indicator_list) == 1:
                signals = indicator_signals[indicator_list[0]]
                
        elif strategy == 'weighted_ensemble':
            # Weighted voting system
            total_weight = sum(indicator_weights.values())
            
            for i in range(len(signals)):
                weighted_signal = 0
                for indicator, signal_series in indicator_signals.items():
                    if i < len(signal_series):
                        weight = indicator_weights[indicator] / total_weight
                        weighted_signal += signal_series.iloc[i] * weight
                
                # Convert weighted signal to discrete signal
                if weighted_signal > 0.5:
                    signals.iloc[i] = 1
                elif weighted_signal < -0.5:
                    signals.iloc[i] = -1
        
        return signals
    
    def calculate_portfolio_value(self, portfolio_name: str, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        portfolio = self.portfolios[portfolio_name]
        total_value = portfolio['cash']
        
        for symbol, shares in portfolio['positions'].items():
            if symbol in prices:
                total_value += shares * prices[symbol]
                
        return total_value
    
    def get_position_size(self, portfolio_name: str, price: float) -> int:
        """Calculate position size"""
        portfolio = self.portfolios[portfolio_name]
        max_positions = portfolio['max_positions']
        
        target_allocation = portfolio['cash'] / max_positions
        shares = int(target_allocation / price)
        
        return max(shares, 0)
    
    def execute_trades(self, portfolio_name: str, symbol: str, signal: int, 
                      price: float, date: datetime):
        """Execute trades based on signals"""
        portfolio = self.portfolios[portfolio_name]
        
        if signal == 1:  # Buy signal
            if symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                return
            
            current_positions = len([s for s, shares in portfolio['positions'].items() if shares > 0])
            if current_positions >= portfolio['max_positions']:
                return
            
            shares = self.get_position_size(portfolio_name, price)
            cost = shares * price
            
            if cost <= portfolio['cash'] and shares > 0:
                portfolio['cash'] -= cost
                portfolio['positions'][symbol] = shares
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'value': cost,
                    'regime': self.current_regime
                })
                
        elif signal == -1:  # Sell signal
            if symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                shares = portfolio['positions'][symbol]
                proceeds = shares * price
                
                portfolio['cash'] += proceeds
                portfolio['positions'][symbol] = 0
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'shares': shares,
                    'price': price,
                    'value': proceeds,
                    'regime': self.current_regime
                })
    
    def update_regime(self, new_regime: str, date: datetime):
        """Update current market regime"""
        if new_regime != self.current_regime:
            logger.info(f"🔄 Regime change: {self.current_regime} → {new_regime} on {date.strftime('%Y-%m-%d')}")
            
            # Record regime switch for all portfolios
            for portfolio_name in self.portfolios.keys():
                self.portfolios[portfolio_name]['regime_switches'].append({
                    'date': date,
                    'from_regime': self.current_regime,
                    'to_regime': new_regime
                })
            
            self.current_regime = new_regime
            self.regime_history.append((date, new_regime))
    
    def run_adaptive_backtest(self, start_date: str, end_date: str, 
                            regime_detection_interval: int = 10):
        """Run adaptive portfolio backtest"""
        logger.info("🚀 Starting Adaptive Portfolio Backtest")
        logger.info(f"📅 Period: {start_date} to {end_date}")
        logger.info(f"🔍 Regime detection interval: {regime_detection_interval} days")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Generate date range
        date_range = pd.date_range(start, end, freq='D')
        
        with tqdm(total=len(date_range), desc="Adaptive Backtest") as pbar:
            for i, current_date in enumerate(date_range):
                # Detect regime periodically
                if i % regime_detection_interval == 0:
                    # Use a representative symbol for regime detection
                    regime_data = self.calc.load_raw_data('ASELS', '1d')
                    if regime_data is not None:
                        # Filter to date
                        data_index = regime_data.index.tz_localize(None) if regime_data.index.tz else regime_data.index
                        mask = data_index <= pd.to_datetime(current_date).tz_localize(None)
                        historical_data = regime_data[mask]
                        
                        if len(historical_data) >= 50:
                            new_regime = self.detect_current_regime(historical_data)
                            self.update_regime(new_regime, current_date)
                
                # Get prices and signals for all symbols
                prices_dict = {}
                signals_dict = {portfolio_name: {} for portfolio_name in self.portfolios.keys()}
                
                for symbol in SACRED_SYMBOLS:
                    try:
                        data = self.calc.load_raw_data(symbol, '1d')
                        if data is None:
                            continue
                        
                        # Filter to current date
                        data_index = data.index.tz_localize(None) if data.index.tz else data.index
                        mask = data_index <= pd.to_datetime(current_date).tz_localize(None)
                        symbol_data = data[mask]
                        
                        if len(symbol_data) < 30:
                            continue
                        
                        current_price = symbol_data['close'].iloc[-1]
                        prices_dict[symbol] = current_price
                        
                        # Generate signals for each portfolio
                        for portfolio_name, portfolio in self.portfolios.items():
                            strategy = portfolio['strategy']
                            indicators = self.get_regime_indicators(self.current_regime, strategy)
                            
                            signals = self.calculate_regime_signals(
                                symbol, symbol_data, indicators, self.current_regime, strategy
                            )
                            
                            if len(signals) > 0:
                                signals_dict[portfolio_name][symbol] = signals.iloc[-1]
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                        continue
                
                # Update portfolios
                for portfolio_name in self.portfolios.keys():
                    # Execute trades
                    if portfolio_name in signals_dict:
                        for symbol, signal in signals_dict[portfolio_name].items():
                            if symbol in prices_dict:
                                self.execute_trades(portfolio_name, symbol, signal, 
                                                  prices_dict[symbol], current_date)
                    
                    # Update equity curve
                    portfolio_value = self.calculate_portfolio_value(portfolio_name, prices_dict)
                    self.portfolios[portfolio_name]['equity_curve'].append(portfolio_value)
                
                pbar.update(1)
    
    def generate_performance_report(self):
        """Generate detailed performance report"""
        logger.info("\n📊 ADAPTIVE PORTFOLIO PERFORMANCE REPORT")
        logger.info("=" * 80)
        
        results = {}
        
        for portfolio_name, portfolio in self.portfolios.items():
            equity_curve = portfolio['equity_curve']
            
            if len(equity_curve) < 2:
                continue
            
            # Calculate metrics
            returns = pd.Series(equity_curve).pct_change().dropna()
            total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
            
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
            
            peak = pd.Series(equity_curve).expanding().max()
            drawdown = (pd.Series(equity_curve) - peak) / peak
            max_drawdown = drawdown.min() * 100
            
            trades = portfolio['trades']
            buy_trades = [t for t in trades if t['action'] == 'buy']
            
            # Regime analysis
            regime_performance = {}
            for trade in trades:
                regime = trade.get('regime', 'unknown')
                if regime not in regime_performance:
                    regime_performance[regime] = []
                if trade['action'] == 'sell':
                    # Find corresponding buy
                    for buy_trade in reversed(buy_trades):
                        if (buy_trade['symbol'] == trade['symbol'] and 
                            buy_trade['date'] < trade['date']):
                            trade_return = (trade['price'] - buy_trade['price']) / buy_trade['price']
                            regime_performance[regime].append(trade_return)
                            break
            
            results[portfolio_name] = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_trades': len(buy_trades),
                'regime_switches': len(portfolio['regime_switches']),
                'regime_performance': {k: np.mean(v) * 100 if v else 0 
                                     for k, v in regime_performance.items()},
                'current_value': equity_curve[-1],
                'current_positions': len([s for s, shares in portfolio['positions'].items() if shares > 0])
            }
            
            print(f"\n🎯 {portfolio['name']}")
            print(f"   {portfolio['description']}")
            print("-" * 60)
            print(f"   💰 Total Return: {total_return:.2f}%")
            print(f"   📈 Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"   📉 Max Drawdown: {max_drawdown:.2f}%")
            print(f"   🔄 Total Trades: {len(buy_trades)}")
            print(f"   🎭 Regime Switches: {len(portfolio['regime_switches'])}")
            print(f"   🏠 Current Positions: {results[portfolio_name]['current_positions']}")
            print(f"   💵 Current Value: ${results[portfolio_name]['current_value']:,.2f}")
            
            if regime_performance:
                print(f"   📊 Regime Performance:")
                for regime, perf in results[portfolio_name]['regime_performance'].items():
                    print(f"     {regime}: {perf:.2f}%")
        
        # Best performer
        if results:
            best_portfolio = max(results.keys(), 
                               key=lambda x: results[x]['total_return'])
            
            print("\n" + "=" * 80)
            print(f"🏆 BEST PERFORMER: {self.portfolios[best_portfolio]['name']}")
            print(f"   💰 Return: {results[best_portfolio]['total_return']:.2f}%")
            print(f"   📈 Sharpe: {results[best_portfolio]['sharpe_ratio']:.2f}")
            print("=" * 80)
        
        return results
    
    def save_results(self):
        """Save adaptive portfolio results"""
        output = {
            'backtest_date': datetime.now().isoformat(),
            'system': 'adaptive_portfolio',
            'regime_history': [(date.isoformat(), regime) for date, regime in self.regime_history],
            'current_regime': self.current_regime,
            'portfolios': {}
        }
        
        for portfolio_name, portfolio in self.portfolios.items():
            output['portfolios'][portfolio_name] = {
                'name': portfolio['name'],
                'description': portfolio['description'],
                'strategy': portfolio['strategy'],
                'equity_curve': portfolio['equity_curve'][-100:],  # Last 100 points
                'regime_switches': portfolio['regime_switches'],
                'final_positions': {k: v for k, v in portfolio['positions'].items() if v > 0},
                'recent_trades': portfolio['trades'][-20:] if portfolio['trades'] else []
            }
        
        with open('adaptive_portfolio_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info("💾 Results saved to: adaptive_portfolio_results.json")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Portfolio System')
    parser.add_argument('--start', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--regime-interval', type=int, default=7,
                       help='Regime detection interval in days')
    
    args = parser.parse_args()
    
    # Create adaptive portfolio system
    adaptive_system = AdaptivePortfolioSystem()
    
    # Run backtest
    adaptive_system.run_adaptive_backtest(
        args.start, args.end, args.regime_interval
    )
    
    # Generate report
    adaptive_system.generate_performance_report()
    
    # Save results
    adaptive_system.save_results()


if __name__ == "__main__":
    main()