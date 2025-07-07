#!/usr/bin/env python3
"""
Portfolio Backtest Runner
Run portfolio strategy backtests for different time periods
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS
from trading.portfolio_strategy import PortfolioStrategy
from backtest.realistic_backtest import IndicatorBacktest
from backtest.backtest_sirali import HierarchicalBacktest


class PortfolioBacktester:
    """Run portfolio strategy backtests"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.portfolio = PortfolioStrategy(initial_capital)
        
    async def backtest_period(self, days: int) -> Dict:
        """Run backtest for specific period"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {days}-day portfolio backtest")
        logger.info(f"{'='*60}")
        
        # Initialize results
        results = {
            'period_days': days,
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return': 0,
            'trades': [],
            'strategy_performance': {},
            'daily_values': []
        }
        
        # Simulate daily trading
        current_capital = self.initial_capital
        positions = {}
        
        # Get historical data for the period
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Track daily portfolio value
        daily_values = []
        
        # Simulate each day
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Generate signals
            signals_dict = self.portfolio.generate_portfolio_signals(SACRED_SYMBOLS[:20])
            
            if not signals_dict:
                continue
                
            # Execute trades
            trades = self.portfolio.execute_portfolio_signals(signals_dict, current_date)
            
            # Process trades
            for trade in trades:
                if trade['action'] == 'BUY':
                    # Open position
                    symbol = trade['symbol']
                    if symbol not in positions:
                        positions[symbol] = {
                            'entry_price': trade['price'],
                            'entry_date': trade['date'],
                            'size': trade['size'],
                            'strategy': trade['strategy']
                        }
                        results['trades'].append(trade)
                        current_capital -= trade['size']
                        
                elif trade['action'] == 'SELL':
                    # Close position
                    symbol = trade['symbol']
                    if symbol in positions:
                        position = positions[symbol]
                        return_pct = ((trade['price'] - position['entry_price']) / position['entry_price']) * 100
                        profit = position['size'] * (return_pct / 100)
                        
                        trade['return_pct'] = return_pct
                        trade['profit'] = profit
                        results['trades'].append(trade)
                        
                        current_capital += position['size'] + profit
                        del positions[symbol]
            
            # Calculate daily portfolio value
            position_value = sum(pos['size'] for pos in positions.values())
            total_value = current_capital + position_value
            daily_values.append({
                'date': current_date,
                'value': total_value,
                'return_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100
            })
        
        # Calculate final metrics
        final_position_value = sum(pos['size'] for pos in positions.values())
        results['final_value'] = current_capital + final_position_value
        results['total_return'] = ((results['final_value'] - self.initial_capital) / self.initial_capital) * 100
        results['daily_values'] = daily_values
        
        # Calculate strategy-specific performance
        for strategy in ['realistic', 'hier_4h_sl_trailing']:
            strategy_trades = [t for t in results['trades'] if t.get('strategy') == strategy]
            sells = [t for t in strategy_trades if t['action'] == 'SELL']
            
            if sells:
                wins = [t for t in sells if t.get('return_pct', 0) > 0]
                total_return = sum(t.get('return_pct', 0) for t in sells)
                
                results['strategy_performance'][strategy] = {
                    'trades': len(sells),
                    'wins': len(wins),
                    'win_rate': (len(wins) / len(sells)) * 100 if sells else 0,
                    'total_return': total_return,
                    'avg_return': total_return / len(sells) if sells else 0
                }
            else:
                results['strategy_performance'][strategy] = {
                    'trades': 0,
                    'wins': 0,
                    'win_rate': 0,
                    'total_return': 0,
                    'avg_return': 0
                }
        
        # Calculate risk metrics
        if daily_values:
            returns = pd.Series([d['return_pct'] for d in daily_values])
            
            # Max drawdown
            cumulative = (1 + returns/100).cumprod()
            running_max = cumulative.expanding().max()
            drawdowns = (cumulative - running_max) / running_max
            results['max_drawdown'] = abs(drawdowns.min()) * 100
            
            # Sharpe ratio
            if len(returns) > 1 and returns.std() > 0:
                results['sharpe_ratio'] = (returns.mean() / returns.std()) * (252 ** 0.5)
            else:
                results['sharpe_ratio'] = 0
        else:
            results['max_drawdown'] = 0
            results['sharpe_ratio'] = 0
        
        # Overall win rate
        all_sells = [t for t in results['trades'] if t['action'] == 'SELL']
        if all_sells:
            wins = [t for t in all_sells if t.get('return_pct', 0) > 0]
            results['win_rate'] = (len(wins) / len(all_sells)) * 100
        else:
            results['win_rate'] = 0
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print(f"\n{'='*80}")
        print(f"PORTFOLIO BACKTEST RESULTS - {results['period_days']} DAYS")
        print(f"{'='*80}")
        
        print(f"\n💰 Capital: ${results['initial_capital']:,.2f} → ${results['final_value']:,.2f}")
        print(f"📈 Total Return: {results['total_return']:.2f}%")
        print(f"📊 Total Trades: {len([t for t in results['trades'] if t['action'] == 'SELL'])}")
        print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
        print(f"📉 Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        
        print(f"\n📊 Strategy Performance:")
        for strategy, perf in results['strategy_performance'].items():
            if perf['trades'] > 0:
                print(f"\n  {strategy}:")
                print(f"    • Trades: {perf['trades']}")
                print(f"    • Win Rate: {perf['win_rate']:.1f}%")
                print(f"    • Total Return: {perf['total_return']:.2f}%")
                print(f"    • Avg Return/Trade: {perf['avg_return']:.2f}%")
        
        # Show best and worst trades
        sells = [t for t in results['trades'] if t['action'] == 'SELL' and 'return_pct' in t]
        if sells:
            best_trade = max(sells, key=lambda x: x.get('return_pct', 0))
            worst_trade = min(sells, key=lambda x: x.get('return_pct', 0))
            
            print(f"\n📊 Best Trade: {best_trade['symbol']} +{best_trade['return_pct']:.2f}%")
            print(f"📊 Worst Trade: {worst_trade['symbol']} {worst_trade['return_pct']:.2f}%")
    
    async def run_multiple_periods(self, periods: List[int]):
        """Run backtests for multiple periods"""
        all_results = {}
        
        for days in periods:
            results = await self.backtest_period(days)
            all_results[days] = results
            self.print_results(results)
        
        # Summary comparison
        print(f"\n{'='*80}")
        print("PORTFOLIO BACKTEST SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Period':<10} {'Return':<10} {'Sharpe':<10} {'Win Rate':<10} {'Max DD':<10}")
        print("-" * 50)
        
        for days, results in sorted(all_results.items()):
            print(f"{days}d{' ':<6} "
                  f"{results['total_return']:>8.2f}% "
                  f"{results['sharpe_ratio']:>9.2f} "
                  f"{results['win_rate']:>8.1f}% "
                  f"{results['max_drawdown']:>8.2f}%")
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"portfolio_backtest_results_{timestamp}.csv"
        
        # Create DataFrame for CSV
        rows = []
        for days, results in all_results.items():
            row = {
                'period_days': days,
                'total_return': results['total_return'],
                'sharpe_ratio': results['sharpe_ratio'],
                'win_rate': results['win_rate'],
                'max_drawdown': results['max_drawdown'],
                'total_trades': len([t for t in results['trades'] if t['action'] == 'SELL']),
                'final_value': results['final_value']
            }
            
            # Add strategy performance
            for strategy in ['realistic', 'hier_4h_sl_trailing']:
                if strategy in results['strategy_performance']:
                    perf = results['strategy_performance'][strategy]
                    row[f'{strategy}_trades'] = perf['trades']
                    row[f'{strategy}_win_rate'] = perf['win_rate']
                    row[f'{strategy}_avg_return'] = perf['avg_return']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(results_file, index=False)
        print(f"\n📁 Detailed results saved to: {results_file}")


async def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚀 PORTFOLIO STRATEGY BACKTESTER")
    print("="*80)
    print("\nThis will run portfolio backtests for multiple time periods")
    print("Using 60% Realistic + 30% Hierarchical 4H + 10% Cash")
    
    # Get periods to test
    print("\nSelect periods to test:")
    print("1. Quick test: 30, 60 days")
    print("2. Standard: 30, 60, 90, 120 days")
    print("3. Extended: 30, 60, 90, 120, 180, 240 days")
    print("4. Custom (enter your own)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        periods = [30, 60]
    elif choice == '2':
        periods = [30, 60, 90, 120]
    elif choice == '3':
        periods = [30, 60, 90, 120, 180, 240]
    elif choice == '4':
        custom = input("Enter periods separated by comma (e.g., 30,60,90): ")
        periods = [int(p.strip()) for p in custom.split(',')]
    else:
        print("Invalid choice, using standard periods")
        periods = [30, 60, 90, 120]
    
    # Get initial capital
    capital_str = input("\nInitial capital (default 100000): ").strip()
    initial_capital = float(capital_str) if capital_str else 100000
    
    print(f"\n✅ Running backtests for periods: {periods}")
    print(f"💰 Initial capital: ${initial_capital:,.2f}")
    
    # Create backtester
    backtester = PortfolioBacktester(initial_capital)
    
    # Run backtests
    await backtester.run_multiple_periods(periods)
    
    print("\n✅ Backtest complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBacktest cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()