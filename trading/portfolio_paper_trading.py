"""
Portfolio-based Paper Trading
Implements the 60/30/10 portfolio strategy in paper trading
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS
from trading.portfolio_strategy import PortfolioStrategy
from core.data_fetcher import DataFetcher
from paper_trading.simple_paper_trader import PaperTrader


class PortfolioPaperTrader:
    """Paper trading with portfolio strategy"""
    
    def __init__(self, initial_capital: float = 100000):
        self.portfolio = PortfolioStrategy(initial_capital)
        self.paper_trader = PaperTrader(initial_capital)
        self.data_fetcher = DataFetcher()
        
        # Trading state
        self.is_running = False
        self.last_check_time = None
        self.check_interval = 60  # 1 minute for testing
        
        # Performance tracking
        self.daily_returns = []
        self.trades_log = []
        
        logger.info("Portfolio Paper Trader initialized")
    
    async def check_signals(self):
        """Check signals from portfolio strategies"""
        try:
            # Generate signals for all symbols
            symbols = SACRED_SYMBOLS[:20]  # Top 20 symbols
            signals_dict = self.portfolio.generate_portfolio_signals(symbols)
            
            if not signals_dict:
                logger.info("No signals generated")
                return
            
            # Get current date/time
            current_date = datetime.now()
            
            # Execute portfolio signals
            trades = self.portfolio.execute_portfolio_signals(signals_dict, current_date)
            
            if trades:
                logger.info(f"Generated {len(trades)} trade signals")
            else:
                logger.debug("No actionable trades from signals")
            
            # Process trades through paper trader
            for trade in trades:
                if trade['action'] == 'BUY':
                    # Check if we have a position
                    existing_position = self.paper_trader.get_position(trade['symbol'])
                    if not existing_position:
                        # Open new position
                        success = self.paper_trader.open_position(
                            symbol=trade['symbol'],
                            quantity=int(trade['size'] / trade['price']),
                            price=trade['price'],
                            position_type='LONG',
                            stop_loss=trade['price'] * 0.97,  # 3% stop loss
                            take_profit=trade['price'] * 1.08  # 8% take profit
                        )
                        
                        if success:
                            logger.info(f"✅ Opened {trade['strategy']} position: {trade['symbol']} @ ${trade['price']:.2f}")
                            self.trades_log.append({
                                'timestamp': current_date,
                                'strategy': trade['strategy'],
                                'symbol': trade['symbol'],
                                'action': 'BUY',
                                'price': trade['price'],
                                'size': trade['size']
                            })
                
                elif trade['action'] == 'SELL':
                    # Close position
                    position = self.paper_trader.get_position(trade['symbol'])
                    if position:
                        success = self.paper_trader.close_position(
                            trade['symbol'],
                            trade['price']
                        )
                        
                        if success:
                            logger.info(f"✅ Closed {trade['strategy']} position: {trade['symbol']} @ ${trade['price']:.2f} ({trade.get('return_pct', 0):.2f}%)")
                            self.trades_log.append({
                                'timestamp': current_date,
                                'strategy': trade['strategy'],
                                'symbol': trade['symbol'],
                                'action': 'SELL',
                                'price': trade['price'],
                                'return_pct': trade.get('return_pct', 0),
                                'size': trade['size']
                            })
            
            # Update portfolio performance
            self._update_performance()
            
        except Exception as e:
            logger.error(f"Error checking signals: {e}")
    
    def _update_performance(self):
        """Update portfolio performance metrics"""
        # Get paper trader stats
        stats = self.paper_trader.get_account_stats()
        
        # Update portfolio performance
        self.portfolio.performance['portfolio']['total_return'] = (
            (stats['total_equity'] - self.portfolio.initial_capital) / self.portfolio.initial_capital * 100
        )
        
        # Calculate strategy-specific performance
        for strategy in ['realistic', 'hier_4h_sl_trailing']:
            strategy_trades = [t for t in self.trades_log if t['strategy'] == strategy and t['action'] == 'SELL']
            if strategy_trades:
                returns = [t['return_pct'] for t in strategy_trades]
                wins = [r for r in returns if r > 0]
                
                self.portfolio.performance[strategy]['trades'] = len(strategy_trades)
                self.portfolio.performance[strategy]['profit'] = sum(returns)
                self.portfolio.performance[strategy]['win_rate'] = len(wins) / len(returns) * 100 if returns else 0
    
    def get_portfolio_report(self) -> Dict:
        """Generate comprehensive portfolio report"""
        # Get current status
        status = self.portfolio.get_portfolio_status()
        paper_stats = self.paper_trader.get_account_stats()
        
        # Calculate metrics
        total_return = ((paper_stats['total_equity'] - self.portfolio.initial_capital) / 
                       self.portfolio.initial_capital * 100)
        
        # Strategy breakdown
        strategy_performance = {}
        for strategy in ['realistic', 'hier_4h_sl_trailing']:
            strategy_trades = [t for t in self.trades_log if t['strategy'] == strategy]
            sells = [t for t in strategy_trades if t['action'] == 'SELL']
            
            if sells:
                avg_return = sum(t['return_pct'] for t in sells) / len(sells)
                total_profit = sum(t['return_pct'] * t['size'] / 100 for t in sells)
            else:
                avg_return = 0
                total_profit = 0
            
            strategy_performance[strategy] = {
                'trades': len(sells),
                'avg_return': avg_return,
                'total_profit': total_profit,
                'allocation': status['current_allocations'].get(strategy, 0) * 100
            }
        
        report = {
            'portfolio_summary': {
                'total_value': paper_stats['total_equity'],
                'total_return': total_return,
                'cash_available': paper_stats['available_balance'],
                'positions_count': paper_stats['positions_count'],
                'unrealized_pnl': paper_stats['unrealized_pnl']
            },
            'allocations': {
                'current': {k: v*100 for k, v in status['current_allocations'].items()},
                'target': {k: v*100 for k, v in status['target_allocations'].items()},
                'deviation': {
                    k: (status['current_allocations'][k] - status['target_allocations'][k]) * 100
                    for k in status['target_allocations']
                }
            },
            'strategy_performance': strategy_performance,
            'recent_trades': self.trades_log[-10:] if self.trades_log else [],
            'risk_metrics': {
                'max_drawdown': abs(min(self.daily_returns, default=0)) if self.daily_returns else 0,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'win_rate': paper_stats.get('win_rate', 0)
            }
        }
        
        return report
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from daily returns"""
        if len(self.daily_returns) < 2:
            return 0
        
        returns = pd.Series(self.daily_returns)
        return (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    def print_portfolio_status(self):
        """Print formatted portfolio status"""
        report = self.get_portfolio_report()
        
        print("\n" + "="*80)
        print("PORTFOLIO PAPER TRADING STATUS")
        print("="*80)
        
        # Portfolio Summary
        summary = report['portfolio_summary']
        print(f"\n💰 Portfolio Value: ${summary['total_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2f}%")
        print(f"💵 Cash Available: ${summary['cash_available']:,.2f}")
        print(f"📊 Open Positions: {summary['positions_count']}")
        print(f"💹 Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
        
        # Allocations
        print("\n📊 Portfolio Allocations:")
        alloc = report['allocations']
        for strategy in alloc['current']:
            current = alloc['current'][strategy]
            target = alloc['target'][strategy]
            deviation = alloc['deviation'][strategy]
            
            status_emoji = "✅" if abs(deviation) < 5 else "⚠️"
            print(f"  {status_emoji} {strategy}: {current:.1f}% (target: {target:.0f}%, dev: {deviation:+.1f}%)")
        
        # Strategy Performance
        print("\n📈 Strategy Performance:")
        for strategy, perf in report['strategy_performance'].items():
            if perf['trades'] > 0:
                print(f"\n  {strategy}:")
                print(f"    Trades: {perf['trades']}")
                print(f"    Avg Return: {perf['avg_return']:.2f}%")
                print(f"    Total Profit: ${perf['total_profit']:,.2f}")
                print(f"    Allocation: {perf['allocation']:.1f}%")
        
        # Risk Metrics
        risk = report['risk_metrics']
        print("\n⚠️ Risk Metrics:")
        print(f"  Max Drawdown: {risk['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {risk['win_rate']:.1f}%")
        
        # Recent Trades
        if report['recent_trades']:
            print("\n📝 Recent Trades:")
            for trade in report['recent_trades'][-5:]:
                emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                print(f"  {emoji} {trade['timestamp'].strftime('%m/%d %H:%M')} - "
                      f"{trade['strategy']}: {trade['action']} {trade['symbol']} @ ${trade['price']:.2f}")
                if trade['action'] == 'SELL' and 'return_pct' in trade:
                    print(f"      Return: {trade['return_pct']:.2f}%")
        
        print("\n" + "="*80)
    
    async def start_trading(self):
        """Start portfolio paper trading"""
        self.is_running = True
        logger.info("Starting portfolio paper trading...")
        
        while self.is_running:
            try:
                # Check signals
                await self.check_signals()
                
                # Print status
                self.print_portfolio_status()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Trading interrupted by user")
                break
            except Exception as e:
                logger.error(f"Trading error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def stop_trading(self):
        """Stop trading"""
        self.is_running = False
        logger.info("Stopping portfolio paper trading...")


async def main():
    """Run portfolio paper trading"""
    trader = PortfolioPaperTrader(initial_capital=100000)
    
    try:
        await trader.start_trading()
    except KeyboardInterrupt:
        trader.stop_trading()
        
    # Final report
    trader.print_portfolio_status()


if __name__ == "__main__":
    import numpy as np  # Add import
    asyncio.run(main())