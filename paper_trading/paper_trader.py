#!/usr/bin/env python3
"""
Paper Trader - Main Trading Engine
Coordinates portfolio management and signal generation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import schedule
import sys
sys.path.append(str(Path(__file__).parent.parent))

from portfolio_manager import PortfolioManager
from signal_generator import SignalGenerator
from performance_tracker import PerformanceTracker
from data_fetcher import DataFetcher
import logging

# Setup logging
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTrader:
    """Main paper trading engine"""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
        self.signal_generator = SignalGenerator()
        self.performance_tracker = PerformanceTracker()
        self.data_fetcher = DataFetcher()
        
        # Trading strategies
        self.strategies = {
            'aggressive': {
                'portfolio_name': 'aggressive',
                'description': 'VixFix Enhanced Strategy',
                'initial_capital': 50000
            },
            'balanced': {
                'portfolio_name': 'balanced',
                'description': 'Balanced Top 3 Strategy',
                'initial_capital': 50000
            },
            'conservative': {
                'portfolio_name': 'conservative',
                'description': 'Conservative 5 Indicator Strategy',
                'initial_capital': 50000
            }
        }
        
        # Initialize portfolios
        self.initialize_portfolios()
        
        # Trading state
        self.is_trading_hours = False
        self.last_update_time = None
        
    def initialize_portfolios(self):
        """Initialize portfolios for each strategy"""
        for strategy_name, config in self.strategies.items():
            portfolio = self.portfolio_manager.get_portfolio(config['portfolio_name'])
            if not portfolio:
                portfolio = self.portfolio_manager.create_portfolio(
                    config['portfolio_name'],
                    config['initial_capital']
                )
                logger.info(f"Created portfolio for {strategy_name} strategy")
    
    def check_trading_hours(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        
        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        
        # BIST trading hours: 10:00 - 18:00 Turkey time
        market_open = now.replace(hour=10, minute=0, second=0)
        market_close = now.replace(hour=18, minute=0, second=0)
        
        return market_open <= now <= market_close
    
    def update_market_data(self):
        """Update market data for all symbols"""
        logger.info("Updating market data...")
        
        try:
            # Fetch latest prices
            current_prices = self.data_fetcher.get_current_prices()
            
            if not current_prices:
                logger.warning("No price data available")
                return False
            
            # Update all portfolios with new prices
            self.portfolio_manager.update_all_portfolios(current_prices)
            
            logger.info(f"Updated prices for {len(current_prices)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
            return False
    
    def execute_strategy(self, strategy_name: str):
        """Execute trading strategy"""
        logger.info(f"Executing {strategy_name} strategy...")
        
        try:
            # Get portfolio
            config = self.strategies[strategy_name]
            portfolio = self.portfolio_manager.get_portfolio(config['portfolio_name'])
            
            if not portfolio:
                logger.error(f"Portfolio not found for {strategy_name}")
                return
            
            # Generate signals
            if strategy_name == 'aggressive':
                signals = self.signal_generator.scan_all_symbols('vixfix_enhanced')
            else:
                signals = self.signal_generator.scan_all_symbols(strategy_name)
            
            # Get current prices
            current_prices = self.data_fetcher.get_current_prices()
            
            # Check exit signals for existing positions
            exit_signals = self.signal_generator.get_exit_signals(
                portfolio.positions,
                'vixfix_enhanced' if strategy_name == 'aggressive' else strategy_name
            )
            
            # Execute exits first
            for symbol, exit_data in exit_signals.items():
                if symbol in portfolio.positions:
                    portfolio.close_position(symbol, exit_data['price'], exit_data['reason'])
                    logger.info(f"{strategy_name}: Closed {symbol} - Reason: {exit_data['reason']}")
            
            # Filter buy signals
            buy_signals = {s: d for s, d in signals.items() if d['signal'] == 1}
            
            # Rebalance portfolio with new signals
            if buy_signals:
                # Create price dict for new signals
                signal_prices = {s: d['price'] for s, d in buy_signals.items()}
                portfolio.rebalance_portfolio(signal_prices, current_prices)
            
            # Save portfolio state
            portfolio.save_state(
                self.portfolio_manager.data_dir / f"{config['portfolio_name']}_portfolio.json"
            )
            
            # Log performance
            metrics = portfolio.get_performance_metrics()
            logger.info(f"{strategy_name} Performance - Value: {metrics['portfolio_value']:.2f} TL, "
                       f"Return: {metrics['total_return']:.2f}%, "
                       f"Positions: {metrics['num_positions']}")
            
        except Exception as e:
            logger.error(f"Error executing {strategy_name} strategy: {e}")
    
    def run_trading_cycle(self):
        """Run a complete trading cycle"""
        logger.info("="*60)
        logger.info("Starting trading cycle...")
        
        # Check if market is open
        if not self.check_trading_hours():
            logger.info("Market is closed")
            return
        
        # Update market data
        if not self.update_market_data():
            logger.error("Failed to update market data")
            return
        
        # Execute each strategy
        for strategy_name in self.strategies.keys():
            self.execute_strategy(strategy_name)
        
        # Update performance tracking
        self.performance_tracker.update_daily_performance(self.portfolio_manager)
        
        # Generate daily report
        self.generate_daily_report()
        
        self.last_update_time = datetime.now()
        logger.info("Trading cycle completed")
    
    def generate_daily_report(self):
        """Generate daily performance report"""
        try:
            # Get portfolio summaries
            summary_df = self.portfolio_manager.get_summary()
            
            # Create report
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'portfolios': []
            }
            
            print("\n" + "="*80)
            print(f"📊 DAILY REPORT - {report['date']} {report['time']}")
            print("="*80)
            
            for _, row in summary_df.iterrows():
                portfolio_data = {
                    'name': row['name'],
                    'value': row['portfolio_value'],
                    'return': row['total_return'],
                    'pnl': row['total_pnl'],
                    'positions': row['num_positions'],
                    'win_rate': row['win_rate'],
                    'sharpe': row['sharpe_ratio'],
                    'max_dd': row['max_drawdown']
                }
                report['portfolios'].append(portfolio_data)
                
                # Print summary
                print(f"\n🎯 {row['name'].upper()} Strategy:")
                print(f"   Portfolio Value: {row['portfolio_value']:,.2f} TL")
                print(f"   Total Return: {row['total_return']:.2f}%")
                print(f"   Today's P&L: {row['total_pnl']:.2f} TL")
                print(f"   Open Positions: {row['num_positions']}")
                print(f"   Win Rate: {row['win_rate']:.1f}%")
                print(f"   Sharpe Ratio: {row['sharpe_ratio']:.2f}")
                print(f"   Max Drawdown: {row['max_drawdown']:.1f}%")
                
                # Show positions
                portfolio = self.portfolio_manager.get_portfolio(row['name'])
                if portfolio and portfolio.positions:
                    print(f"\n   Current Positions:")
                    for symbol, pos in portfolio.positions.items():
                        print(f"     {symbol}: {pos.shares} @ {pos.entry_price:.2f} "
                              f"(P&L: {pos.get_pnl_percentage():.1f}%)")
            
            # Best performer
            best_idx = summary_df['total_return'].idxmax()
            best_portfolio = summary_df.loc[best_idx]
            print(f"\n🏆 Best Performer: {best_portfolio['name']} ({best_portfolio['total_return']:.2f}%)")
            
            # Save report
            report_path = Path('paper_trading/reports') / f"daily_report_{report['date']}.json"
            report_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📄 Report saved to: {report_path}")
            print("="*80)
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def run_continuous(self):
        """Run paper trader continuously"""
        logger.info("Starting paper trader in continuous mode...")
        
        # Schedule trading cycles
        schedule.every(5).minutes.do(self.run_trading_cycle)
        
        # Schedule daily report at market close
        schedule.every().day.at("18:15").do(self.generate_daily_report)
        
        print("\n🚀 Paper Trader Started!")
        print("="*60)
        print("Strategies:")
        for name, config in self.strategies.items():
            print(f"  - {name}: {config['description']}")
        print("\nTrading will run every 5 minutes during market hours")
        print("Daily reports will be generated at 18:15")
        print("\nPress Ctrl+C to stop")
        print("="*60)
        
        # Run initial cycle
        self.run_trading_cycle()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Paper trader stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in continuous run: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def backtest_mode(self, start_date: str, end_date: str):
        """Run paper trader in backtest mode"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # This would replay historical data
        # Implementation depends on data availability
        pass


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading System')
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live',
                       help='Trading mode')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit')
    parser.add_argument('--report', action='store_true',
                       help='Generate report only')
    
    args = parser.parse_args()
    
    # Create paper trader
    trader = PaperTrader()
    
    if args.report:
        trader.generate_daily_report()
    elif args.once:
        trader.run_trading_cycle()
    else:
        trader.run_continuous()


if __name__ == "__main__":
    main()