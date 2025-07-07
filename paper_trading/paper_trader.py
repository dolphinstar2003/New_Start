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
import asyncio
import threading
sys.path.append(str(Path(__file__).parent.parent))

from portfolio_manager import PortfolioManager
from signal_generator import SignalGenerator
from performance_tracker import PerformanceTracker
from data_fetcher import DataFetcher
from telegram_bot import PaperTradingBot
from target_manager import TargetManager
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
    
    def __init__(self, enable_telegram=False):
        self.portfolio_manager = PortfolioManager()
        self.signal_generator = SignalGenerator()
        self.performance_tracker = PerformanceTracker()
        self.data_fetcher = DataFetcher()
        self.target_manager = TargetManager()
        
        # Telegram bot
        self.telegram_bot = None
        self.enable_telegram = enable_telegram
        if enable_telegram:
            self._init_telegram_bot()
        
        # Trading strategies based on optimization results
        self.strategies = {
            'aggressive': {
                'portfolio_name': 'aggressive',
                'description': 'Supertrend Only (41,347% avg return)',
                'signal_method': 'aggressive',
                'initial_capital': 50000
            },
            'balanced': {
                'portfolio_name': 'balanced', 
                'description': 'VixFix Enhanced Supertrend (56,908% backtest)',
                'signal_method': 'vixfix_enhanced',
                'initial_capital': 50000
            },
            'conservative': {
                'portfolio_name': 'conservative',
                'description': 'MACD + ADX Balanced (3,585% + 3,603% avg)',
                'signal_method': 'conservative',
                'initial_capital': 50000
            }
        }
        
        # Initialize portfolios
        self.initialize_portfolios()
        
        # Trading state
        self.is_trading_hours = False
        self.last_update_time = None
    
    def _init_telegram_bot(self):
        """Initialize Telegram bot"""
        try:
            # Load telegram config
            config_file = Path(__file__).parent.parent / 'telegram_config.json'
            if config_file.exists():
                self.telegram_bot = PaperTradingBot(self)
                logger.info("Telegram bot initialized")
            else:
                logger.warning("Telegram config not found. Run telegram_bot.py to configure.")
                self.enable_telegram = False
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            self.enable_telegram = False
    
    async def _send_telegram_notification(self, message: str, notification_type: str = "info"):
        """Send notification via Telegram"""
        if self.telegram_bot:
            try:
                await self.telegram_bot.send_notification(message, notification_type)
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")
        
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
    
    def display_market_status(self, prices: dict):
        """Display market status with fixed daily targets for all strategies"""
        # Clear screen for cleaner display
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        
        print("╔" + "═"*98 + "╗")
        print(f"║{' '*40}📊 MARKET STATUS{' '*41}║")
        print(f"║{' '*36}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{' '*39}║")
        print("╠" + "═"*98 + "╣")
        
        # Get targets for each strategy
        strategies_targets = {}
        for strategy in ['aggressive', 'balanced', 'conservative']:
            strategies_targets[strategy] = self.target_manager.get_all_targets(strategy)
        
        # Display header
        print(f"║ {'Symbol':<8} {'Current':<10} {'Aggressive':<12} {'Balanced':<12} {'Conservative':<12} {'Status':<9} ║")
        print("╠" + "─"*98 + "╣")
        
        # Sort symbols by nearest to target
        symbol_distances = []
        for symbol in sorted(prices.keys()):
            if symbol == 'XU100':  # Skip index
                continue
                
            current_price = prices[symbol]
            
            # Get target for aggressive strategy (closest to market)
            agg_target = strategies_targets['aggressive'].get(symbol, {})
            if agg_target:
                distance = agg_target.get('distance', 100)
                symbol_distances.append((symbol, distance, current_price))
        
        # Sort by distance (closest first)
        symbol_distances.sort(key=lambda x: abs(x[1]))
        
        # Display each symbol
        for symbol, _, current_price in symbol_distances[:10]:  # Show top 10 nearest
            # Get targets for each strategy
            agg_data = strategies_targets['aggressive'].get(symbol, {})
            bal_data = strategies_targets['balanced'].get(symbol, {})
            con_data = strategies_targets['conservative'].get(symbol, {})
            
            agg_target = agg_data.get('target_price', 0)
            bal_target = bal_data.get('target_price', 0)
            con_target = con_data.get('target_price', 0)
            
            # Determine status (check if any strategy is ready)
            status = "WAIT"
            if (agg_target > 0 and current_price <= agg_target) or \
               (bal_target > 0 and current_price <= bal_target) or \
               (con_target > 0 and current_price <= con_target):
                status = "READY"
            
            # Format targets
            agg_str = f"₺{agg_target:.2f}" if agg_target > 0 else "-"
            bal_str = f"₺{bal_target:.2f}" if bal_target > 0 else "-"
            con_str = f"₺{con_target:.2f}" if con_target > 0 else "-"
            
            # Color status
            status_symbol = "🟢" if status == "READY" else "🟡"
            
            print(f"║ {symbol:<8} ₺{current_price:<9.2f} {agg_str:<12} {bal_str:<12} {con_str:<12} {status_symbol} {status:<8} ║")
        
        # Summary
        print("╠" + "═"*98 + "╣")
        
        summary = self.target_manager.get_status_summary(prices)
        
        # Show summary in one line per strategy
        for strategy, data in summary.items():
            ready_pct = (data['ready_count'] / data['total_symbols'] * 100) if data['total_symbols'] > 0 else 0
            print(f"║ {strategy.upper():<12}: Ready {data['ready_count']:>2}/{data['total_symbols']:<2} ({ready_pct:>4.1f}%) | Avg Distance: {data['avg_distance']:>5.2f}%{' '*38} ║")
        
        print("╠" + "═"*98 + "╣")
        print(f"║ 🔄 Next update in: {300 - (int(time.time()) % 300)} seconds{' '*70} ║")
        print("╚" + "═"*98 + "╝")
    
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
            
            # Generate signals using the strategy's signal method
            signal_method = config.get('signal_method', strategy_name)
            signals = self.signal_generator.scan_all_symbols(signal_method)
            
            # Get current prices
            current_prices = self.data_fetcher.get_current_prices()
            
            # Check exit signals for existing positions
            exit_signals = self.signal_generator.get_exit_signals(
                portfolio.positions,
                signal_method
            )
            
            # Execute exits first
            for symbol, exit_data in exit_signals.items():
                if symbol in portfolio.positions:
                    pos = portfolio.positions[symbol]
                    profit = portfolio.close_position(symbol, exit_data['price'], exit_data['reason'])
                    logger.info(f"{strategy_name}: Closed {symbol} - Reason: {exit_data['reason']}")
                    
                    # Send Telegram notification
                    if self.enable_telegram:
                        asyncio.create_task(
                            self.telegram_bot.send_position_closed_notification(
                                strategy_name, symbol, pos.entry_price, 
                                exit_data['price'], pos.shares, profit, exit_data['reason']
                            )
                        )
            
            # Filter buy signals
            buy_signals = {s: d for s, d in signals.items() if d['signal'] == 1}
            
            # Rebalance portfolio with new signals
            if buy_signals:
                # Create price dict for new signals
                signal_prices = {s: d['price'] for s, d in buy_signals.items()}
                
                # Store positions before rebalance
                positions_before = set(portfolio.positions.keys())
                
                # Rebalance
                portfolio.rebalance_portfolio(signal_prices, current_prices)
                
                # Find new positions
                positions_after = set(portfolio.positions.keys())
                new_positions = positions_after - positions_before
                
                # Send notifications for new positions
                if self.enable_telegram:
                    for symbol in new_positions:
                        if symbol in portfolio.positions:
                            pos = portfolio.positions[symbol]
                            asyncio.create_task(
                                self.telegram_bot.send_trade_notification(
                                    strategy_name, "BUY", symbol,
                                    pos.shares, pos.entry_price, "Signal"
                                )
                            )
            
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
        logger.info("Updating market data...")
        prices = self.data_fetcher.get_current_prices()
        
        if not prices:
            logger.error("Failed to get price data")
            # For demonstration, load from cache file
            cache_file = Path('data/cache/latest_prices_algolab.json')
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        prices = data.get('prices', {})
                        logger.info(f"Loaded {len(prices)} prices from cache for demonstration")
                except:
                    return
            else:
                return
        
        logger.info(f"Updated prices for {len(prices)} symbols")
        
        # Update daily targets (only once per day)
        self.target_manager.update_targets(prices)
        logger.info(f"Targets updated. Current targets count: {len(self.target_manager.targets.get('strategies', {}).get('aggressive', {}))}")
        
        # Display current market status with targets
        self.display_market_status(prices)
        
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
        
        # Schedule daily Telegram summary if enabled
        if self.enable_telegram:
            schedule.every().day.at("18:30").do(lambda: asyncio.create_task(self.telegram_bot.send_daily_summary()))
        
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
    parser.add_argument('--telegram', action='store_true',
                       help='Enable Telegram notifications')
    
    args = parser.parse_args()
    
    # Create paper trader
    trader = PaperTrader(enable_telegram=args.telegram)
    
    if args.report:
        trader.generate_daily_report()
    elif args.once:
        trader.run_trading_cycle()
    else:
        trader.run_continuous()


if __name__ == "__main__":
    main()