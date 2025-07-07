"""
Portfolio Rotation Paper Trader
Implements dynamic top 10 rotation strategy
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
from trading.dynamic_portfolio_rotation import DynamicRotationStrategy
from paper_trading.simple_paper_trader import PaperTrader
from core.data_fetcher import DataFetcher


class PortfolioRotationTrader:
    """Paper trading with dynamic rotation strategy"""
    
    def __init__(self, initial_capital: float = 100000):
        self.rotation_strategy = DynamicRotationStrategy(initial_capital, max_positions=10)
        self.paper_trader = PaperTrader(initial_capital)
        self.data_fetcher = DataFetcher()
        
        # Trading state
        self.is_running = False
        self.check_interval = 300  # 5 minutes
        self.last_rotation = None
        self.min_rotation_interval = 3600  # 1 hour minimum between rotations
        
        # Performance tracking
        self.rotation_history = []
        self.daily_snapshots = []
        
        logger.info("Portfolio Rotation Trader initialized")
        logger.info(f"Max positions: 10, Capital: ${initial_capital:,.2f}")
    
    async def check_and_rotate(self):
        """Check for rotation opportunities"""
        try:
            # Check if enough time has passed since last rotation
            if self.last_rotation:
                time_since_rotation = (datetime.now() - self.last_rotation).seconds
                if time_since_rotation < self.min_rotation_interval:
                    logger.debug(f"Skipping rotation check, only {time_since_rotation}s since last rotation")
                    return
            
            # Get current positions
            current_positions = self.paper_trader.positions.copy()
            
            # Update rotation strategy positions
            self.rotation_strategy.positions = {}
            for symbol, pos in current_positions.items():
                entry_price = pos['entry_price']
                current_price = pos['current_price']
                return_pct = ((current_price - entry_price) / entry_price) * 100
                
                self.rotation_strategy.positions[symbol] = {
                    'entry_date': pos['entry_date'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'return_pct': return_pct
                }
            
            # Generate rotation signals
            logger.info("\n🔄 Checking for rotation opportunities...")
            signals = self.rotation_strategy.generate_rotation_signals(SACRED_SYMBOLS[:20])
            
            if not signals:
                return
            
            sell_list = signals.get('sell', [])
            buy_list = signals.get('buy', [])
            
            if not sell_list and not buy_list:
                logger.info("No rotation needed - portfolio is optimal")
                return
            
            # Execute sells first
            for symbol in sell_list:
                if symbol in current_positions:
                    # Get current price
                    scores = signals.get('scores', {})
                    if symbol in scores:
                        current_price = scores[symbol].current_price
                    else:
                        # Fetch current price
                        data = await self.data_fetcher.fetch_symbol_data(symbol, '1d', limit=1)
                        if data and not data.empty:
                            current_price = data.iloc[-1]['close']
                        else:
                            continue
                    
                    # Close position
                    success = self.paper_trader.close_position(symbol, current_price)
                    if success:
                        logger.info(f"🔴 ROTATED OUT: {symbol} @ ${current_price:.2f}")
                        self.rotation_history.append({
                            'timestamp': datetime.now(),
                            'action': 'SELL',
                            'symbol': symbol,
                            'price': current_price,
                            'reason': 'rotation'
                        })
            
            # Execute buys with available capital
            available_capital = self.paper_trader.available_balance
            
            for symbol in buy_list:
                if symbol not in self.paper_trader.positions:
                    # Get score and price
                    scores = signals.get('scores', {})
                    if symbol in scores:
                        stock_score = scores[symbol]
                        current_price = stock_score.current_price
                        score = stock_score.score
                    else:
                        continue
                    
                    # Calculate position size
                    position_size = self.rotation_strategy.calculate_position_size(
                        score, available_capital
                    )
                    
                    if position_size < 1000:  # Minimum $1000 position
                        continue
                    
                    # Calculate shares
                    shares = int(position_size / current_price)
                    if shares < 1:
                        continue
                    
                    # Open position
                    success = self.paper_trader.open_position(
                        symbol=symbol,
                        quantity=shares,
                        price=current_price,
                        position_type='LONG',
                        stop_loss=current_price * 0.95,  # 5% stop loss
                        take_profit=current_price * 1.15  # 15% take profit
                    )
                    
                    if success:
                        logger.info(f"🟢 ROTATED IN: {symbol} @ ${current_price:.2f} "
                                   f"(Score: {score:.3f}, Size: ${position_size:.0f})")
                        self.rotation_history.append({
                            'timestamp': datetime.now(),
                            'action': 'BUY',
                            'symbol': symbol,
                            'price': current_price,
                            'score': score,
                            'reason': 'rotation'
                        })
                        
                        available_capital = self.paper_trader.available_balance
            
            # Mark rotation time
            if sell_list or buy_list:
                self.last_rotation = datetime.now()
                logger.info(f"✅ Rotation complete: {len(sell_list)} sells, {len(buy_list)} buys")
            
        except Exception as e:
            logger.error(f"Error in rotation check: {e}")
            import traceback
            traceback.print_exc()
    
    def print_portfolio_status(self):
        """Print current portfolio status"""
        stats = self.paper_trader.get_account_stats()
        
        print("\n" + "="*80)
        print("DYNAMIC ROTATION PORTFOLIO STATUS")
        print("="*80)
        
        print(f"\n💰 Portfolio Value: ${stats['total_equity']:,.2f}")
        print(f"📈 Total Return: {stats['total_return_pct']:.2f}%")
        print(f"💵 Cash Available: ${stats['available_balance']:,.2f}")
        print(f"📊 Positions: {stats['positions_count']}/10")
        
        # Show current positions
        if self.paper_trader.positions:
            print("\n📊 Current Holdings:")
            positions_df = self.paper_trader.get_positions_df()
            if not positions_df.empty:
                positions_df = positions_df.sort_values('unrealized_pnl_pct', ascending=False)
                for _, pos in positions_df.iterrows():
                    emoji = "🟢" if pos['unrealized_pnl_pct'] > 0 else "🔴"
                    print(f"  {emoji} {pos['symbol']}: "
                          f"${pos['current_price']:.2f} ({pos['unrealized_pnl_pct']:+.2f}%)")
        
        # Recent rotations
        if self.rotation_history:
            print("\n🔄 Recent Rotations:")
            recent = self.rotation_history[-5:]
            for rot in recent:
                emoji = "🟢" if rot['action'] == 'BUY' else "🔴"
                print(f"  {emoji} {rot['timestamp'].strftime('%m/%d %H:%M')} - "
                      f"{rot['action']} {rot['symbol']} @ ${rot['price']:.2f}")
        
        print("\n" + "="*80)
    
    async def update_prices(self):
        """Update current prices for all positions"""
        for symbol in list(self.paper_trader.positions.keys()):
            try:
                # Fetch latest price
                data = await self.data_fetcher.fetch_symbol_data(symbol, '1h', limit=1)
                if data is not None and not data.empty:
                    current_price = data.iloc[-1]['close']
                    self.paper_trader.update_position_price(symbol, current_price)
                    
                    # Check stop loss/take profit
                    exit_reason = self.paper_trader.check_stop_loss_take_profit(symbol, current_price)
                    if exit_reason:
                        self.paper_trader.close_position(symbol, current_price)
                        logger.info(f"Position closed: {symbol} - {exit_reason}")
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
    
    async def start_trading(self):
        """Start rotation trading"""
        self.is_running = True
        logger.info("Starting portfolio rotation trading...")
        
        while self.is_running:
            try:
                # Update prices
                await self.update_prices()
                
                # Check for rotations
                await self.check_and_rotate()
                
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
        logger.info("Stopping portfolio rotation trading...")
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        stats = self.paper_trader.get_account_stats()
        
        # Calculate rotation metrics
        total_rotations = len([r for r in self.rotation_history if r['action'] == 'SELL'])
        
        report = {
            'portfolio_metrics': stats,
            'rotation_metrics': {
                'total_rotations': total_rotations,
                'avg_holding_period': self._calculate_avg_holding_period(),
                'rotation_win_rate': self._calculate_rotation_win_rate()
            },
            'current_positions': self.paper_trader.get_positions_df().to_dict('records') if self.paper_trader.positions else []
        }
        
        return report
    
    def _calculate_avg_holding_period(self) -> float:
        """Calculate average holding period in days"""
        if not self.paper_trader.closed_trades:
            return 0
        
        holding_periods = []
        for trade in self.paper_trader.closed_trades:
            period = (trade['exit_date'] - trade['entry_date']).days
            holding_periods.append(period)
        
        return sum(holding_periods) / len(holding_periods) if holding_periods else 0
    
    def _calculate_rotation_win_rate(self) -> float:
        """Calculate win rate for rotated positions"""
        if not self.paper_trader.closed_trades:
            return 0
        
        wins = sum(1 for t in self.paper_trader.closed_trades if t['pnl'] > 0)
        total = len(self.paper_trader.closed_trades)
        
        return (wins / total * 100) if total > 0 else 0


async def main():
    """Run rotation trading"""
    trader = PortfolioRotationTrader(initial_capital=100000)
    
    print("\n" + "="*60)
    print("🔄 DYNAMIC TOP 10 ROTATION STRATEGY")
    print("="*60)
    print("\n📊 Features:")
    print("  • Keeps best 10 stocks by composite score")
    print("  • Rotates weak performers for high potential stocks")
    print("  • Score-based position sizing (8-15%)")
    print("  • Automatic profit taking and stop loss")
    print("\n💡 Rotation checks every 5 minutes")
    print("⏰ Minimum 1 hour between rotations\n")
    
    try:
        await trader.start_trading()
    except KeyboardInterrupt:
        trader.stop_trading()
    
    # Final report
    report = trader.get_performance_report()
    print("\n📊 Final Performance Report:")
    print(f"Total Return: {report['portfolio_metrics']['total_return_pct']:.2f}%")
    print(f"Total Rotations: {report['rotation_metrics']['total_rotations']}")
    print(f"Win Rate: {report['rotation_metrics']['rotation_win_rate']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())