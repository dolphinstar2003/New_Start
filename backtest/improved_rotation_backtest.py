"""
Improved Rotation Backtest
Simplified but effective rotation strategy backtest
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from loguru import logger
import asyncio

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from trading.improved_rotation_strategy import ImprovedRotationStrategy, ImprovedStockScore


class ImprovedRotationBacktest:
    """Backtest for improved rotation strategy"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy = ImprovedRotationStrategy(initial_capital, max_positions=10)
        
        # Backtest state
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.rotation_history = []
        
        # Performance tracking
        self.exit_reasons = {}
        
        # Backtest parameters
        self.commission = 0.001  # 0.1% commission
        self.min_rotation_days = 2
        
    async def run(self, days: int = 30, symbols: List[str] = None) -> Dict:
        """Run improved rotation backtest"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:20]
        
        logger.info(f"Running {days}-day improved rotation backtest")
        
        # Calculate date range
        end_date = datetime.now()
        current_date = end_date - timedelta(days=days)
        last_rotation_date = None
        
        # Simulate each day
        for day_offset in range(days):
            sim_date = current_date + timedelta(days=day_offset)
            
            # Check if we should rotate
            should_rotate = True
            if last_rotation_date:
                days_since_rotation = (sim_date - last_rotation_date).days
                if days_since_rotation < self.min_rotation_days:
                    should_rotate = False
            
            if should_rotate:
                # Get signals
                rotation_trades = await self._check_rotation(symbols, sim_date)
                
                if rotation_trades:
                    last_rotation_date = sim_date
                    self.rotation_history.append({
                        'date': sim_date,
                        'trades': len(rotation_trades)
                    })
            
            # Update position prices
            await self._update_prices(sim_date)
            
            # Record daily value
            portfolio_value = self._calculate_portfolio_value()
            self.daily_values.append({
                'date': sim_date,
                'value': portfolio_value,
                'positions': len(self.positions),
                'cash': self.capital
            })
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    async def _check_rotation(self, symbols: List[str], sim_date: datetime) -> List[Dict]:
        """Check and execute rotation"""
        # Update strategy positions
        self.strategy.positions = self.positions.copy()
        self.strategy.trailing_stops = {}
        
        # Generate signals
        signals = self.strategy.generate_signals(symbols)
        
        trades = []
        
        # Execute sells
        for symbol in signals['sell']:
            if symbol in self.positions:
                # Get current price
                current_price = await self._get_price(symbol, sim_date)
                if current_price is None:
                    continue
                
                position = self.positions[symbol]
                return_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
                profit = position['size'] * (return_pct / 100)
                
                # Update capital
                self.capital += position['size'] + profit - (position['size'] * self.commission)
                
                # Track exit reason
                reason = 'rotation'
                # Get the sell candidates list from identify_rotation_trades
                sell_candidates, _ = self.strategy.identify_rotation_trades(
                    list(signals['scores'].values()))
                # The function returns a list of symbols, not tuples with reasons
                # So we need to check the actual exit conditions
                if symbol in self.positions:
                    stock_score = signals['scores'].get(symbol)
                    if stock_score:
                        _, exit_reason = self.strategy.check_exit_conditions(
                            self.positions[symbol], stock_score)
                        if exit_reason:
                            reason = exit_reason
                
                if reason not in self.exit_reasons:
                    self.exit_reasons[reason] = {'count': 0, 'total_return': 0}
                self.exit_reasons[reason]['count'] += 1
                self.exit_reasons[reason]['total_return'] += return_pct
                
                # Record trade
                trade = {
                    'date': sim_date,
                    'symbol': symbol,
                    'action': 'SELL',
                    'price': current_price,
                    'return_pct': return_pct,
                    'profit': profit,
                    'reason': reason
                }
                self.trades.append(trade)
                trades.append(trade)
                
                # Remove position
                del self.positions[symbol]
        
        # Execute buys
        available_capital = self.capital * 0.95  # Keep 5% buffer
        
        for symbol in signals['buy']:
            if symbol not in self.positions and len(self.positions) < 10:
                # Get score and price
                stock_score = signals['scores'].get(symbol)
                if not stock_score:
                    continue
                
                current_price = await self._get_price(symbol, sim_date)
                if current_price is None:
                    continue
                
                # Calculate position size
                position_size = self.strategy.calculate_position_size(stock_score, available_capital)
                
                if position_size < 1000:  # Min $1000
                    continue
                
                shares = int(position_size / current_price)
                if shares < 1:
                    continue
                
                actual_size = shares * current_price
                
                # Update capital
                self.capital -= actual_size + (actual_size * self.commission)
                available_capital = self.capital * 0.95
                
                # Create position
                self.positions[symbol] = {
                    'symbol': symbol,
                    'entry_date': sim_date,
                    'entry_price': current_price,
                    'shares': shares,
                    'size': actual_size,
                    'score': stock_score.total_score
                }
                
                # Record trade
                trade = {
                    'date': sim_date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': current_price,
                    'size': actual_size,
                    'score': stock_score.total_score
                }
                self.trades.append(trade)
                trades.append(trade)
        
        return trades
    
    async def _get_price(self, symbol: str, sim_date: datetime) -> float:
        """Get historical price for date"""
        try:
            clean_symbol = symbol.replace('.IS', '')
            data_file = DATA_DIR / 'raw' / '1d' / f"{clean_symbol}_1d_raw.csv"
            
            if not data_file.exists():
                return None
            
            df = pd.read_csv(data_file)
            df['Date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df['Date'])
            df['Date'] = df['Date'].dt.tz_localize(None)
            df = df[df['Date'] <= sim_date].tail(1)
            
            if not df.empty:
                return df.iloc[0]['close']
            
            return None
        except:
            return None
    
    async def _update_prices(self, sim_date: datetime):
        """Update position prices"""
        for symbol, position in self.positions.items():
            current_price = await self._get_price(symbol, sim_date)
            if current_price:
                position['current_price'] = current_price
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.get('current_price', pos['entry_price']) * pos['shares'] 
                             for pos in self.positions.values())
        return self.capital + positions_value
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest metrics"""
        if not self.daily_values:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'backtest_engine': 'improved_rotation'
            }
        
        # Basic metrics
        final_value = self.daily_values[-1]['value']
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Daily returns
        values = pd.Series([d['value'] for d in self.daily_values])
        daily_returns = values.pct_change().dropna()
        
        # Sharpe ratio
        sharpe_ratio = 0
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
        
        # Trade statistics
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        winning_trades = [t for t in sell_trades if t.get('return_pct', 0) > 0]
        
        win_rate = (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0
        
        # Average returns
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in sell_trades if t.get('return_pct', 0) < 0]) if sell_trades else 0
        
        # Exit reason statistics
        exit_stats = {}
        for reason, stats in self.exit_reasons.items():
            if stats['count'] > 0:
                exit_stats[reason] = {
                    'count': stats['count'],
                    'avg_return': stats['total_return'] / stats['count']
                }
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'buy_trades': len([t for t in self.trades if t['action'] == 'BUY']),
            'sell_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(sell_trades) - len(winning_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(sum(t.get('profit', 0) for t in winning_trades) / 
                                sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) < 0)) 
                            if any(t.get('profit', 0) < 0 for t in sell_trades) else 0,
            'total_rotations': len(self.rotation_history),
            'avg_positions': np.mean([d['positions'] for d in self.daily_values]),
            'exit_reasons': exit_stats,
            'backtest_engine': 'improved_rotation'
        }


async def run_improved_rotation_backtest(days: int = 30, symbols: List[str] = None) -> Dict:
    """Run improved rotation backtest"""
    backtest = ImprovedRotationBacktest()
    return await backtest.run(days, symbols)


if __name__ == "__main__":
    async def test():
        print("\n" + "="*60)
        print("IMPROVED ROTATION STRATEGY BACKTEST")
        print("="*60)
        
        for days in [30, 60, 90]:
            print(f"\n📊 Testing {days}-day period...")
            
            result = await run_improved_rotation_backtest(days)
            
            print(f"\nResults:")
            print(f"  Return: {result['total_return']:+.2f}%")
            print(f"  Sharpe: {result['sharpe_ratio']:.2f}")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Max DD: {result['max_drawdown']:.2f}%")
            print(f"  Rotations: {result['total_rotations']}")
            
            if result['exit_reasons']:
                print(f"\n  Exit Reasons:")
                for reason, stats in result['exit_reasons'].items():
                    print(f"    {reason}: {stats['count']} trades, avg: {stats['avg_return']:.1f}%")
    
    asyncio.run(test())