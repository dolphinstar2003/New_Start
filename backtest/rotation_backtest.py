"""
Dynamic Rotation Strategy Backtest
Tests the top 10 rotation strategy with historical data
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
from trading.dynamic_portfolio_rotation import DynamicRotationStrategy


class RotationBacktest:
    """Backtest for dynamic rotation strategy"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.rotation_strategy = DynamicRotationStrategy(initial_capital, max_positions=10)
        
        # Backtest state
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.rotation_history = []
        
        # Backtest parameters
        self.commission = 0.001  # 0.1% commission
        self.min_rotation_days = 2  # Minimum days between rotations
        
    async def run(self, days: int = 30, symbols: List[str] = None) -> Dict:
        """Run rotation backtest"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:20]
        
        logger.info(f"Running {days}-day rotation backtest")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # Extra days for indicators
        
        # Track daily performance
        current_date = end_date - timedelta(days=days)
        last_rotation_date = None
        
        # Simulate each day
        for day_offset in range(days):
            sim_date = current_date + timedelta(days=day_offset)
            
            # Update current prices for positions
            await self._update_position_prices(sim_date)
            
            # Check if we should rotate
            should_rotate = True
            if last_rotation_date:
                days_since_rotation = (sim_date - last_rotation_date).days
                if days_since_rotation < self.min_rotation_days:
                    should_rotate = False
            
            if should_rotate:
                # Calculate scores for all symbols
                scores = await self._calculate_historical_scores(symbols, sim_date)
                
                if scores:
                    # Check for rotation opportunities
                    rotation_trades = self._check_rotation(scores, sim_date)
                    
                    if rotation_trades:
                        last_rotation_date = sim_date
                        self.rotation_history.append({
                            'date': sim_date,
                            'trades': len(rotation_trades),
                            'details': rotation_trades
                        })
            
            # Record daily value
            portfolio_value = self._calculate_portfolio_value(sim_date)
            self.daily_values.append({
                'date': sim_date,
                'value': portfolio_value,
                'positions': len(self.positions),
                'cash': self.capital
            })
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    async def _calculate_historical_scores(self, symbols: List[str], target_date: datetime) -> List:
        """Calculate scores using historical data up to target date"""
        scores = []
        
        for symbol in symbols:
            try:
                # Load historical data
                clean_symbol = symbol.replace('.IS', '')
                data_file = DATA_DIR / 'raw' / '1d' / f"{clean_symbol}_1d_raw.csv"
                
                if not data_file.exists():
                    continue
                
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df['Date'])
                # Remove timezone info for comparison
                df['Date'] = df['Date'].dt.tz_localize(None)
                df = df.sort_values('Date')
                
                # Filter to data available at target_date
                df = df[df['Date'] <= target_date]
                
                if len(df) < 20:
                    continue
                
                # Get latest 50 days for calculations
                df = df.tail(50)
                
                # Calculate score components
                latest = df.iloc[-1]
                current_price = latest['close']
                
                # Momentum (5, 10, 20 day returns)
                momentum_score = 0
                if len(df) >= 20:
                    ret_5d = (df.iloc[-1]['close'] / df.iloc[-5]['close'] - 1) if len(df) >= 5 else 0
                    ret_10d = (df.iloc[-1]['close'] / df.iloc[-10]['close'] - 1) if len(df) >= 10 else 0
                    ret_20d = (df.iloc[-1]['close'] / df.iloc[-20]['close'] - 1) if len(df) >= 20 else 0
                    momentum_score = np.clip((ret_5d * 0.5 + ret_10d * 0.3 + ret_20d * 0.2 + 0.1) / 0.2, 0, 1)
                
                # Volatility (20-day)
                returns = df['close'].pct_change().dropna()
                volatility = returns.tail(20).std() if len(returns) >= 20 else 0.02
                volatility_score = 1 - min(volatility / 0.04, 1.0)
                
                # Trend (simple MA crossover)
                ma_10 = df['close'].rolling(10).mean().iloc[-1]
                ma_20 = df['close'].rolling(20).mean().iloc[-1]
                trend_score = 0.7 if ma_10 > ma_20 else 0.3
                
                # Oversold/Overbought (RSI-like)
                rsi_period = 14
                if len(returns) >= rsi_period:
                    gains = returns[returns > 0].tail(rsi_period)
                    losses = -returns[returns < 0].tail(rsi_period)
                    avg_gain = gains.mean() if len(gains) > 0 else 0
                    avg_loss = losses.mean() if len(losses) > 0 else 0
                    rs = avg_gain / avg_loss if avg_loss > 0 else 1
                    rsi = 100 - (100 / (1 + rs))
                    indicator_score = 0.8 if rsi < 30 else (0.2 if rsi > 70 else 0.5)
                else:
                    indicator_score = 0.5
                
                # Calculate total score
                total_score = (
                    momentum_score * 0.30 +
                    trend_score * 0.25 +
                    volatility_score * 0.20 +
                    indicator_score * 0.25
                )
                
                scores.append({
                    'symbol': symbol,
                    'score': total_score,
                    'price': current_price,
                    'momentum': momentum_score,
                    'trend': trend_score,
                    'volatility': volatility_score,
                    'indicators': indicator_score
                })
                
            except Exception as e:
                logger.error(f"Error calculating score for {symbol}: {e}")
                continue
        
        # Sort by score
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores
    
    def _check_rotation(self, scores: List[Dict], sim_date: datetime) -> List[Dict]:
        """Check and execute rotation trades"""
        trades = []
        
        # Get top 10 symbols
        top_10_symbols = [s['symbol'] for s in scores[:10]]
        
        # Check current positions for sells
        sells = []
        for symbol, position in list(self.positions.items()):
            # Get current score
            stock_score = next((s for s in scores if s['symbol'] == symbol), None)
            if not stock_score:
                continue
            
            current_price = stock_score['price']
            return_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
            
            # Sell if:
            # 1. Not in top 10
            if symbol not in top_10_symbols:
                sells.append((symbol, current_price, 'not_top_10'))
            # 2. Stop loss
            elif return_pct < -5:
                sells.append((symbol, current_price, 'stop_loss'))
            # 3. Profit taking with weak momentum
            elif return_pct > 12 and stock_score['momentum'] < 0.4:
                sells.append((symbol, current_price, 'profit_taking'))
        
        # Execute sells
        for symbol, price, reason in sells[:3]:  # Max 3 sells
            position = self.positions[symbol]
            return_pct = ((price - position['entry_price']) / position['entry_price']) * 100
            profit = position['size'] * (return_pct / 100)
            
            # Update capital
            self.capital += position['size'] + profit - (position['size'] * self.commission)
            
            # Record trade
            trade = {
                'date': sim_date,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'shares': position['shares'],
                'return_pct': return_pct,
                'profit': profit,
                'reason': reason
            }
            self.trades.append(trade)
            trades.append(trade)
            
            # Remove position
            del self.positions[symbol]
        
        # Check for buys
        buys = []
        available_capital = self.capital * 0.95  # Keep 5% cash
        
        for stock in scores[:15]:  # Look at top 15
            symbol = stock['symbol']
            if symbol not in self.positions and len(self.positions) < 10:
                # Calculate position size based on score
                position_pct = 0.08 + (0.07 * stock['score'])  # 8-15%
                position_size = min(available_capital * position_pct, available_capital / 3)
                
                if position_size > 1000:  # Minimum position
                    buys.append((symbol, stock['price'], stock['score'], position_size))
        
        # Execute buys
        for symbol, price, score, size in buys[:3]:  # Max 3 buys
            shares = int(size / price)
            if shares < 1:
                continue
            
            actual_size = shares * price
            
            # Update capital
            self.capital -= actual_size + (actual_size * self.commission)
            
            # Create position
            self.positions[symbol] = {
                'entry_date': sim_date,
                'entry_price': price,
                'shares': shares,
                'size': actual_size,
                'score': score
            }
            
            # Record trade
            trade = {
                'date': sim_date,
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'size': actual_size,
                'score': score
            }
            self.trades.append(trade)
            trades.append(trade)
        
        return trades
    
    async def _update_position_prices(self, sim_date: datetime):
        """Update position prices and check exits"""
        for symbol, position in list(self.positions.items()):
            try:
                # Load price data
                clean_symbol = symbol.replace('.IS', '')
                data_file = DATA_DIR / 'raw' / '1d' / f"{clean_symbol}_1d_raw.csv"
                
                if not data_file.exists():
                    continue
                
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df['Date'])
                # Remove timezone info for comparison
                df['Date'] = df['Date'].dt.tz_localize(None)
                df = df[df['Date'] <= sim_date].tail(1)
                
                if not df.empty:
                    current_price = df.iloc[0]['close']
                    position['current_price'] = current_price
                    position['return_pct'] = ((current_price - position['entry_price']) / position['entry_price']) * 100
                
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
    
    def _calculate_portfolio_value(self, sim_date: datetime) -> float:
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
                'backtest_engine': 'rotation'
            }
        
        # Calculate returns
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
        
        # Rotation statistics
        total_rotations = len(self.rotation_history)
        avg_positions = np.mean([d['positions'] for d in self.daily_values]) if self.daily_values else 0
        
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
            'total_rotations': total_rotations,
            'avg_positions': avg_positions,
            'profit_factor': abs(sum(t.get('profit', 0) for t in winning_trades) / 
                                sum(t.get('profit', 0) for t in sell_trades if t.get('profit', 0) < 0)) 
                            if any(t.get('profit', 0) < 0 for t in sell_trades) else 0,
            'backtest_engine': 'rotation'
        }


async def run_rotation_backtest(days: int = 30, symbols: List[str] = None) -> Dict:
    """Run rotation backtest"""
    backtest = RotationBacktest()
    return await backtest.run(days, symbols)


if __name__ == "__main__":
    async def test():
        for days in [30, 60, 90]:
            print(f"\n{'='*60}")
            print(f"Running {days}-day rotation backtest")
            print(f"{'='*60}")
            
            result = await run_rotation_backtest(days)
            
            print(f"\nResults:")
            print(f"Total Return: {result['total_return']:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"Win Rate: {result['win_rate']:.1f}%")
            print(f"Total Trades: {result['total_trades']}")
            print(f"Total Rotations: {result['total_rotations']}")
            print(f"Avg Positions: {result['avg_positions']:.1f}")
    
    asyncio.run(test())