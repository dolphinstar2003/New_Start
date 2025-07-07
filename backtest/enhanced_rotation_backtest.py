"""
Enhanced Rotation Strategy Backtest
Tests the improved rotation strategy with all enhancements
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
from trading.enhanced_rotation_strategy import EnhancedRotationStrategy, EnhancedStockScore


class EnhancedRotationBacktest:
    """Backtest for enhanced rotation strategy"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.strategy = EnhancedRotationStrategy(initial_capital, max_positions=10)
        
        # Backtest state
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.rotation_history = []
        self.partial_exits = {}
        
        # Performance tracking
        self.regime_performance = {'bull': [], 'bear': [], 'neutral': []}
        self.exit_reason_stats = {}
        
        # Backtest parameters
        self.commission = 0.001  # 0.1% commission
        
    async def run(self, days: int = 30, symbols: List[str] = None) -> Dict:
        """Run enhanced rotation backtest"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:20]
        
        logger.info(f"Running {days}-day enhanced rotation backtest")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 60)  # Extra days for indicators
        
        # Track performance
        current_date = end_date - timedelta(days=days)
        last_rotation_date = None
        market_volatility = 0.02  # Default 2%
        
        # Simulate each day
        for day_offset in range(days):
            sim_date = current_date + timedelta(days=day_offset)
            
            # Update market volatility
            market_volatility = await self._calculate_market_volatility(symbols, sim_date)
            
            # Calculate rotation interval dynamically
            rotation_interval = self.strategy.calculate_dynamic_rotation_interval(market_volatility)
            
            # Check if we should rotate
            should_rotate = True
            if last_rotation_date:
                days_since_rotation = (sim_date - last_rotation_date).days
                if days_since_rotation < rotation_interval:
                    should_rotate = False
            
            # Always check exits
            await self._check_exits(symbols, sim_date)
            
            if should_rotate:
                # Calculate enhanced scores
                scores = await self._calculate_historical_enhanced_scores(symbols, sim_date)
                
                if scores:
                    # Update market regime
                    self.strategy._update_market_regime(symbols)
                    
                    # Check for rotation opportunities
                    rotation_trades = self._check_enhanced_rotation(scores, sim_date)
                    
                    if rotation_trades:
                        last_rotation_date = sim_date
                        self.rotation_history.append({
                            'date': sim_date,
                            'trades': len(rotation_trades),
                            'regime': self.strategy.market_regime,
                            'volatility': market_volatility,
                            'details': rotation_trades
                        })
            
            # Record daily value
            portfolio_value = self._calculate_portfolio_value(sim_date)
            self.daily_values.append({
                'date': sim_date,
                'value': portfolio_value,
                'positions': len(self.positions),
                'cash': self.capital,
                'regime': self.strategy.market_regime
            })
            
            # Track regime performance
            if self.daily_values:
                daily_return = (portfolio_value / self.daily_values[-2]['value'] - 1) if len(self.daily_values) > 1 else 0
                self.regime_performance[self.strategy.market_regime].append(daily_return)
        
        # Calculate final metrics
        return self._calculate_enhanced_metrics()
    
    async def _calculate_historical_enhanced_scores(self, symbols: List[str], target_date: datetime) -> List[EnhancedStockScore]:
        """Calculate enhanced scores using historical data"""
        scores = []
        
        # Simplified scoring for backtest (full implementation would load all indicators)
        for symbol in symbols:
            try:
                clean_symbol = symbol.replace('.IS', '')
                data_file = DATA_DIR / 'raw' / '1d' / f"{clean_symbol}_1d_raw.csv"
                
                if not data_file.exists():
                    continue
                
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df['Date'])
                df['Date'] = df['Date'].dt.tz_localize(None)
                df = df.sort_values('Date')
                df = df[df['Date'] <= target_date]
                
                if len(df) < 50:
                    continue
                
                # Get latest data
                df = df.tail(50)
                latest = df.iloc[-1]
                current_price = latest['close']
                
                # Calculate components
                momentum_score = self._calc_hist_momentum(df)
                trend_score = self._calc_hist_trend(df)
                volatility_score = self._calc_hist_volatility(df)
                relative_strength = self._calc_hist_relative_strength(df, symbols)
                volume_score = self._calc_hist_volume(df)
                
                # ATR calculation
                atr = self._calc_hist_atr(df)
                
                # Kelly fraction
                kelly = self._calc_hist_kelly(df)
                
                # Total score
                total_score = (
                    momentum_score * 0.25 +
                    trend_score * 0.20 +
                    volatility_score * 0.15 +
                    relative_strength * 0.20 +
                    volume_score * 0.05 +
                    0.15  # Placeholder for indicators
                )
                
                scores.append(EnhancedStockScore(
                    symbol=symbol,
                    total_score=total_score,
                    momentum_score=momentum_score,
                    trend_score=trend_score,
                    volatility_score=volatility_score,
                    relative_strength=relative_strength,
                    indicator_score=0.5,
                    ml_score=0,
                    volume_score=volume_score,
                    current_price=current_price,
                    atr=atr,
                    signals={'realistic': 1, 'hierarchical': 1},
                    kelly_fraction=kelly
                ))
                
            except Exception as e:
                logger.error(f"Error calculating score for {symbol}: {e}")
                continue
        
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _calc_hist_momentum(self, df: pd.DataFrame) -> float:
        """Calculate historical momentum score"""
        if len(df) < 20:
            return 0.5
        
        prices = df['close'].values
        ret_5d = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        ret_10d = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        ret_20d = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
        
        momentum = (ret_5d * 0.5 + ret_10d * 0.3 + ret_20d * 0.2)
        return np.clip((momentum + 0.1) / 0.2, 0, 1)
    
    def _calc_hist_trend(self, df: pd.DataFrame) -> float:
        """Calculate historical trend score"""
        if len(df) < 20:
            return 0.5
        
        prices = df['close'].values
        ma_10 = prices[-10:].mean()
        ma_20 = prices[-20:].mean()
        
        if len(prices) >= 50:
            ma_50 = prices[-50:].mean()
            if ma_10 > ma_20 > ma_50:
                return 0.9
            elif ma_10 > ma_20:
                return 0.7
            else:
                return 0.3
        else:
            return 0.7 if ma_10 > ma_20 else 0.3
    
    def _calc_hist_volatility(self, df: pd.DataFrame) -> float:
        """Calculate historical volatility score"""
        if len(df) < 20:
            return 0.5
        
        returns = df['close'].pct_change().dropna()
        volatility = returns.tail(20).std()
        
        # Lower volatility = higher score
        return 1 - min(volatility / 0.03, 1.0)
    
    def _calc_hist_relative_strength(self, df: pd.DataFrame, symbols: List[str]) -> float:
        """Calculate relative strength"""
        if len(df) < 20:
            return 0.5
        
        stock_return = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
        # Simplified: assume market return is 0.01 (1%)
        market_return = 0.01
        
        rs = stock_return / market_return if market_return != 0 else 1
        return np.clip((rs - 0.8) / 0.4, 0, 1)
    
    def _calc_hist_volume(self, df: pd.DataFrame) -> float:
        """Calculate volume score"""
        if 'volume' not in df.columns or len(df) < 20:
            return 0.5
        
        recent_vol = df['volume'].tail(5).mean()
        avg_vol = df['volume'].tail(20).mean()
        
        if avg_vol > 0:
            vol_ratio = recent_vol / avg_vol
            return np.clip(vol_ratio / 2, 0, 1)
        return 0.5
    
    def _calc_hist_atr(self, df: pd.DataFrame) -> float:
        """Calculate ATR"""
        if len(df) < 14:
            return df['close'].iloc[-1] * 0.02
        
        # Simplified ATR using high-low range
        if 'high' in df.columns and 'low' in df.columns:
            ranges = df['high'] - df['low']
            atr = ranges.tail(14).mean()
        else:
            # Use 2% of price as default
            atr = df['close'].iloc[-1] * 0.02
        
        return atr
    
    def _calc_hist_kelly(self, df: pd.DataFrame) -> float:
        """Calculate Kelly fraction"""
        if len(df) < 30:
            return 0.1
        
        returns = df['close'].pct_change().dropna()
        wins = returns[returns > 0.01]
        losses = returns[returns < -0.01]
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / (len(wins) + len(losses))
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            
            if avg_loss > 0:
                b = avg_win / avg_loss
                q = 1 - win_rate
                kelly = (win_rate * b - q) / b
                return max(0, min(kelly * 0.25, 0.25))  # Conservative Kelly
        
        return 0.1
    
    async def _calculate_market_volatility(self, symbols: List[str], sim_date: datetime) -> float:
        """Calculate market volatility"""
        volatilities = []
        
        for symbol in symbols[:10]:  # Top 10 as proxy
            try:
                clean_symbol = symbol.replace('.IS', '')
                data_file = DATA_DIR / 'raw' / '1d' / f"{clean_symbol}_1d_raw.csv"
                
                if not data_file.exists():
                    continue
                
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df['Date'])
                df['Date'] = df['Date'].dt.tz_localize(None)
                df = df[df['Date'] <= sim_date].tail(20)
                
                if len(df) >= 10:
                    returns = df['close'].pct_change().dropna()
                    vol = returns.std()
                    volatilities.append(vol)
            except:
                continue
        
        return np.mean(volatilities) if volatilities else 0.02
    
    async def _check_exits(self, symbols: List[str], sim_date: datetime):
        """Check exit conditions for all positions"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current score
                scores = await self._calculate_historical_enhanced_scores([symbol], sim_date)
                if not scores:
                    continue
                
                current_score = scores[0]
                
                # Enhanced exit check
                should_exit, reason, exit_percent = self.strategy.check_enhanced_exit_conditions(
                    position, current_score
                )
                
                if should_exit:
                    self._execute_exit(symbol, current_score.current_price, reason, exit_percent, sim_date)
                
            except Exception as e:
                logger.error(f"Error checking exit for {symbol}: {e}")
    
    def _execute_exit(self, symbol: str, price: float, reason: str, exit_percent: float, sim_date: datetime):
        """Execute position exit"""
        position = self.positions[symbol]
        
        # Calculate return
        return_pct = ((price - position['entry_price']) / position['entry_price']) * 100
        
        # Calculate exit size
        exit_size = position['size'] * exit_percent
        remaining_size = position['size'] * (1 - exit_percent)
        
        # Calculate profit
        profit = exit_size * (return_pct / 100)
        
        # Update capital
        self.capital += exit_size + profit - (exit_size * self.commission)
        
        # Record trade
        trade = {
            'date': sim_date,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'size': exit_size,
            'return_pct': return_pct,
            'profit': profit,
            'reason': reason,
            'exit_percent': exit_percent
        }
        self.trades.append(trade)
        
        # Track exit reasons
        if reason not in self.exit_reason_stats:
            self.exit_reason_stats[reason] = {'count': 0, 'total_return': 0}
        self.exit_reason_stats[reason]['count'] += 1
        self.exit_reason_stats[reason]['total_return'] += return_pct
        
        # Update or remove position
        if exit_percent < 1.0:
            # Partial exit
            position['size'] = remaining_size
            self.partial_exits[symbol] = True
        else:
            # Full exit
            del self.positions[symbol]
            if symbol in self.partial_exits:
                del self.partial_exits[symbol]
    
    def _check_enhanced_rotation(self, scores: List[EnhancedStockScore], sim_date: datetime) -> List[Dict]:
        """Check and execute enhanced rotation trades"""
        trades = []
        
        # Get top 10 symbols
        top_10_symbols = [s.symbol for s in scores[:10]]
        
        # Enhanced sell logic
        sells = []
        for symbol, position in list(self.positions.items()):
            stock_score = next((s for s in scores if s.symbol == symbol), None)
            if not stock_score:
                continue
            
            current_price = stock_score.current_price
            return_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
            
            # Sell if not in top 10 and there's a better opportunity
            if symbol not in top_10_symbols:
                # Find replacement
                for new_stock in scores[:15]:
                    if new_stock.symbol not in self.positions:
                        score_diff = new_stock.total_score - stock_score.total_score
                        if score_diff > 0.1:  # 10% better score
                            sells.append((symbol, current_price, 'rotation', new_stock.symbol))
                            break
        
        # Execute sells
        for symbol, price, reason, replacement in sells[:3]:
            position = self.positions[symbol]
            return_pct = ((price - position['entry_price']) / position['entry_price']) * 100
            profit = position['size'] * (return_pct / 100)
            
            self.capital += position['size'] + profit - (position['size'] * self.commission)
            
            trade = {
                'date': sim_date,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'size': position['size'],
                'return_pct': return_pct,
                'profit': profit,
                'reason': reason,
                'replacement': replacement
            }
            self.trades.append(trade)
            trades.append(trade)
            
            del self.positions[symbol]
        
        # Enhanced buy logic
        buys = []
        available_capital = self.capital * 0.95  # Keep 5% buffer
        
        for stock in scores[:15]:
            if stock.symbol not in self.positions and len(self.positions) < 10:
                # Enhanced position sizing
                position_size = self.strategy.enhanced_position_sizing(stock, available_capital)
                
                if position_size > 1000:  # Minimum position
                    buys.append((stock, position_size))
        
        # Execute buys
        for stock, size in buys[:3]:
            shares = int(size / stock.current_price)
            if shares < 1:
                continue
            
            actual_size = shares * stock.current_price
            self.capital -= actual_size + (actual_size * self.commission)
            
            self.positions[stock.symbol] = {
                'symbol': stock.symbol,
                'entry_date': sim_date,
                'entry_price': stock.current_price,
                'shares': shares,
                'size': actual_size,
                'score': stock.total_score,
                'kelly': stock.kelly_fraction
            }
            
            trade = {
                'date': sim_date,
                'symbol': stock.symbol,
                'action': 'BUY',
                'price': stock.current_price,
                'size': actual_size,
                'score': stock.total_score,
                'kelly': stock.kelly_fraction
            }
            self.trades.append(trade)
            trades.append(trade)
        
        return trades
    
    async def _update_position_prices(self, sim_date: datetime):
        """Update position prices"""
        for symbol, position in list(self.positions.items()):
            try:
                clean_symbol = symbol.replace('.IS', '')
                data_file = DATA_DIR / 'raw' / '1d' / f"{clean_symbol}_1d_raw.csv"
                
                if not data_file.exists():
                    continue
                
                df = pd.read_csv(data_file)
                df['Date'] = pd.to_datetime(df['datetime'] if 'datetime' in df.columns else df['Date'])
                df['Date'] = df['Date'].dt.tz_localize(None)
                df = df[df['Date'] <= sim_date].tail(1)
                
                if not df.empty:
                    position['current_price'] = df.iloc[0]['close']
                
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
    
    def _calculate_portfolio_value(self, sim_date: datetime) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.get('current_price', pos['entry_price']) * pos['shares'] 
                             for pos in self.positions.values())
        return self.capital + positions_value
    
    def _calculate_enhanced_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics"""
        if not self.daily_values:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'total_trades': 0,
                'backtest_engine': 'enhanced_rotation'
            }
        
        # Basic metrics
        final_value = self.daily_values[-1]['value']
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Daily returns
        values = pd.Series([d['value'] for d in self.daily_values])
        daily_returns = values.pct_change().dropna()
        
        # Enhanced Sharpe ratio
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
        
        # Enhanced metrics
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in sell_trades if t.get('return_pct', 0) < 0]) if sell_trades else 0
        
        # Exit reason analysis
        exit_stats = {}
        for reason, stats in self.exit_reason_stats.items():
            if stats['count'] > 0:
                exit_stats[reason] = {
                    'count': stats['count'],
                    'avg_return': stats['total_return'] / stats['count']
                }
        
        # Regime performance
        regime_returns = {}
        for regime, returns in self.regime_performance.items():
            if returns:
                regime_returns[regime] = {
                    'days': len(returns),
                    'avg_return': np.mean(returns) * 100,
                    'total_return': (np.prod([1 + r for r in returns]) - 1) * 100
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
            'partial_exits': len([t for t in self.trades if t.get('exit_percent', 1) < 1]),
            'exit_reasons': exit_stats,
            'regime_performance': regime_returns,
            'backtest_engine': 'enhanced_rotation'
        }


async def run_enhanced_rotation_backtest(days: int = 30, symbols: List[str] = None) -> Dict:
    """Run enhanced rotation backtest"""
    backtest = EnhancedRotationBacktest()
    return await backtest.run(days, symbols)


if __name__ == "__main__":
    async def test():
        print("\n" + "="*60)
        print("ENHANCED ROTATION STRATEGY BACKTEST")
        print("="*60)
        
        for days in [30, 60, 90]:
            print(f"\n📊 Running {days}-day enhanced backtest...")
            
            result = await run_enhanced_rotation_backtest(days)
            
            print(f"\nResults:")
            print(f"Total Return: {result['total_return']:.2f}%")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"Win Rate: {result['win_rate']:.1f}%")
            print(f"Partial Exits: {result['partial_exits']}")
            
            if result['exit_reasons']:
                print("\nExit Reasons:")
                for reason, stats in result['exit_reasons'].items():
                    print(f"  {reason}: {stats['count']} trades, avg return: {stats['avg_return']:.2f}%")
            
            if result['regime_performance']:
                print("\nRegime Performance:")
                for regime, perf in result['regime_performance'].items():
                    print(f"  {regime}: {perf['days']} days, total: {perf['total_return']:.2f}%")
    
    asyncio.run(test())