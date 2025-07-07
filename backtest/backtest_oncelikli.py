"""
Öncelikli (Priority-based) Backtest Module
Core indicators (Supertrend + Squeeze) must agree, others confirm
Includes VIX-based risk adjustment and Kelly Criterion position sizing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import asyncio
from typing import Dict, List, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.vixfix import get_vix_risk_adjustment, calculate_kelly_criterion


class PriorityBacktest:
    """Priority-based backtest with core indicator requirements"""
    
    def __init__(self):
        """Initialize priority-based backtest"""
        self.indicators_dir = DATA_DIR / 'indicators'
        
        # Base trading parameters
        self.initial_capital = 100000
        self.base_position_size = 0.05  # 5% for normal signals
        self.strong_position_size = 0.10  # 10% for strong signals
        self.base_stop_loss = 0.03
        self.base_take_profit = 0.08
        self.commission = 0.001
        
        # Risk management
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        self.risk_reward_min = 2.0  # Minimum 2:1 risk/reward
        self.max_positions = 5
        
        # Kelly Criterion parameters
        self.use_kelly = True
        self.kelly_fraction = 0.25  # Use 25% of Kelly
        
        # Track performance for Kelly
        self.trade_history = []
        
        logger.info("PriorityBacktest initialized with core indicator requirements")
    
    def load_indicator_data(self, symbol: str, timeframe: str, indicator: str) -> pd.DataFrame:
        """Load pre-calculated indicator data from CSV"""
        clean_symbol = symbol.replace('.IS', '')
        filepath = self.indicators_dir / timeframe / f"{clean_symbol}_{timeframe}_{indicator}.csv"
        
        if not filepath.exists():
            logger.warning(f"Indicator file not found: {filepath}")
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    
    def load_raw_data(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Load raw OHLCV data"""
        clean_symbol = symbol.replace('.IS', '')
        filepath = DATA_DIR / 'raw' / timeframe / f"{clean_symbol}_{timeframe}_raw.csv"
        
        if not filepath.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    
    def calculate_position_size(self, signal_type: str, vix_value: float) -> float:
        """Calculate position size based on signal strength and VIX"""
        if self.use_kelly and len(self.trade_history) > 10:
            # Calculate Kelly criterion
            wins = [t for t in self.trade_history if t['profit'] > 0]
            losses = [t for t in self.trade_history if t['profit'] <= 0]
            
            if wins and losses:
                win_rate = len(wins) / len(self.trade_history)
                avg_win = np.mean([t['profit'] for t in wins])
                avg_loss = abs(np.mean([t['profit'] for t in losses]))
                
                # VIX adjustment (0-1 scale)
                vix_factor = 1 - min(vix_value / 50, 0.8)  # Max 80% reduction
                
                kelly_size = calculate_kelly_criterion(win_rate, avg_win, avg_loss, vix_factor)
                
                # Use Kelly size but respect signal type limits
                if signal_type == 'strong':
                    return min(kelly_size, self.strong_position_size)
                else:
                    return min(kelly_size, self.base_position_size)
        
        # Default sizing with VIX adjustment
        vix_adjustments = get_vix_risk_adjustment(vix_value)
        
        if signal_type == 'strong':
            return self.strong_position_size * vix_adjustments['position_size_multiplier']
        else:
            return self.base_position_size * vix_adjustments['position_size_multiplier']
    
    def generate_priority_signals(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Generate signals based on priority system"""
        # Load raw data
        ohlcv = self.load_raw_data(symbol, timeframe)
        if ohlcv.empty:
            return pd.DataFrame()
        
        # Load indicators
        indicators = {}
        for ind in ['supertrend', 'squeeze_momentum', 'wavetrend', 'adx_di', 'vixfix']:
            df = self.load_indicator_data(symbol, timeframe, ind)
            if not df.empty:
                indicators[ind] = df
        
        # Create signals dataframe
        signals = pd.DataFrame(index=ohlcv.index)
        signals['close'] = ohlcv['close']
        signals['signal'] = 0
        signals['signal_type'] = 'none'
        signals['vix'] = 20  # Default VIX
        
        # Add VIX values if available
        if 'vixfix' in indicators:
            signals['vix'] = indicators['vixfix']['vixfix'].fillna(20)
        
        # Generate signals
        for i in range(20, len(signals)):
            date = signals.index[i]
            
            # Core indicators check (MUST BOTH AGREE)
            core_buy = False
            core_sell = False
            
            # 1. Supertrend (Core)
            if 'supertrend' in indicators:
                try:
                    st_trend = indicators['supertrend'].loc[date, 'trend']
                    st_buy = (st_trend == 1)
                    st_sell = (st_trend == -1)
                except:
                    continue
            else:
                continue
            
            # 2. Squeeze Momentum (Core)
            if 'squeeze_momentum' in indicators:
                try:
                    sq_mom = indicators['squeeze_momentum'].loc[date, 'momentum']
                    sq_on = indicators['squeeze_momentum'].loc[date, 'squeeze_on']
                    sq_buy = (not sq_on and sq_mom > 0)
                    sq_sell = (not sq_on and sq_mom < 0)
                except:
                    continue
            else:
                continue
            
            # Both core indicators must agree
            core_buy = st_buy and sq_buy
            core_sell = st_sell and sq_sell
            
            # Skip if core indicators don't agree
            if not core_buy and not core_sell:
                continue
            
            # Check confirmation indicators for signal strength
            confirmations = 0
            
            # 3. WaveTrend confirmation
            if 'wavetrend' in indicators:
                try:
                    wt1 = indicators['wavetrend'].loc[date, 'wt1']
                    cross_up = indicators['wavetrend'].loc[date, 'cross_up']
                    cross_down = indicators['wavetrend'].loc[date, 'cross_down']
                    
                    if core_buy and cross_up and wt1 < -60:
                        confirmations += 1
                    elif core_sell and cross_down and wt1 > 60:
                        confirmations += 1
                except:
                    pass
            
            # 4. VIX confirmation (high VIX = potential reversal)
            if 'vixfix' in indicators:
                try:
                    vix_high = indicators['vixfix'].loc[date, 'high_volatility']
                    market_bottom = indicators['vixfix'].loc[date, 'market_bottom']
                    
                    if core_buy and (vix_high or market_bottom):
                        confirmations += 2  # Double weight for VIX
                    elif core_sell and vix_high:
                        confirmations += 1
                except:
                    pass
            
            # 5. ADX confirmation (strong trend)
            if 'adx_di' in indicators:
                try:
                    adx = indicators['adx_di'].loc[date, 'adx']
                    plus_di = indicators['adx_di'].loc[date, 'plus_di']
                    minus_di = indicators['adx_di'].loc[date, 'minus_di']
                    
                    if adx > 25:
                        if core_buy and plus_di > minus_di:
                            confirmations += 1
                        elif core_sell and minus_di > plus_di:
                            confirmations += 1
                except:
                    pass
            
            # Generate signal based on confirmations
            if core_buy:
                signals.loc[date, 'signal'] = 1
                if confirmations >= 2:
                    signals.loc[date, 'signal_type'] = 'strong_buy'
                else:
                    signals.loc[date, 'signal_type'] = 'buy'
            
            elif core_sell:
                signals.loc[date, 'signal'] = -1
                if confirmations >= 2:
                    signals.loc[date, 'signal_type'] = 'strong_sell'
                else:
                    signals.loc[date, 'signal_type'] = 'sell'
        
        return signals
    
    def apply_dynamic_risk_management(self, position: Dict, current_price: float, 
                                    vix_value: float, high_since_entry: float) -> Tuple[bool, str]:
        """Apply VIX-adjusted risk management"""
        # Get VIX adjustments
        vix_adj = get_vix_risk_adjustment(vix_value)
        
        # Adjusted stop loss
        adjusted_stop = position['entry_price'] * (1 - self.base_stop_loss * vix_adj['stop_loss_multiplier'])
        if current_price <= adjusted_stop:
            return True, "stop_loss"
        
        # Adjusted take profit
        adjusted_target = position['entry_price'] * (1 + self.base_take_profit * vix_adj['take_profit_multiplier'])
        if current_price >= adjusted_target:
            return True, "take_profit"
        
        # Conditional trailing stop
        profit_pct = (current_price - position['entry_price']) / position['entry_price']
        
        # Only activate trailing after 5% profit and indicators still positive
        if profit_pct >= 0.05 and position.get('signal_type', '').startswith('strong'):
            trailing_distance = vix_adj['trailing_stop_distance']
            trailing_stop = high_since_entry * (1 - trailing_distance)
            
            if current_price <= trailing_stop:
                return True, "trailing_stop"
        
        # Time-based exit (optional)
        # Use current date from backtest data, not real-time
        if hasattr(self, 'current_date'):
            days_held = (self.current_date - position['entry_date']).days
            if days_held > 10 and profit_pct < -0.02:  # Exit losing positions after 10 days
                return True, "time_stop"
        
        return False, ""
    
    async def backtest_symbol(self, symbol: str, days: int) -> List[Dict]:
        """Run backtest for single symbol"""
        signals = self.generate_priority_signals(symbol)
        
        if signals.empty:
            return []
        
        # Filter to requested days
        end_date = signals.index[-1]
        start_date = end_date - timedelta(days=days)
        signals = signals[signals.index >= start_date]
        
        trades = []
        position = None
        high_since_entry = 0
        
        for date, row in signals.iterrows():
            current_price = row['close']
            current_vix = row['vix']
            
            # Set current date for time-based exits
            self.current_date = date
            
            # Check exit conditions
            if position:
                # Update high water mark
                high_since_entry = max(high_since_entry, current_price)
                
                # Check risk management
                should_exit, exit_reason = self.apply_dynamic_risk_management(
                    position, current_price, current_vix, high_since_entry
                )
                
                if should_exit or (row['signal'] == -1 and not position.get('signal_type', '').startswith('strong')):
                    # Calculate return
                    return_pct = (current_price - position['entry_price']) / position['entry_price']
                    profit = return_pct * position['position_size'] * self.initial_capital
                    
                    trade_result = {
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'profit': profit,
                        'exit_reason': exit_reason,
                        'signal_type': position['signal_type'],
                        'position_size': position['position_size']
                    }
                    
                    trades.append(trade_result)
                    self.trade_history.append({'profit': profit})  # For Kelly
                    
                    position = None
                    high_since_entry = 0
            
            # Check entry conditions
            elif row['signal'] == 1 and position is None:
                signal_type = row['signal_type']
                position_size = self.calculate_position_size(signal_type, current_vix)
                
                # Check risk/reward ratio
                vix_adj = get_vix_risk_adjustment(current_vix)
                stop_distance = self.base_stop_loss * vix_adj['stop_loss_multiplier']
                target_distance = self.base_take_profit * vix_adj['take_profit_multiplier']
                
                if target_distance / stop_distance >= self.risk_reward_min:
                    position = {
                        'entry_date': date,
                        'entry_price': current_price,
                        'signal_type': signal_type,
                        'position_size': position_size,
                        'vix_at_entry': current_vix
                    }
                    high_since_entry = current_price
        
        return trades
    
    async def run(self, days: int = 30) -> Dict:
        """Run priority-based backtest"""
        all_trades = []
        self.trade_history = []  # Reset for Kelly calculation
        
        # Test first 10 symbols
        for symbol in SACRED_SYMBOLS[:10]:
            logger.info(f"Backtesting {symbol} with priority system...")
            trades = await self.backtest_symbol(symbol, days)
            all_trades.extend(trades)
        
        # Calculate metrics
        if not all_trades:
            return {
                'backtest_engine': 'priority_based',
                'initial_capital': self.initial_capital,
                'final_value': self.initial_capital,
                'total_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'profitable_trades': 0,
                'losing_trades': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_position_size': 0,
                'signal_breakdown': {},
                'timestamp': datetime.now().isoformat()
            }
        
        # Convert to DataFrame
        trades_df = pd.DataFrame(all_trades)
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df = trades_df.sort_values('exit_date')
        
        # Calculate portfolio performance
        capital = self.initial_capital
        equity_curve = [capital]
        max_capital = capital
        
        for _, trade in trades_df.iterrows():
            position_value = capital * trade['position_size']
            trade_return = position_value * trade['return_pct']
            commission = position_value * self.commission * 2
            capital += trade_return - commission
            equity_curve.append(capital)
            
            # Track max for drawdown
            max_capital = max(max_capital, capital)
            current_dd = (capital - max_capital) / max_capital
            
            # Stop if max drawdown exceeded
            if abs(current_dd) > self.max_drawdown_limit:
                logger.warning(f"Max drawdown limit exceeded: {current_dd:.2%}")
                break
        
        # Calculate metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        winning_trades = len(trades_df[trades_df['return_pct'] > 0])
        losing_trades = len(trades_df[trades_df['return_pct'] <= 0])
        win_rate = winning_trades / len(trades_df) * 100 if len(trades_df) > 0 else 0
        
        # Sharpe ratio
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
        
        # Max drawdown
        equity = pd.Series(equity_curve)
        running_max = equity.expanding().max()
        dd = (equity - running_max) / running_max
        max_dd = abs(dd.min()) * 100
        
        # Best/worst trades
        best_trade = trades_df['return_pct'].max() * 100 if len(trades_df) > 0 else 0
        worst_trade = trades_df['return_pct'].min() * 100 if len(trades_df) > 0 else 0
        
        # Average position size
        avg_position_size = trades_df['position_size'].mean() * 100 if len(trades_df) > 0 else 0
        
        # Signal type breakdown
        signal_breakdown = trades_df['signal_type'].value_counts().to_dict()
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        # Profit factor
        gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'backtest_engine': 'priority_based',
            'initial_capital': self.initial_capital,
            'final_value': capital,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'total_trades': len(trades_df),
            'profitable_trades': winning_trades,
            'losing_trades': losing_trades,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'profit_factor': profit_factor,
            'avg_position_size': avg_position_size,
            'signal_breakdown': signal_breakdown,
            'exit_reasons': exit_reasons,
            'timestamp': datetime.now().isoformat()
        }


# Convenience function
async def run_priority_backtest(days: int = 30) -> Dict:
    """Run priority-based backtest with core indicators"""
    backtest = PriorityBacktest()
    return await backtest.run(days)