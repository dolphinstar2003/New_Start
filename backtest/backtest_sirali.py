"""
Sıralı/Hiyerarşik Backtest Module
Multi-timeframe approach with sequential indicator checking
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

from config.settings import SACRED_SYMBOLS, DATA_DIR, BACKTEST_CONFIG


class HierarchicalBacktest:
    """Hierarchical multi-timeframe backtest"""
    
    def __init__(self, timeframe_mode: str = 'hybrid'):
        """
        Initialize hierarchical backtest
        
        Args:
            timeframe_mode: '4h', '1d', or 'hybrid' (1d + 4h)
        """
        self.timeframe_mode = timeframe_mode
        self.indicators_dir = DATA_DIR / 'indicators'
        
        # Trading parameters
        self.initial_capital = 100000
        self.position_size_pct = 0.1
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.08
        self.commission = 0.001
        
        # Risk management strategies
        self.risk_strategy = 'sl_plus_conditional_trailing'
        self.trailing_activation_pct = 0.05  # Activate trailing at 5% profit
        self.trailing_distance_pct = 0.03    # 3% trailing distance
        
        logger.info(f"HierarchicalBacktest initialized with {timeframe_mode} mode")
    
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
    
    def load_raw_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load raw OHLCV data"""
        clean_symbol = symbol.replace('.IS', '')
        filepath = DATA_DIR / 'raw' / timeframe / f"{clean_symbol}_{timeframe}_raw.csv"
        
        if not filepath.exists():
            return pd.DataFrame()
        
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    
    def check_daily_trend(self, symbol: str, date: datetime) -> int:
        """Check daily trend for given date"""
        # Load daily indicators
        st_df = self.load_indicator_data(symbol, '1d', 'supertrend')
        sq_df = self.load_indicator_data(symbol, '1d', 'squeeze_momentum')
        
        if st_df.empty or sq_df.empty:
            return 0
        
        # Find the daily data for given date
        daily_date = date.date()
        
        try:
            # Get the most recent daily data before or on the date
            st_trend = st_df[st_df.index.date <= daily_date]['trend'].iloc[-1]
            sq_momentum = sq_df[sq_df.index.date <= daily_date]['momentum'].iloc[-1]
            sq_on = sq_df[sq_df.index.date <= daily_date]['squeeze_on'].iloc[-1]
            
            # Both indicators bullish
            if st_trend == 1 and sq_momentum > 0 and not sq_on:
                return 1
            # Both indicators bearish
            elif st_trend == -1 and sq_momentum < 0 and not sq_on:
                return -1
            else:
                return 0
                
        except:
            return 0
    
    def generate_signals_sequential(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Generate signals by checking indicators sequentially"""
        # Load raw data
        ohlcv = self.load_raw_data(symbol, timeframe)
        if ohlcv.empty:
            return pd.DataFrame()
        
        # Load all indicators
        indicators = {}
        for ind in ['supertrend', 'squeeze_momentum', 'wavetrend', 'adx_di', 'macd_custom']:
            df = self.load_indicator_data(symbol, timeframe, ind)
            if not df.empty:
                indicators[ind] = df
        
        # Create signals dataframe
        signals = pd.DataFrame(index=ohlcv.index)
        signals['close'] = ohlcv['close']
        signals['signal'] = 0
        signals['signal_strength'] = 0
        
        # Generate signals sequentially
        for i in range(20, len(signals)):
            date = signals.index[i]
            
            # Skip if in hybrid mode and daily trend is not favorable
            if self.timeframe_mode == 'hybrid' and timeframe == '4h':
                daily_trend = self.check_daily_trend(symbol, date)
                if daily_trend == 0:
                    continue
            
            buy_score = 0
            sell_score = 0
            
            # 1. First check Supertrend (primary trend)
            if 'supertrend' in indicators:
                try:
                    st_trend = indicators['supertrend'].loc[date, 'trend']
                    if st_trend == 1:
                        buy_score += 2  # Double weight for primary indicator
                    else:
                        sell_score += 2
                except:
                    continue
            
            # 2. Then check Squeeze Momentum (volatility breakout)
            if 'squeeze_momentum' in indicators:
                try:
                    sq_mom = indicators['squeeze_momentum'].loc[date, 'momentum']
                    sq_on = indicators['squeeze_momentum'].loc[date, 'squeeze_on']
                    
                    if not sq_on and sq_mom > 0:
                        buy_score += 2  # Double weight for core indicator
                    elif not sq_on and sq_mom < 0:
                        sell_score += 2
                except:
                    pass
            
            # 3. Check WaveTrend for oversold/overbought
            if 'wavetrend' in indicators:
                try:
                    wt1 = indicators['wavetrend'].loc[date, 'wt1']
                    cross_up = indicators['wavetrend'].loc[date, 'cross_up']
                    cross_down = indicators['wavetrend'].loc[date, 'cross_down']
                    
                    if cross_up and wt1 < -60:
                        buy_score += 1
                    elif cross_down and wt1 > 60:
                        sell_score += 1
                except:
                    pass
            
            # 4. Check ADX for trend strength
            if 'adx_di' in indicators:
                try:
                    adx = indicators['adx_di'].loc[date, 'adx']
                    plus_di = indicators['adx_di'].loc[date, 'plus_di']
                    minus_di = indicators['adx_di'].loc[date, 'minus_di']
                    
                    if adx > 25:
                        if plus_di > minus_di:
                            buy_score += 1
                        else:
                            sell_score += 1
                except:
                    pass
            
            # 5. Check MACD for momentum
            if 'macd_custom' in indicators:
                try:
                    hist = indicators['macd_custom'].loc[date, 'histogram']
                    if i > 0:
                        prev_hist = indicators['macd_custom'].iloc[i-1]['histogram']
                        if hist > prev_hist and hist > 0:
                            buy_score += 1
                        elif hist < prev_hist and hist < 0:
                            sell_score += 1
                except:
                    pass
            
            # Generate signal if score is high enough (more aggressive)
            if buy_score >= 3:  # Lowered from 5 for more signals
                signals.loc[date, 'signal'] = 1
                signals.loc[date, 'signal_strength'] = buy_score
                if i == len(signals) - 1:  # Log last signal
                    logger.debug(f"Hier BUY: buy_score={buy_score}, sell_score={sell_score}")
            elif sell_score >= 3:  # Lowered from 5
                signals.loc[date, 'signal'] = -1
                signals.loc[date, 'signal_strength'] = sell_score
                if i == len(signals) - 1:  # Log last signal
                    logger.debug(f"Hier SELL: buy_score={buy_score}, sell_score={sell_score}")
        
        return signals
    
    def apply_risk_management(self, position: Dict, current_price: float, 
                            high_since_entry: float) -> Tuple[bool, str]:
        """Apply risk management rules"""
        if self.risk_strategy == 'only_signals':
            return False, ""
        
        # Always check stop loss
        if current_price <= position['stop_loss']:
            return True, "stop_loss"
        
        # Check take profit
        if current_price >= position['take_profit']:
            return True, "take_profit"
        
        # Conditional trailing stop
        if self.risk_strategy in ['sl_plus_trailing', 'sl_plus_conditional_trailing']:
            profit_pct = (current_price - position['entry_price']) / position['entry_price']
            
            # Only activate trailing after reaching activation threshold
            if self.risk_strategy == 'sl_plus_conditional_trailing':
                if profit_pct < self.trailing_activation_pct:
                    return False, ""
            
            # Update trailing stop
            trailing_stop = high_since_entry * (1 - self.trailing_distance_pct)
            if current_price <= trailing_stop:
                return True, "trailing_stop"
        
        return False, ""
    
    async def backtest_symbol(self, symbol: str, days: int) -> List[Dict]:
        """Run backtest for single symbol"""
        # Determine timeframe
        if self.timeframe_mode == 'hybrid':
            signals = self.generate_signals_sequential(symbol, '4h')
        else:
            signals = self.generate_signals_sequential(symbol, self.timeframe_mode)
        
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
            
            # Check exit conditions
            if position:
                # Update high water mark
                high_since_entry = max(high_since_entry, current_price)
                
                # Check risk management
                should_exit, exit_reason = self.apply_risk_management(
                    position, current_price, high_since_entry
                )
                
                if should_exit or row['signal'] == -1:
                    # Calculate return
                    return_pct = (current_price - position['entry_price']) / position['entry_price']
                    
                    trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'return_pct': return_pct,
                        'exit_reason': exit_reason or 'signal',
                        'signal_strength': position['signal_strength']
                    })
                    
                    position = None
                    high_since_entry = 0
            
            # Check entry conditions
            elif row['signal'] == 1 and position is None:
                position = {
                    'entry_date': date,
                    'entry_price': current_price,
                    'stop_loss': current_price * (1 - self.stop_loss_pct),
                    'take_profit': current_price * (1 + self.take_profit_pct),
                    'signal_strength': row['signal_strength']
                }
                high_since_entry = current_price
        
        return trades
    
    async def run(self, days: int = 30) -> Dict:
        """Run hierarchical backtest"""
        all_trades = []
        
        # Test first 10 symbols
        for symbol in SACRED_SYMBOLS[:10]:
            logger.info(f"Backtesting {symbol} with {self.timeframe_mode} mode...")
            trades = await self.backtest_symbol(symbol, days)
            all_trades.extend(trades)
        
        # Calculate metrics
        if not all_trades:
            return {
                'backtest_engine': f'hierarchical_{self.timeframe_mode}',
                'risk_strategy': self.risk_strategy,
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
                'avg_signal_strength': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Convert to DataFrame for easier calculation
        trades_df = pd.DataFrame(all_trades)
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        trades_df = trades_df.sort_values('exit_date')
        
        # Calculate portfolio performance
        capital = self.initial_capital
        equity_curve = [capital]
        
        for _, trade in trades_df.iterrows():
            position_size = capital * self.position_size_pct
            trade_return = position_size * trade['return_pct']
            commission = position_size * self.commission * 2
            capital += trade_return - commission
            equity_curve.append(capital)
        
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
        
        # Average signal strength
        avg_signal_strength = trades_df['signal_strength'].mean() if len(trades_df) > 0 else 0
        
        # Exit reason breakdown
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
        
        return {
            'backtest_engine': f'hierarchical_{self.timeframe_mode}',
            'risk_strategy': self.risk_strategy,
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
            'avg_signal_strength': avg_signal_strength,
            'exit_reasons': exit_reasons,
            'timestamp': datetime.now().isoformat()
        }


# Convenience functions for different modes
async def run_backtest_4h(days: int = 30, risk_strategy: str = 'sl_plus_conditional_trailing') -> Dict:
    """Run 4h timeframe backtest"""
    backtest = HierarchicalBacktest(timeframe_mode='4h')
    backtest.risk_strategy = risk_strategy
    return await backtest.run(days)


async def run_backtest_1d(days: int = 30, risk_strategy: str = 'sl_plus_conditional_trailing') -> Dict:
    """Run 1d timeframe backtest"""
    backtest = HierarchicalBacktest(timeframe_mode='1d')
    backtest.risk_strategy = risk_strategy
    return await backtest.run(days)


async def run_backtest_hybrid(days: int = 30, risk_strategy: str = 'sl_plus_conditional_trailing') -> Dict:
    """Run hybrid (1d + 4h) timeframe backtest"""
    backtest = HierarchicalBacktest(timeframe_mode='hybrid')
    backtest.risk_strategy = risk_strategy
    return await backtest.run(days)