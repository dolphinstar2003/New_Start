"""
Realistic Backtest with 5 Core Indicators
Uses actual indicator signals for trading decisions
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import asyncio

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR, BACKTEST_CONFIG
from indicators.supertrend import calculate_supertrend
from indicators.macd_custom import calculate_macd_custom
from indicators.squeeze_momentum import calculate_squeeze_momentum
from indicators.wavetrend import calculate_wavetrend
from indicators.adx_di import calculate_adx_di


class IndicatorBacktest:
    """Backtest using real indicator signals"""
    
    def __init__(self):
        # Trading parameters
        self.initial_capital = 100000
        self.position_size_pct = 0.1  # 10% per position
        self.stop_loss_pct = 0.03     # 3% stop loss
        self.take_profit_pct = 0.08   # 8% take profit
        self.commission = 0.001       # 0.1% commission
        
    def load_data(self, symbol, days):
        """Load historical data"""
        try:
            data_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
            if not data_file.exists():
                return None
            
            df = pd.read_csv(data_file)
            # Set Date column
            if 'datetime' in df.columns:
                df['Date'] = pd.to_datetime(df['datetime'])
            elif 'Date' not in df.columns:
                logger.error(f"No date column found in {data_file}")
                return None
            df = df.sort_values('Date')
            
            # Get last N days
            end_date = df['Date'].max()
            start_date = end_date - timedelta(days=days)
            df = df[df['Date'] >= start_date].copy()
            
            logger.debug(f"Loaded {symbol}: {len(df)} rows after filtering to {days} days")
            return df
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate all 5 indicators"""
        try:
            # Supertrend
            st = calculate_supertrend(df)
            df['st_trend'] = st['trend'].values
            
            # MACD
            macd = calculate_macd_custom(df)
            df['macd_hist'] = macd['histogram'].values
            
            # Squeeze
            sq = calculate_squeeze_momentum(df)
            df['sq_mom'] = sq['momentum'].values
            df['sq_on'] = sq['squeeze_on'].values
            
            # WaveTrend
            wt = calculate_wavetrend(df)
            df['wt1'] = wt['wt1'].values
            df['wt2'] = wt['wt2'].values
            df['wt_cross_up'] = wt['cross_up'].values
            df['wt_cross_down'] = wt['cross_down'].values
            
            # ADX
            adx = calculate_adx_di(df)
            df['adx'] = adx['adx'].values
            df['plus_di'] = adx['plus_di'].values
            df['minus_di'] = adx['minus_di'].values
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            import traceback
            traceback.print_exc()
            return df
    
    def generate_signals(self, symbol_or_df):
        """Generate trading signals"""
        # Handle both symbol string and DataFrame inputs
        if isinstance(symbol_or_df, str):
            symbol = symbol_or_df.replace('.IS', '')
            df = self.load_data(symbol, 100)  # Load last 100 days
            if df is None or len(df) < 30:
                return pd.DataFrame()
            df = self.calculate_indicators(df)
        else:
            df = symbol_or_df.copy()  # Work with a copy to avoid warnings
        
        df['signal'] = 0
        
        for i in range(20, len(df)):
            buy_count = 0
            sell_count = 0
            
            # 1. Supertrend
            if df.iloc[i]['st_trend'] == 1:
                buy_count += 1
            else:
                sell_count += 1
            
            # 2. MACD histogram momentum
            if df.iloc[i]['macd_hist'] > df.iloc[i-1]['macd_hist']:
                buy_count += 1
            else:
                sell_count += 1
            
            # 3. Squeeze momentum
            if not df.iloc[i]['sq_on'] and df.iloc[i]['sq_mom'] > 0:
                buy_count += 1
            elif not df.iloc[i]['sq_on'] and df.iloc[i]['sq_mom'] < 0:
                sell_count += 1
            
            # 4. WaveTrend oversold/overbought
            if df.iloc[i]['wt_cross_up'] and df.iloc[i]['wt1'] < -60:
                buy_count += 1
            elif df.iloc[i]['wt_cross_down'] and df.iloc[i]['wt1'] > 60:
                sell_count += 1
            
            # 5. ADX trend strength
            if df.iloc[i]['adx'] > 25:
                if df.iloc[i]['plus_di'] > df.iloc[i]['minus_di']:
                    buy_count += 1
                else:
                    sell_count += 1
            
            # Signal if 2+ indicators agree (more aggressive)
            if buy_count >= 2:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                if i == len(df) - 1:  # Log last row
                    logger.debug(f"BUY signal: buy_count={buy_count}, sell_count={sell_count}")
            elif sell_count >= 2:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                if i == len(df) - 1:  # Log last row
                    logger.debug(f"SELL signal: buy_count={buy_count}, sell_count={sell_count}")
        
        return df
    
    async def run(self, days=30):
        """Run backtest"""
        all_trades = []
        
        # Test with first 5 symbols for speed
        for symbol in SACRED_SYMBOLS[:5]:
            # Remove .IS suffix for file loading
            clean_symbol = symbol.replace('.IS', '')
            df = self.load_data(clean_symbol, days + 100)  # Extra days for indicators
            if df is None or len(df) < 30:
                logger.warning(f"Skipping {symbol}: insufficient data")
                continue
            
            logger.info(f"Processing {symbol} with {len(df)} rows")
            df = self.calculate_indicators(df)
            df = self.generate_signals(df)
            
            # Log signal counts
            buy_signals = len(df[df['signal'] == 1])
            sell_signals = len(df[df['signal'] == -1])
            logger.info(f"{symbol}: {buy_signals} buy signals, {sell_signals} sell signals")
            
            # Simulate trading
            position = None
            
            for i in range(len(df)-days, len(df)):
                if i < 20:
                    continue
                
                price = df.iloc[i]['close']
                date = df.iloc[i]['Date']
                
                # Exit logic
                if position:
                    # Stop loss
                    if price <= position['stop_loss']:
                        ret = -self.stop_loss_pct
                        all_trades.append({'return': ret, 'win': False})
                        position = None
                    # Take profit
                    elif price >= position['take_profit']:
                        ret = self.take_profit_pct
                        all_trades.append({'return': ret, 'win': True})
                        position = None
                    # Exit signal
                    elif df.iloc[i]['signal'] == -1:
                        ret = (price - position['entry']) / position['entry']
                        all_trades.append({'return': ret, 'win': ret > 0})
                        position = None
                
                # Entry logic
                elif df.iloc[i]['signal'] == 1 and not position:
                    position = {
                        'entry': price,
                        'stop_loss': price * (1 - self.stop_loss_pct),
                        'take_profit': price * (1 + self.take_profit_pct)
                    }
        
        # Calculate results
        if not all_trades:
            return {
                'backtest_engine': 'indicator_based',
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
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate cumulative returns
        capital = self.initial_capital
        equity_curve = [capital]
        
        for trade in all_trades:
            position_size = capital * self.position_size_pct
            trade_return = position_size * trade['return']
            commission = position_size * self.commission * 2
            capital += trade_return - commission
            equity_curve.append(capital)
        
        # Metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        wins = sum(1 for t in all_trades if t['win'])
        losses = len(all_trades) - wins
        win_rate = wins / len(all_trades) * 100 if all_trades else 0
        
        # Sharpe
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0
        
        # Max drawdown
        equity = pd.Series(equity_curve)
        running_max = equity.expanding().max()
        dd = (equity - running_max) / running_max
        max_dd = abs(dd.min()) * 100
        
        # Best/worst
        trade_returns = [t['return'] * 100 for t in all_trades]
        best = max(trade_returns) if trade_returns else 0
        worst = min(trade_returns) if trade_returns else 0
        
        return {
            'backtest_engine': 'indicator_based',
            'initial_capital': self.initial_capital,
            'final_value': capital,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd,
            'total_trades': len(all_trades),
            'profitable_trades': wins,
            'losing_trades': losses,
            'best_trade': best,
            'worst_trade': worst,
            'timestamp': datetime.now().isoformat()
        }


async def run_realistic_backtest(days=30):
    """Run indicator-based backtest"""
    backtest = IndicatorBacktest()
    return await backtest.run(days)