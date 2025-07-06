"""
Simplified MTF Backtest
Using technical indicators only, no ML
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from backtest.backtest_engine import BacktestEngine
from strategies.signal_generator import TradingSignalGenerator

class SimpleMTFBacktest(BacktestEngine):
    """Simple MTF backtest using technical signals"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trained = True  # Skip ML
        
    def prepare_training_data(self, train_end_date: str) -> None:
        """No training needed for technical strategy"""
        logger.info("Using technical MTF strategy - no ML training")
        
    def get_mtf_signal(self, symbol: str, date: pd.Timestamp) -> tuple:
        """Get multi-timeframe signal for a symbol"""
        signals = {}
        weights = {'1h': 0.2, '4h': 0.3, '1d': 0.5}
        
        for tf in ['1h', '4h', '1d']:
            # Try to load indicator file
            indicator_file = DATA_DIR / 'indicators' / tf / f"{symbol}_{tf}_supertrend.csv"
            
            # Fallback to 1d only if other timeframes not available
            if not indicator_file.exists() and tf == '1d':
                indicator_file = DATA_DIR / 'indicators' / f"{symbol}_{tf}_supertrend.csv"
            
            if indicator_file.exists():
                try:
                    df = pd.read_csv(indicator_file)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # For non-daily timeframes, get the latest signal before date
                    if tf != '1d':
                        mask = df.index <= date
                        if mask.any():
                            latest_idx = df.index[mask][-1]
                            row = df.loc[latest_idx]
                        else:
                            continue
                    else:
                        if date in df.index:
                            row = df.loc[date]
                        else:
                            continue
                    
                    # Get signal
                    if row.get('buy_signal', False):
                        signals[tf] = 2  # Strong buy
                    elif row.get('trend', 0) == 1:
                        signals[tf] = 1  # Buy
                    elif row.get('sell_signal', False):
                        signals[tf] = -2  # Strong sell
                    elif row.get('trend', 0) == -1:
                        signals[tf] = -1  # Sell
                    else:
                        signals[tf] = 0  # Neutral
                        
                except Exception as e:
                    logger.debug(f"Error reading {tf} for {symbol}: {e}")
                    
        # Calculate weighted signal
        if signals:
            weighted_sum = sum(signals.get(tf, 0) * weights.get(tf, 0) for tf in weights)
            weight_sum = sum(weights.get(tf, 0) for tf in signals)
            
            if weight_sum > 0:
                weighted_signal = weighted_sum / weight_sum
            else:
                weighted_signal = 0
                
            # Determine final signal
            if weighted_signal >= 1.5:
                return 'STRONG_BUY', 0.9
            elif weighted_signal >= 0.5:
                return 'WEAK_BUY', 0.7
            elif weighted_signal <= -1.5:
                return 'STRONG_SELL', 0.9
            elif weighted_signal <= -0.5:
                return 'WEAK_SELL', 0.7
            else:
                return 'HOLD', 0.5
        else:
            return 'HOLD', 0.5
            
    def run_single_day(self, date: pd.Timestamp) -> dict:
        """Run backtest for a single day with MTF signals"""
        current_prices = self.get_current_prices(date)
        
        if not current_prices:
            return {'date': date, 'trades': 0, 'portfolio_value': self.portfolio_manager.current_capital}
        
        # Update portfolio valuation
        portfolio_value = self.portfolio_manager.calculate_portfolio_value(current_prices)
        
        # Update cash balance
        self.portfolio_manager.cash_balance = portfolio_value - sum(
            holding['shares'] * current_prices.get(symbol, 0) 
            for symbol, holding in self.portfolio_manager.holdings.items()
        )
        
        daily_trades = 0
        
        # Check signals for each symbol
        for symbol in SACRED_SYMBOLS:
            if symbol not in current_prices:
                continue
                
            try:
                signal, confidence = self.get_mtf_signal(symbol, date)
                
                # Execute buy signals
                if signal in ['STRONG_BUY', 'WEAK_BUY'] and symbol not in self.portfolio_manager.holdings:
                    if self.portfolio_manager.cash_balance > 10000:
                        trade_result = self.execute_backtest_trade(
                            symbol, signal, confidence, date
                        )
                        if trade_result:
                            daily_trades += 1
                            logger.debug(f"{date.strftime('%Y-%m-%d')}: BUY {symbol} - MTF signal")
                            
                # Execute sell signals
                elif signal in ['STRONG_SELL', 'WEAK_SELL'] and symbol in self.portfolio_manager.holdings:
                    shares = self.portfolio_manager.holdings[symbol]['shares']
                    price = current_prices[symbol]
                    
                    success = self.portfolio_manager.execute_trade(symbol, 'SELL', shares, price, date)
                    if success:
                        commission = shares * price * self.commission_rate
                        self.portfolio_manager.cash_balance -= commission
                        
                        trade_record = {
                            'date': date,
                            'symbol': symbol,
                            'signal': signal,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price,
                            'value': shares * price,
                            'commission': commission,
                            'confidence': confidence,
                            'total_cost': shares * price * (1 - self.commission_rate)
                        }
                        self.trade_log.append(trade_record)
                        daily_trades += 1
                        logger.debug(f"{date.strftime('%Y-%m-%d')}: SELL {symbol} - MTF signal")
                        
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                
        # Update portfolio performance
        performance = self.portfolio_manager.update_portfolio_performance(current_prices)
        
        # Check stop losses
        stop_losses = self.risk_manager.check_stop_losses(current_prices)
        for stop_loss in stop_losses:
            symbol = stop_loss['symbol']
            price = stop_loss['current_price']
            
            if symbol in self.portfolio_manager.holdings:
                shares = self.portfolio_manager.holdings[symbol]['shares']
                success = self.portfolio_manager.execute_trade(symbol, 'SELL', shares, price, date)
                
                if success:
                    trade_record = {
                        'date': date,
                        'symbol': symbol,
                        'signal': 'STOP_LOSS',
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'commission': shares * price * self.commission_rate,
                        'confidence': 1.0,
                        'total_cost': shares * price * (1 - self.commission_rate)
                    }
                    self.trade_log.append(trade_record)
                    daily_trades += 1
                    
        # Reset daily risk metrics
        self.risk_manager.reset_daily_metrics()
        
        daily_result = {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash_balance': self.portfolio_manager.cash_balance,
            'trades': daily_trades,
            'open_positions': len(self.portfolio_manager.holdings),
            'daily_return': performance.get('daily_return', 0.0) if performance else 0.0
        }
        
        self.daily_results.append(daily_result)
        return daily_result


# Run MTF backtest
logger.info("="*80)
logger.info("RUNNING SIMPLE MTF BACKTEST")
logger.info("="*80)

backtest = SimpleMTFBacktest(
    start_date="2023-01-01",
    end_date="2024-12-31",
    initial_capital=100000.0,
    commission_rate=0.001,
    slippage_rate=0.0005
)

results = backtest.run_backtest(train_split=0.0)

# Save results
backtest.save_backtest_results("simple_mtf_backtest")

# Generate report
report = backtest.generate_backtest_report()
print("\n" + "="*80)
print("SIMPLE MTF BACKTEST RESULTS")
print("="*80)
print(report)

# Save report
report_path = DATA_DIR.parent / 'backtest_results' / 'simple_mtf_backtest_report.txt'
with open(report_path, 'w') as f:
    f.write(report)

logger.info(f"\n✅ Results saved to: {report_path}")

# Show trade summary
if backtest.trade_log:
    trades_df = pd.DataFrame(backtest.trade_log)
    print(f"\nTotal trades: {len(trades_df)}")
    print(f"Buy trades: {len(trades_df[trades_df['action'] == 'BUY'])}")
    print(f"Sell trades: {len(trades_df[trades_df['action'] == 'SELL'])}")
    
    # Symbol distribution
    print("\nTrades by symbol:")
    print(trades_df['symbol'].value_counts().head(10))
else:
    print("\nNo trades executed")