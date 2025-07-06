"""
Test technical indicator-based strategy without ML
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from backtest.backtest_engine import BacktestEngine


class TechnicalBacktestEngine(BacktestEngine):
    """Modified backtest engine for pure technical strategy"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trained = True  # Skip ML requirement
    
    def prepare_training_data(self, train_end_date: str) -> None:
        """Skip ML training for technical strategy"""
        logger.info("Using pure technical strategy - no ML training needed")
    
    def run_single_day(self, date: pd.Timestamp) -> dict:
        """Run backtest with technical signals only"""
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
        
        # Generate technical signals
        for symbol in SACRED_SYMBOLS:
            if symbol not in current_prices:
                continue
            
            try:
                # Load indicators for the symbol
                indicators_dir = DATA_DIR / 'indicators'
                
                # Load supertrend
                st_file = indicators_dir / f"{symbol}_1d_supertrend.csv"
                if st_file.exists():
                    st_df = pd.read_csv(st_file)
                    st_df['datetime'] = pd.to_datetime(st_df['datetime'])
                    st_df.set_index('datetime', inplace=True)
                    
                    if st_df.index.tz is not None:
                        st_df.index = st_df.index.tz_localize(None)
                    
                    if date in st_df.index:
                        row = st_df.loc[date]
                        
                        # Check for buy signal
                        if row.get('buy_signal', False) and symbol not in self.portfolio_manager.holdings:
                            # Only trade if we have significant cash
                            if self.portfolio_manager.cash_balance > 10000:
                                signal = 'STRONG_BUY'
                                confidence = 0.8
                                
                                trade_result = self.execute_backtest_trade(
                                    symbol, signal, confidence, date
                                )
                                
                                if trade_result:
                                    daily_trades += 1
                                    logger.info(f"{date.strftime('%Y-%m-%d')}: BUY {symbol} - Supertrend signal")
                        
                        # Check for sell signal (if we have position)
                        elif row.get('sell_signal', False) and symbol in self.portfolio_manager.holdings:
                            # Sell the position
                            shares = self.portfolio_manager.holdings[symbol]['shares']
                            price = current_prices[symbol]
                            
                            success = self.portfolio_manager.execute_trade(symbol, 'SELL', shares, price, date)
                            
                            if success:
                                commission = shares * price * self.commission_rate
                                self.portfolio_manager.cash_balance -= commission
                                
                                trade_record = {
                                    'date': date,
                                    'symbol': symbol,
                                    'signal': 'TECHNICAL_SELL',
                                    'action': 'SELL',
                                    'shares': shares,
                                    'price': price,
                                    'value': shares * price,
                                    'commission': commission,
                                    'confidence': 0.8,
                                    'total_cost': shares * price * (1 - self.commission_rate)
                                }
                                
                                self.trade_log.append(trade_record)
                                daily_trades += 1
                                logger.info(f"{date.strftime('%Y-%m-%d')}: SELL {symbol} - Supertrend signal")
                
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
        
        # Update portfolio performance
        performance = self.portfolio_manager.update_portfolio_performance(current_prices)
        
        # Check for stop losses
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
                    logger.info(f"{date.strftime('%Y-%m-%d')}: STOP LOSS {symbol} at {price:.2f}")
        
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


def main():
    """Run technical strategy backtest"""
    logger.info("Starting technical strategy backtest...")
    
    # Create backtest engine
    backtest = TechnicalBacktestEngine(
        start_date="2022-07-07",  # Start from when indicators are available
        end_date="2024-12-31",
        initial_capital=100000.0,
        commission_rate=0.001,
        slippage_rate=0.0005
    )
    
    # Run backtest
    results = backtest.run_backtest(train_split=0.0)  # No training needed
    
    # Save results
    backtest.save_backtest_results("technical_strategy")
    
    # Generate report
    report = backtest.generate_backtest_report()
    print(report)
    
    # Save report
    report_path = DATA_DIR.parent / 'backtest_results' / 'technical_strategy_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()