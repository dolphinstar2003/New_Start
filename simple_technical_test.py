"""Simple technical strategy test"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from backtest.backtest_engine import BacktestEngine

# Override run_single_day to add debug
class DebugTechnicalBacktest(BacktestEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_trained = True
        self.signal_count = 0
        
    def prepare_training_data(self, train_end_date: str) -> None:
        logger.info("Using technical strategy - no ML training")
        
    def run_single_day(self, date: pd.Timestamp) -> dict:
        current_prices = self.get_current_prices(date)
        
        if not current_prices:
            return {'date': date, 'trades': 0, 'portfolio_value': self.portfolio_manager.current_capital}
        
        portfolio_value = self.portfolio_manager.calculate_portfolio_value(current_prices)
        daily_trades = 0
        
        # Only check GARAN for debugging
        symbol = 'GARAN'
        if symbol in current_prices:
            try:
                st_file = DATA_DIR / 'indicators' / f"{symbol}_1d_supertrend.csv"
                if st_file.exists():
                    st_df = pd.read_csv(st_file)
                    st_df['datetime'] = pd.to_datetime(st_df['datetime'])
                    st_df.set_index('datetime', inplace=True)
                    
                    if st_df.index.tz is not None:
                        st_df.index = st_df.index.tz_localize(None)
                    
                    if date in st_df.index:
                        row = st_df.loc[date]
                        
                        if row.get('buy_signal', False):
                            self.signal_count += 1
                            logger.info(f"SIGNAL {self.signal_count}: {date} - BUY signal for {symbol}")
                            logger.info(f"  Holdings: {list(self.portfolio_manager.holdings.keys())}")
                            logger.info(f"  Cash: ${self.portfolio_manager.cash_balance:,.2f}")
                            
                            if symbol not in self.portfolio_manager.holdings and self.portfolio_manager.cash_balance > 10000:
                                trade_result = self.execute_backtest_trade(
                                    symbol, 'STRONG_BUY', 0.8, date
                                )
                                if trade_result:
                                    daily_trades += 1
                                    logger.info(f"  ✅ EXECUTED BUY")
                                else:
                                    logger.info(f"  ❌ TRADE FAILED")
                        
                        elif row.get('sell_signal', False) and symbol in self.portfolio_manager.holdings:
                            logger.info(f"SELL signal for {symbol} on {date}")
                            shares = self.portfolio_manager.holdings[symbol]['shares']
                            price = current_prices[symbol]
                            
                            success = self.portfolio_manager.execute_trade(symbol, 'SELL', shares, price, date)
                            if success:
                                daily_trades += 1
                                logger.info(f"  ✅ EXECUTED SELL")
                                
            except Exception as e:
                logger.error(f"Error: {e}")
        
        # Update performance
        performance = self.portfolio_manager.update_portfolio_performance(current_prices)
        self.risk_manager.reset_daily_metrics()
        
        return {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash_balance': self.portfolio_manager.cash_balance,
            'trades': daily_trades,
            'open_positions': len(self.portfolio_manager.holdings),
            'daily_return': performance.get('daily_return', 0.0) if performance else 0.0
        }


# Run test
backtest = DebugTechnicalBacktest(
    start_date="2022-07-07",
    end_date="2023-12-31",  # Shorter period for debug
    initial_capital=100000.0
)

results = backtest.run_backtest(train_split=0.0)
print(f"\nTotal signals found: {backtest.signal_count}")
print(f"Total trades: {len(backtest.trade_log)}")
print(f"Final value: ${results['final_value']:,.2f}")