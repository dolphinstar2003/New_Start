"""
Quick MTF Technical Backtest
Simple test without ML models
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR

logger.info("Quick MTF Technical Backtest")
logger.info("="*60)

# Test parameters
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"
INITIAL_CAPITAL = 100000

# Simple portfolio tracking
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': []
}

# Load daily data for all symbols
symbol_data = {}
for symbol in SACRED_SYMBOLS[:20]:  # Test with first 5
    try:
        # Load 1d raw data and indicators
        raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
        indicator_file = DATA_DIR / 'indicators' / '1d' / f"{symbol}_1d_supertrend.csv"
        
        if raw_file.exists() and indicator_file.exists():
            # Load raw data for prices
            raw_df = pd.read_csv(raw_file)
            raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
            raw_df.set_index('datetime', inplace=True)
            
            # Load indicator data
            ind_df = pd.read_csv(indicator_file)
            ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
            ind_df.set_index('datetime', inplace=True)
            
            # Merge data
            df = raw_df.join(ind_df, rsuffix='_ind')
            
            # Filter date range
            mask = (df.index >= START_DATE) & (df.index <= END_DATE)
            symbol_data[symbol] = df[mask]
            logger.info(f"Loaded {symbol}: {len(df[mask])} days")
    except Exception as e:
        logger.error(f"Error loading {symbol}: {e}")

# Run simple backtest
trade_count = 0
win_count = 0

# Get all trading days
all_dates = pd.DatetimeIndex([])
for df in symbol_data.values():
    all_dates = all_dates.union(df.index)
all_dates = all_dates.sort_values()

logger.info(f"\nRunning backtest from {all_dates[0]} to {all_dates[-1]}")

for date in all_dates:
    daily_value = portfolio['cash']
    
    # Check each symbol
    for symbol, df in symbol_data.items():
        if date not in df.index:
            continue
        
        row = df.loc[date]
        current_price = row['close']
        
        # Update position values
        if symbol in portfolio['positions']:
            pos = portfolio['positions'][symbol]
            pos['value'] = pos['shares'] * current_price
            daily_value += pos['value']
            
            # Check exit signal
            if row.get('sell_signal', False) or (row['trend'] == -1 and pos['entry_trend'] == 1):
                # Close position
                exit_value = pos['shares'] * current_price
                profit = exit_value - pos['entry_value']
                portfolio['cash'] += exit_value
                
                portfolio['trades'].append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'shares': pos['shares'],
                    'profit': profit,
                    'profit_pct': (profit / pos['entry_value']) * 100
                })
                
                del portfolio['positions'][symbol]
                trade_count += 1
                if profit > 0:
                    win_count += 1
                
                logger.debug(f"{date.date()}: SELL {symbol} - Profit: ${profit:.2f}")
        
        else:
            # Check entry signal
            if row.get('buy_signal', False) and len(portfolio['positions']) < 3:
                # Calculate position size (equal weight)
                position_size = min(portfolio['cash'] * 0.3, 10000)  # Max 30% or $10k
                shares = int(position_size / current_price)
                
                if shares > 0 and position_size <= portfolio['cash']:
                    portfolio['cash'] -= shares * current_price
                    portfolio['positions'][symbol] = {
                        'shares': shares,
                        'entry_price': current_price,
                        'entry_value': shares * current_price,
                        'entry_date': date,
                        'entry_trend': row['trend'],
                        'value': shares * current_price
                    }
                    
                    logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f}")
    
    # Record daily value
    portfolio['daily_values'].append({
        'date': date,
        'value': daily_value,
        'cash': portfolio['cash'],
        'positions': len(portfolio['positions'])
    })

# Close all remaining positions
final_date = all_dates[-1]
for symbol, pos in list(portfolio['positions'].items()):
    if symbol in symbol_data:
        df = symbol_data[symbol]
        if final_date in df.index:
            current_price = df.loc[final_date]['close']
            exit_value = pos['shares'] * current_price
            profit = exit_value - pos['entry_value']
            portfolio['cash'] += exit_value
            
            portfolio['trades'].append({
                'symbol': symbol,
                'entry_date': pos['entry_date'],
                'exit_date': final_date,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'shares': pos['shares'],
                'profit': profit,
                'profit_pct': (profit / pos['entry_value']) * 100
            })
            
            trade_count += 1
            if profit > 0:
                win_count += 1

# Calculate final metrics
final_value = portfolio['cash']
total_return = final_value - INITIAL_CAPITAL
total_return_pct = (total_return / INITIAL_CAPITAL) * 100
win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

# Calculate Sharpe ratio
daily_df = pd.DataFrame(portfolio['daily_values'])
daily_df['returns'] = daily_df['value'].pct_change()
sharpe = daily_df['returns'].mean() / daily_df['returns'].std() * np.sqrt(252) if daily_df['returns'].std() > 0 else 0

# Calculate max drawdown
daily_df['cummax'] = daily_df['value'].cummax()
daily_df['drawdown'] = (daily_df['cummax'] - daily_df['value']) / daily_df['cummax']
max_drawdown = daily_df['drawdown'].max() * 100

# Print results
print("\n" + "="*60)
print("QUICK MTF TECHNICAL BACKTEST RESULTS")
print("="*60)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("="*60)

# Trade summary
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    print("\nTop 5 Winning Trades:")
    print(trades_df.nlargest(5, 'profit')[['symbol', 'entry_date', 'exit_date', 'profit', 'profit_pct']])
    
    print("\nTop 5 Losing Trades:")
    print(trades_df.nsmallest(5, 'profit')[['symbol', 'entry_date', 'exit_date', 'profit', 'profit_pct']])
    
    # Symbol performance
    print("\nPerformance by Symbol:")
    symbol_perf = trades_df.groupby('symbol').agg({
        'profit': ['sum', 'count'],
        'profit_pct': 'mean'
    }).round(2)
    print(symbol_perf)