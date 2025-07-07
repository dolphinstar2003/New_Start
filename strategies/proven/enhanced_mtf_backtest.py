"""
Enhanced MTF Backtest with Advanced Risk Management
Kademeli stop loss ve trailing stop ile geliştirilmiş sistem
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR

logger.info("Enhanced MTF Backtest with Advanced Risk Management")
logger.info("="*80)

# Test parameters
START_DATE = "2024-01-01"
END_DATE = "2024-06-30"
INITIAL_CAPITAL = 100000

# Enhanced risk parameters
RISK_PARAMS = {
    'max_position_pct': 0.15,      # %15'e çıkardık (was 10%)
    'max_positions': 8,             # 8 pozisyon (was 5)
    'initial_stop_loss': 0.03,      # %3 ilk stop loss
    'trailing_stop_levels': [       # Kademeli trailing stop
        {'profit': 0.02, 'stop': 0.01},    # %2 karda %1 trailing
        {'profit': 0.05, 'stop': 0.025},   # %5 karda %2.5 trailing  
        {'profit': 0.10, 'stop': 0.05},    # %10 karda %5 trailing
        {'profit': 0.15, 'stop': 0.08},    # %15 karda %8 trailing
        {'profit': 0.20, 'stop': 0.12},    # %20 karda %12 trailing
    ],
    'position_sizing': 'volatility',  # Volatilite bazlı pozisyon boyutu
    'use_mtf_signals': True,          # Multi-timeframe sinyaller
}

# Portfolio tracking
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {}
}

# Load multi-timeframe data
def load_mtf_data(symbol):
    """Load data from multiple timeframes"""
    data = {}
    
    # Load 1d data (primary)
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    ind_file = DATA_DIR / 'indicators' / '1d' / f"{symbol}_1d_supertrend.csv"
    
    if raw_file.exists() and ind_file.exists():
        raw_df = pd.read_csv(raw_file)
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
        raw_df.set_index('datetime', inplace=True)
        if raw_df.index.tz is not None:
            raw_df.index = raw_df.index.tz_localize(None)
        
        ind_df = pd.read_csv(ind_file)
        ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
        ind_df.set_index('datetime', inplace=True)
        if ind_df.index.tz is not None:
            ind_df.index = ind_df.index.tz_localize(None)
        
        data['1d'] = raw_df.join(ind_df, rsuffix='_ind')
    
    # Load 1h data for intraday signals
    raw_1h = DATA_DIR / 'raw' / '1h' / f"{symbol}_1h_raw.csv"
    ind_1h = DATA_DIR / 'indicators' / '1h' / f"{symbol}_1h_supertrend.csv"
    
    if raw_1h.exists() and ind_1h.exists():
        raw_df = pd.read_csv(raw_1h)
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
        raw_df.set_index('datetime', inplace=True)
        
        ind_df = pd.read_csv(ind_1h)
        ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
        ind_df.set_index('datetime', inplace=True)
        
        # Resample to daily for alignment
        raw_daily = raw_df.resample('D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 
            'close': 'last', 'volume': 'sum'
        })
        
        ind_daily = ind_df.resample('D').agg({
            'trend': 'last',
            'buy_signal': 'any',
            'sell_signal': 'any'
        })
        
        data['1h'] = raw_daily.join(ind_daily, rsuffix='_1h')
    
    return data

# Calculate volatility for position sizing
def calculate_volatility(df, period=20):
    """Calculate historical volatility"""
    if len(df) < period:
        return 0.02  # Default 2%
    
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(period).std().iloc[-1]
    return max(volatility, 0.01)  # Min 1% volatility

# Get MTF signal strength
def get_mtf_signal(symbol_data, date):
    """Combine signals from multiple timeframes"""
    signals = {}
    weights = {'1h': 0.3, '1d': 0.7}  # 1d daha önemli
    
    # 1d signal
    if '1d' in symbol_data and date in symbol_data['1d'].index:
        row = symbol_data['1d'].loc[date]
        if row.get('buy_signal', False):
            signals['1d'] = 2
        elif row.get('trend', 0) == 1:
            signals['1d'] = 1
        elif row.get('sell_signal', False):
            signals['1d'] = -2
        elif row.get('trend', 0) == -1:
            signals['1d'] = -1
        else:
            signals['1d'] = 0
    
    # 1h signal (if available)
    if '1h' in symbol_data and date in symbol_data['1h'].index:
        row = symbol_data['1h'].loc[date]
        if row.get('buy_signal_1h', False):
            signals['1h'] = 2
        elif row.get('trend_1h', 0) == 1:
            signals['1h'] = 1
        elif row.get('sell_signal_1h', False):
            signals['1h'] = -2
        elif row.get('trend_1h', 0) == -1:
            signals['1h'] = -1
        else:
            signals['1h'] = 0
    
    # Calculate weighted signal
    if signals:
        weighted_signal = sum(signals.get(tf, 0) * weights.get(tf, 0) for tf in weights)
        return weighted_signal
    return 0

# Update trailing stops
def update_trailing_stops(portfolio, current_prices):
    """Update trailing stops based on profit levels"""
    for symbol, position in portfolio['positions'].items():
        if symbol not in current_prices:
            continue
        
        current_price = current_prices[symbol]
        entry_price = position['entry_price']
        
        # Calculate current profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Find appropriate trailing stop level
        trailing_stop = None
        for level in RISK_PARAMS['trailing_stop_levels']:
            if profit_pct >= level['profit']:
                trailing_stop = current_price * (1 - level['stop'])
        
        # Update trailing stop
        if trailing_stop:
            if symbol not in portfolio['trailing_stops']:
                portfolio['trailing_stops'][symbol] = trailing_stop
            else:
                # Only update if new stop is higher
                portfolio['trailing_stops'][symbol] = max(portfolio['trailing_stops'][symbol], trailing_stop)

# Load all symbol data
logger.info("Loading multi-timeframe data...")
all_symbol_data = {}
for symbol in SACRED_SYMBOLS:
    data = load_mtf_data(symbol)
    if data:
        all_symbol_data[symbol] = data
        logger.debug(f"Loaded {symbol}: {len(data)} timeframes")

# Get all trading days
all_dates = pd.DatetimeIndex([])
for symbol_data in all_symbol_data.values():
    if '1d' in symbol_data:
        # Remove timezone if present
        idx = symbol_data['1d'].index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        all_dates = all_dates.union(idx)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
if all_dates.tz is not None:
    start_dt = start_dt.tz_localize(all_dates.tz)
    end_dt = end_dt.tz_localize(all_dates.tz)

mask = (all_dates >= start_dt) & (all_dates <= end_dt)
trading_days = all_dates[mask]

logger.info(f"Running backtest from {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)")

# Run backtest
trade_count = 0
win_count = 0

for date in trading_days:
    daily_value = portfolio['cash']
    current_prices = {}
    
    # Get current prices
    for symbol, symbol_data in all_symbol_data.items():
        if '1d' in symbol_data and date in symbol_data['1d'].index:
            current_prices[symbol] = symbol_data['1d'].loc[date]['close']
    
    # Update trailing stops
    update_trailing_stops(portfolio, current_prices)
    
    # Check existing positions
    positions_to_close = []
    
    for symbol, position in portfolio['positions'].items():
        if symbol not in current_prices:
            continue
        
        current_price = current_prices[symbol]
        entry_price = position['entry_price']
        
        # Update position value
        position['current_value'] = position['shares'] * current_price
        daily_value += position['current_value']
        
        # Check stop loss
        stop_price = portfolio['stop_losses'].get(symbol, entry_price * (1 - RISK_PARAMS['initial_stop_loss']))
        
        # Check trailing stop
        if symbol in portfolio['trailing_stops']:
            stop_price = max(stop_price, portfolio['trailing_stops'][symbol])
        
        # Check if should close
        should_close = False
        close_reason = ""
        
        if current_price <= stop_price:
            should_close = True
            close_reason = "Stop Loss"
        elif '1d' in all_symbol_data[symbol]:
            row = all_symbol_data[symbol]['1d'].loc[date]
            mtf_signal = get_mtf_signal(all_symbol_data[symbol], date)
            
            # Exit on strong sell signal
            if mtf_signal <= -1.5 or row.get('sell_signal', False):
                should_close = True
                close_reason = "Sell Signal"
        
        if should_close:
            positions_to_close.append((symbol, close_reason))
    
    # Close positions
    for symbol, reason in positions_to_close:
        position = portfolio['positions'][symbol]
        current_price = current_prices[symbol]
        
        # Calculate profit
        exit_value = position['shares'] * current_price
        profit = exit_value - position['entry_value']
        profit_pct = (profit / position['entry_value']) * 100
        
        # Update portfolio
        portfolio['cash'] += exit_value
        
        # Record trade
        portfolio['trades'].append({
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'shares': position['shares'],
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason
        })
        
        # Clean up
        del portfolio['positions'][symbol]
        if symbol in portfolio['stop_losses']:
            del portfolio['stop_losses'][symbol]
        if symbol in portfolio['trailing_stops']:
            del portfolio['trailing_stops'][symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check for new entries
    if len(portfolio['positions']) < RISK_PARAMS['max_positions']:
        # Sort symbols by signal strength
        entry_candidates = []
        
        for symbol, symbol_data in all_symbol_data.items():
            if symbol in portfolio['positions']:
                continue
            
            if '1d' not in symbol_data or date not in symbol_data['1d'].index:
                continue
            
            row = symbol_data['1d'].loc[date]
            mtf_signal = get_mtf_signal(symbol_data, date)
            
            # Strong buy signal
            if mtf_signal >= 1.0 or row.get('buy_signal', False):
                volatility = calculate_volatility(symbol_data['1d'].loc[:date])
                entry_candidates.append({
                    'symbol': symbol,
                    'signal': mtf_signal,
                    'volatility': volatility,
                    'price': current_prices.get(symbol, 0)
                })
        
        # Sort by signal strength
        entry_candidates.sort(key=lambda x: x['signal'], reverse=True)
        
        # Enter positions
        for candidate in entry_candidates[:2]:  # Max 2 new positions per day
            symbol = candidate['symbol']
            current_price = candidate['price']
            volatility = candidate['volatility']
            
            if current_price <= 0:
                continue
            
            # Calculate position size (volatility-based)
            base_position = portfolio['cash'] * RISK_PARAMS['max_position_pct']
            volatility_adj = 1.0 / (1.0 + volatility * 10)  # Lower size for higher volatility
            position_size = base_position * volatility_adj
            
            # Ensure we have enough cash
            position_size = min(position_size, portfolio['cash'] * 0.95)
            shares = int(position_size / current_price)
            
            if shares > 0:
                # Open position
                entry_value = shares * current_price
                portfolio['cash'] -= entry_value
                
                portfolio['positions'][symbol] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'entry_value': entry_value,
                    'entry_date': date,
                    'current_value': entry_value
                }
                
                # Set initial stop loss
                portfolio['stop_losses'][symbol] = current_price * (1 - RISK_PARAMS['initial_stop_loss'])
                
                logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} (Vol: {volatility:.1%})")
    
    # Record daily value
    portfolio['daily_values'].append({
        'date': date,
        'value': daily_value,
        'cash': portfolio['cash'],
        'positions': len(portfolio['positions'])
    })

# Close all remaining positions
final_date = trading_days[-1]
for symbol, position in list(portfolio['positions'].items()):
    if symbol in current_prices:
        current_price = current_prices[symbol]
        exit_value = position['shares'] * current_price
        profit = exit_value - position['entry_value']
        profit_pct = (profit / position['entry_value']) * 100
        
        portfolio['cash'] += exit_value
        
        portfolio['trades'].append({
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': final_date,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'shares': position['shares'],
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': 'End of Test'
        })
        
        trade_count += 1
        if profit > 0:
            win_count += 1

# Calculate final metrics
final_value = portfolio['cash']
total_return = final_value - INITIAL_CAPITAL
total_return_pct = (total_return / INITIAL_CAPITAL) * 100
monthly_return = total_return_pct / 6  # 6 months
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
print("\n" + "="*80)
print("ENHANCED MTF BACKTEST RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return: {monthly_return:.2f}%")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("="*80)

# Trade analysis
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    
    print("\nTrade Analysis:")
    print(f"Average Win: {trades_df[trades_df['profit'] > 0]['profit_pct'].mean():.2f}%")
    print(f"Average Loss: {trades_df[trades_df['profit'] < 0]['profit_pct'].mean():.2f}%")
    print(f"Profit Factor: {abs(trades_df[trades_df['profit'] > 0]['profit'].sum() / trades_df[trades_df['profit'] < 0]['profit'].sum()):.2f}")
    
    # Exit reason analysis
    print("\nExit Reasons:")
    print(trades_df['reason'].value_counts())
    
    print("\nTop 5 Winning Trades:")
    print(trades_df.nlargest(5, 'profit')[['symbol', 'entry_date', 'exit_date', 'profit', 'profit_pct', 'reason']])
    
    print("\nTop 5 Losing Trades:")
    print(trades_df.nsmallest(5, 'profit')[['symbol', 'entry_date', 'exit_date', 'profit', 'profit_pct', 'reason']])
    
    # Monthly breakdown
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly_profits = trades_df.groupby('exit_month')['profit'].sum()
    
    print("\nMonthly P&L:")
    for month, profit in monthly_profits.items():
        pct = (profit / INITIAL_CAPITAL) * 100
        print(f"  {month}: ${profit:,.2f} ({pct:.2f}%)")

# Save detailed results
results_file = DATA_DIR.parent / 'enhanced_mtf_backtest_results.csv'
trades_df = pd.DataFrame(portfolio['trades'])
trades_df.to_csv(results_file, index=False)
logger.info(f"\nDetailed results saved to: {results_file}")