"""
Realistic High Return MTF Backtest
Target: 5-6% monthly with proper risk controls
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

logger.info("Realistic High Return MTF Backtest")
logger.info("="*80)

# Test parameters
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"
INITIAL_CAPITAL = 100000

# Aggressive but realistic parameters
RISK_PARAMS = {
    'max_position_pct': 0.20,       # %20 pozisyon büyüklüğü
    'max_positions': 8,             # 8 pozisyon aynı anda
    'max_portfolio_risk': 0.80,     # Portfolio'nun max %80'i pozisyonda olabilir
    'initial_stop_loss': 0.025,     # %2.5 ilk stop loss
    'profit_targets': {             # Kar hedefleri
        'quick': 0.03,              # %3 hızlı kar al
        'normal': 0.06,             # %6 normal hedef
        'extended': 0.10,           # %10 genişletilmiş hedef
    },
    'trailing_stop_levels': [       # Kademeli trailing stop
        {'profit': 0.03, 'stop': 0.01},    # %3 karda %1 trailing
        {'profit': 0.05, 'stop': 0.02},    # %5 karda %2 trailing  
        {'profit': 0.08, 'stop': 0.035},   # %8 karda %3.5 trailing
        {'profit': 0.12, 'stop': 0.06},    # %12 karda %6 trailing
    ],
    'signal_threshold': 2.0,        # Güçlü sinyal eşiği
    'exit_signal_threshold': -1.5,  # Çıkış sinyali eşiği
}

# Portfolio tracking
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'profit_targets': {}
}

# Load multi-timeframe data with all indicators
def load_enhanced_data(symbol):
    """Load data with all indicators"""
    data = {}
    
    # Load 1d data with all indicators
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    
    # Load all indicators
    indicators = ['supertrend', 'adx_di', 'squeeze_momentum', 'wavetrend', 'macd_custom']
    ind_dfs = []
    
    for indicator in indicators:
        ind_file = DATA_DIR / 'indicators' / '1d' / f"{symbol}_1d_{indicator}.csv"
        if ind_file.exists():
            df = pd.read_csv(ind_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Prefix columns
            df.columns = [f"{indicator}_{col}" if col != 'datetime' else col for col in df.columns]
            ind_dfs.append(df)
    
    if raw_file.exists() and ind_dfs:
        raw_df = pd.read_csv(raw_file)
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
        raw_df.set_index('datetime', inplace=True)
        if raw_df.index.tz is not None:
            raw_df.index = raw_df.index.tz_localize(None)
        
        # Merge all indicators
        merged_df = raw_df
        for ind_df in ind_dfs:
            merged_df = merged_df.join(ind_df, how='left')
        
        data['1d'] = merged_df
        
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
            
            # Resample to daily
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

# Calculate comprehensive signal strength
def calculate_signal_strength(row):
    """Calculate signal strength from all indicators"""
    signal = 0
    
    # Supertrend (primary signal)
    if row.get('supertrend_buy_signal', False):
        signal += 2.5
    elif row.get('supertrend_trend', 0) == 1:
        signal += 1
    elif row.get('supertrend_sell_signal', False):
        signal -= 2.5
    elif row.get('supertrend_trend', 0) == -1:
        signal -= 1
    
    # ADX/DI (trend strength)
    if 'adx_di_adx' in row and pd.notna(row['adx_di_adx']):
        if row['adx_di_adx'] > 25:  # Strong trend
            if row.get('adx_di_plus_di', 0) > row.get('adx_di_minus_di', 0):
                signal += 1.5
            else:
                signal -= 1.5
    
    # Squeeze Momentum (volatility breakout)
    if 'squeeze_momentum_squeeze' in row:
        if row.get('squeeze_momentum_squeeze', False) == False:  # Squeeze off
            if row.get('squeeze_momentum_momentum', 0) > 0:
                signal += 1.5
            else:
                signal -= 1.5
    
    # WaveTrend (momentum)
    if 'wavetrend_buy' in row:
        if row.get('wavetrend_buy', False):
            signal += 1.5
        elif row.get('wavetrend_sell', False):
            signal -= 1.5
    
    # MACD (confirmation)
    if 'macd_custom_buy_signal' in row:
        if row.get('macd_custom_buy_signal', False):
            signal += 1.5
        elif row.get('macd_custom_sell_signal', False):
            signal -= 1.5
    
    # 1h intraday confirmation
    if row.get('buy_signal_1h', False):
        signal += 0.5
    elif row.get('sell_signal_1h', False):
        signal -= 0.5
    
    return signal

# Calculate position size based on volatility and signal strength
def calculate_position_size(portfolio_value, cash_available, volatility, signal_strength):
    """Calculate position size with risk management"""
    # Base position size
    base_size = portfolio_value * RISK_PARAMS['max_position_pct']
    
    # Volatility adjustment (lower volatility = larger position)
    vol_adj = 1.0 / (1.0 + volatility * 8)
    
    # Signal strength adjustment
    signal_adj = min(1.2, 0.8 + (signal_strength / 10))
    
    # Calculate final position size
    position_size = base_size * vol_adj * signal_adj
    
    # Ensure we don't exceed available cash
    position_size = min(position_size, cash_available * 0.95)
    
    # Ensure portfolio risk limits
    return position_size

# Update trailing stops based on profit
def update_trailing_stops(portfolio, current_prices):
    """Update trailing stops based on current profit"""
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

# Check profit targets
def check_profit_targets(portfolio, symbol, current_price):
    """Check if profit targets are hit"""
    position = portfolio['positions'][symbol]
    entry_price = position['entry_price']
    profit_pct = (current_price - entry_price) / entry_price
    
    # Initialize profit target tracking
    if symbol not in portfolio['profit_targets']:
        portfolio['profit_targets'][symbol] = {
            'quick_hit': False,
            'normal_hit': False,
            'extended_hit': False
        }
    
    targets = portfolio['profit_targets'][symbol]
    
    # Check quick target (partial exit)
    if not targets['quick_hit'] and profit_pct >= RISK_PARAMS['profit_targets']['quick']:
        targets['quick_hit'] = True
        return 'quick', 0.33  # Exit 33% of position
    
    # Check normal target (partial exit)
    elif not targets['normal_hit'] and profit_pct >= RISK_PARAMS['profit_targets']['normal']:
        targets['normal_hit'] = True
        return 'normal', 0.50  # Exit 50% of remaining
    
    # Check extended target (full exit)
    elif not targets['extended_hit'] and profit_pct >= RISK_PARAMS['profit_targets']['extended']:
        targets['extended_hit'] = True
        return 'extended', 1.0  # Exit all remaining
    
    return None, 0

# Load all symbol data
logger.info("Loading enhanced multi-timeframe data...")
all_symbol_data = {}
for symbol in SACRED_SYMBOLS:
    data = load_enhanced_data(symbol)
    if data:
        all_symbol_data[symbol] = data
        logger.debug(f"Loaded {symbol}: {len(data)} timeframes")

# Get all trading days
all_dates = pd.DatetimeIndex([])
for symbol_data in all_symbol_data.values():
    if '1d' in symbol_data:
        idx = symbol_data['1d'].index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        all_dates = all_dates.union(idx)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
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
    positions_to_reduce = []
    
    for symbol, position in portfolio['positions'].items():
        if symbol not in current_prices:
            continue
        
        current_price = current_prices[symbol]
        entry_price = position['entry_price']
        
        # Update position value
        position['current_value'] = position['shares'] * current_price
        daily_value += position['current_value']
        
        # Check profit targets
        target_type, exit_pct = check_profit_targets(portfolio, symbol, current_price)
        if target_type:
            shares_to_sell = int(position['shares'] * exit_pct)
            if shares_to_sell > 0:
                positions_to_reduce.append((symbol, target_type, shares_to_sell))
                continue
        
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
            signal_strength = calculate_signal_strength(row)
            
            # Exit on strong reversal signal
            if signal_strength <= RISK_PARAMS['exit_signal_threshold']:
                should_close = True
                close_reason = "Signal Exit"
        
        if should_close:
            positions_to_close.append((symbol, close_reason))
    
    # Process partial exits (profit targets)
    for symbol, target_type, shares_to_sell in positions_to_reduce:
        position = portfolio['positions'][symbol]
        current_price = current_prices[symbol]
        
        # Calculate profit for partial exit
        exit_value = shares_to_sell * current_price
        entry_value = shares_to_sell * position['entry_price']
        profit = exit_value - entry_value
        profit_pct = (profit / entry_value) * 100
        
        # Update position
        position['shares'] -= shares_to_sell
        portfolio['cash'] += exit_value
        
        # Record trade
        portfolio['trades'].append({
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'shares': shares_to_sell,
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': f"Target {target_type}"
        })
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: TARGET {symbol} - {target_type} - {shares_to_sell} shares - Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Close full positions
    for symbol, reason in positions_to_close:
        position = portfolio['positions'][symbol]
        if position['shares'] <= 0:  # Already fully exited
            continue
            
        current_price = current_prices[symbol]
        
        # Calculate profit
        exit_value = position['shares'] * current_price
        entry_value = position['shares'] * position['entry_price']
        profit = exit_value - entry_value
        profit_pct = (profit / entry_value) * 100
        
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
        if symbol in portfolio['profit_targets']:
            del portfolio['profit_targets'][symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check for new entries
    current_portfolio_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
    portfolio_risk_pct = current_portfolio_value / daily_value if daily_value > 0 else 0
    
    if len(portfolio['positions']) < RISK_PARAMS['max_positions'] and portfolio_risk_pct < RISK_PARAMS['max_portfolio_risk']:
        # Sort symbols by signal strength
        entry_candidates = []
        
        for symbol, symbol_data in all_symbol_data.items():
            if symbol in portfolio['positions']:
                continue
            
            if '1d' not in symbol_data or date not in symbol_data['1d'].index:
                continue
            
            row = symbol_data['1d'].loc[date]
            signal_strength = calculate_signal_strength(row)
            
            # Strong buy signal
            if signal_strength >= RISK_PARAMS['signal_threshold']:
                # Calculate volatility
                if len(symbol_data['1d'].loc[:date]) >= 20:
                    returns = symbol_data['1d'].loc[:date]['close'].pct_change().dropna()
                    volatility = returns.rolling(20).std().iloc[-1]
                else:
                    volatility = 0.02
                
                entry_candidates.append({
                    'symbol': symbol,
                    'signal': signal_strength,
                    'volatility': volatility,
                    'price': current_prices.get(symbol, 0)
                })
        
        # Sort by signal strength
        entry_candidates.sort(key=lambda x: x['signal'], reverse=True)
        
        # Enter positions
        positions_opened = 0
        for candidate in entry_candidates:
            if positions_opened >= 2:  # Max 2 new positions per day
                break
            
            # Check portfolio risk again
            current_portfolio_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
            portfolio_risk_pct = current_portfolio_value / daily_value if daily_value > 0 else 0
            
            if portfolio_risk_pct >= RISK_PARAMS['max_portfolio_risk']:
                break
            
            symbol = candidate['symbol']
            current_price = candidate['price']
            volatility = candidate['volatility']
            signal_strength = candidate['signal']
            
            if current_price <= 0:
                continue
            
            # Calculate position size
            position_size = calculate_position_size(daily_value, portfolio['cash'], volatility, signal_strength)
            
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
                    'current_value': entry_value,
                    'signal_strength': signal_strength
                }
                
                # Set initial stop loss
                portfolio['stop_losses'][symbol] = current_price * (1 - RISK_PARAMS['initial_stop_loss'])
                
                positions_opened += 1
                logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} (Signal: {signal_strength:.1f}, Vol: {volatility:.1%})")
    
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
    if position['shares'] <= 0:
        continue
        
    if symbol in current_prices:
        current_price = current_prices[symbol]
        exit_value = position['shares'] * current_price
        entry_value = position['shares'] * position['entry_price']
        profit = exit_value - entry_value
        profit_pct = (profit / entry_value) * 100
        
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

# Calculate compound monthly return
compound_monthly = ((final_value / INITIAL_CAPITAL) ** (1/6) - 1) * 100

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
print("REALISTIC HIGH RETURN MTF BACKTEST RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return (Simple): {monthly_return:.2f}%")
print(f"Monthly Return (Compound): {compound_monthly:.2f}%")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("="*80)

# Trade analysis
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    
    print("\nTrade Analysis:")
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] < 0]
    
    if len(winning_trades) > 0:
        print(f"Average Win: {winning_trades['profit_pct'].mean():.2f}%")
        print(f"Largest Win: {winning_trades['profit_pct'].max():.2f}%")
    
    if len(losing_trades) > 0:
        print(f"Average Loss: {losing_trades['profit_pct'].mean():.2f}%")
        print(f"Largest Loss: {losing_trades['profit_pct'].min():.2f}%")
    
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum())
        print(f"Profit Factor: {profit_factor:.2f}")
    
    # Exit reason analysis
    print("\nExit Reasons:")
    print(trades_df['reason'].value_counts())
    
    # Monthly breakdown
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly_profits = trades_df.groupby('exit_month')['profit'].sum()
    
    print("\nMonthly P&L:")
    cumulative_capital = INITIAL_CAPITAL
    for month, profit in monthly_profits.items():
        monthly_pct = (profit / cumulative_capital) * 100
        cumulative_capital += profit
        print(f"  {month}: ${profit:,.2f} ({monthly_pct:.2f}%)")
    
    # Best trades
    print("\nTop 10 Trades:")
    print(trades_df.nlargest(10, 'profit')[['symbol', 'entry_date', 'exit_date', 'profit', 'profit_pct', 'reason']])

# Save detailed results
results_file = DATA_DIR.parent / 'realistic_high_return_results.csv'
trades_df = pd.DataFrame(portfolio['trades'])
trades_df.to_csv(results_file, index=False)
logger.info(f"\nDetailed results saved to: {results_file}")

print("\n" + "="*80)
print("STRATEGY SUMMARY:")
print(f"- Position Size: {RISK_PARAMS['max_position_pct']*100}% (volatility adjusted)")
print(f"- Max Positions: {RISK_PARAMS['max_positions']}")
print(f"- Max Portfolio Risk: {RISK_PARAMS['max_portfolio_risk']*100}%")
print(f"- Stop Loss: {RISK_PARAMS['initial_stop_loss']*100}%")
print(f"- Profit Targets: {RISK_PARAMS['profit_targets']['quick']*100}%, {RISK_PARAMS['profit_targets']['normal']*100}%, {RISK_PARAMS['profit_targets']['extended']*100}%")
print(f"- Signal Threshold: {RISK_PARAMS['signal_threshold']}")
print("- Uses all 5 core indicators with multi-timeframe confirmation")
print("="*80)