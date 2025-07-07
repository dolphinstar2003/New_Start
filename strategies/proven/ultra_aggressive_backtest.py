"""
Ultra Aggressive MTF Backtest
Hedef: Aylık %5-6 getiri
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR

logger.info("Ultra Aggressive MTF Backtest")
logger.info("="*80)

# Test parameters
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"
INITIAL_CAPITAL = 100000

# Ultra aggressive parameters
RISK_PARAMS = {
    'max_position_pct': 0.25,      # %25 pozisyon büyüklüğü
    'max_positions': 10,            # 10 pozisyon aynı anda
    'initial_stop_loss': 0.02,      # %2 ilk stop loss (sıkı)
    'take_profit_levels': [         # Kademeli kar al
        {'profit': 0.03, 'exit_pct': 0.25},   # %3'te %25 çık
        {'profit': 0.05, 'exit_pct': 0.25},   # %5'te %25 çık
        {'profit': 0.08, 'exit_pct': 0.25},   # %8'de %25 çık
        {'profit': 0.12, 'exit_pct': 0.25},   # %12'de kalan çık
    ],
    'trailing_stop_levels': [       # Agresif trailing stop
        {'profit': 0.02, 'stop': 0.005},   # %2 karda %0.5 trailing
        {'profit': 0.04, 'stop': 0.015},   # %4 karda %1.5 trailing  
        {'profit': 0.06, 'stop': 0.025},   # %6 karda %2.5 trailing
        {'profit': 0.10, 'stop': 0.04},    # %10 karda %4 trailing
    ],
    'use_leverage': True,           # Kaldıraç kullan
    'leverage_ratio': 2.0,          # 2x kaldıraç
    'signal_threshold': 1.5,        # Daha yüksek sinyal eşiği
}

# Portfolio tracking
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'take_profits': {}
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
    
    # Supertrend
    if row.get('supertrend_buy_signal', False):
        signal += 3
    elif row.get('supertrend_trend', 0) == 1:
        signal += 1
    elif row.get('supertrend_sell_signal', False):
        signal -= 3
    elif row.get('supertrend_trend', 0) == -1:
        signal -= 1
    
    # ADX/DI
    if 'adx_di_adx' in row and pd.notna(row['adx_di_adx']):
        if row['adx_di_adx'] > 25:  # Strong trend
            if row.get('adx_di_plus_di', 0) > row.get('adx_di_minus_di', 0):
                signal += 2
            else:
                signal -= 2
    
    # Squeeze Momentum
    if 'squeeze_momentum_squeeze' in row:
        if row.get('squeeze_momentum_squeeze', False) == False:  # Squeeze off = momentum
            if row.get('squeeze_momentum_momentum', 0) > 0:
                signal += 2
            else:
                signal -= 2
    
    # WaveTrend
    if 'wavetrend_buy' in row:
        if row.get('wavetrend_buy', False):
            signal += 2
        elif row.get('wavetrend_sell', False):
            signal -= 2
    
    # MACD
    if 'macd_custom_buy_signal' in row:
        if row.get('macd_custom_buy_signal', False):
            signal += 2
        elif row.get('macd_custom_sell_signal', False):
            signal -= 2
    
    # 1h confirmation
    if row.get('buy_signal_1h', False):
        signal += 1
    elif row.get('sell_signal_1h', False):
        signal -= 1
    
    return signal

# Calculate position size with leverage
def calculate_leveraged_position_size(portfolio_value, volatility, leverage=2.0):
    """Calculate position size with leverage"""
    base_size = portfolio_value * RISK_PARAMS['max_position_pct']
    
    # Volatility adjustment (inverse - lower volatility = larger position)
    vol_adj = 1.0 / (1.0 + volatility * 5)  # More aggressive
    
    # Apply leverage
    position_size = base_size * vol_adj * leverage
    
    # Cap at available cash (with margin)
    max_size = portfolio['cash'] * leverage
    
    return min(position_size, max_size * 0.9)

# Update trailing stops and take profits
def update_stops_and_targets(portfolio, current_prices):
    """Update trailing stops and take profit levels"""
    positions_to_close = []
    
    for symbol, position in portfolio['positions'].items():
        if symbol not in current_prices:
            continue
        
        current_price = current_prices[symbol]
        entry_price = position['entry_price']
        
        # Calculate current profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Check take profit levels
        if symbol not in portfolio['take_profits']:
            portfolio['take_profits'][symbol] = {'levels_hit': 0}
        
        tp_data = portfolio['take_profits'][symbol]
        levels_hit = tp_data['levels_hit']
        
        # Check if we should take partial profits
        for i, level in enumerate(RISK_PARAMS['take_profit_levels']):
            if i >= levels_hit and profit_pct >= level['profit']:
                # Take partial profit
                shares_to_sell = int(position['original_shares'] * level['exit_pct'])
                if shares_to_sell > 0 and shares_to_sell <= position['shares']:
                    positions_to_close.append((symbol, 'Partial TP', shares_to_sell))
                    tp_data['levels_hit'] = i + 1
                    break
        
        # Update trailing stop
        trailing_stop = None
        for level in RISK_PARAMS['trailing_stop_levels']:
            if profit_pct >= level['profit']:
                trailing_stop = current_price * (1 - level['stop'])
        
        if trailing_stop:
            if symbol not in portfolio['trailing_stops']:
                portfolio['trailing_stops'][symbol] = trailing_stop
            else:
                portfolio['trailing_stops'][symbol] = max(portfolio['trailing_stops'][symbol], trailing_stop)
    
    return positions_to_close

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

logger.info(f"Running ultra aggressive backtest from {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)")

# Run backtest
trade_count = 0
win_count = 0
total_trades = []

for date in trading_days:
    daily_value = portfolio['cash']
    current_prices = {}
    
    # Get current prices
    for symbol, symbol_data in all_symbol_data.items():
        if '1d' in symbol_data and date in symbol_data['1d'].index:
            current_prices[symbol] = symbol_data['1d'].loc[date]['close']
    
    # Update stops and take profits
    partial_closes = update_stops_and_targets(portfolio, current_prices)
    
    # Process partial closes
    for symbol, reason, shares_to_sell in partial_closes:
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
        
        # Record partial trade
        total_trades.append({
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': date,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'shares': shares_to_sell,
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason
        })
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: PARTIAL {symbol} - {reason} - {shares_to_sell} shares - Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check existing positions
    positions_to_close = []
    
    for symbol, position in list(portfolio['positions'].items()):
        if position['shares'] <= 0:  # Position fully closed
            del portfolio['positions'][symbol]
            continue
            
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
            signal_strength = calculate_signal_strength(row)
            
            # Exit on strong reversal signal
            if signal_strength <= -RISK_PARAMS['signal_threshold']:
                should_close = True
                close_reason = "Signal Reversal"
        
        if should_close:
            positions_to_close.append((symbol, close_reason))
    
    # Close positions
    for symbol, reason in positions_to_close:
        position = portfolio['positions'][symbol]
        current_price = current_prices[symbol]
        
        # Calculate profit
        exit_value = position['shares'] * current_price
        entry_value = position['shares'] * position['entry_price']
        profit = exit_value - entry_value
        profit_pct = (profit / entry_value) * 100
        
        # Update portfolio
        portfolio['cash'] += exit_value
        
        # Record trade
        total_trades.append({
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
        if symbol in portfolio['take_profits']:
            del portfolio['take_profits'][symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - Profit: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check for new entries
    if len(portfolio['positions']) < RISK_PARAMS['max_positions']:
        # Calculate portfolio value for position sizing
        portfolio_value = daily_value
        
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
            if positions_opened >= 3:  # Max 3 new positions per day
                break
                
            symbol = candidate['symbol']
            current_price = candidate['price']
            volatility = candidate['volatility']
            
            if current_price <= 0:
                continue
            
            # Calculate leveraged position size
            position_size = calculate_leveraged_position_size(portfolio_value, volatility, RISK_PARAMS['leverage_ratio'])
            
            # Ensure we have enough cash (considering leverage)
            required_cash = position_size / RISK_PARAMS['leverage_ratio']
            if required_cash > portfolio['cash'] * 0.95:
                continue
            
            shares = int(position_size / current_price)
            
            if shares > 0:
                # Open position
                actual_cost = (shares * current_price) / RISK_PARAMS['leverage_ratio']
                portfolio['cash'] -= actual_cost
                
                portfolio['positions'][symbol] = {
                    'shares': shares,
                    'original_shares': shares,
                    'entry_price': current_price,
                    'entry_value': shares * current_price,
                    'entry_date': date,
                    'current_value': shares * current_price,
                    'leveraged': True,
                    'leverage_ratio': RISK_PARAMS['leverage_ratio']
                }
                
                # Set initial stop loss
                portfolio['stop_losses'][symbol] = current_price * (1 - RISK_PARAMS['initial_stop_loss'])
                
                positions_opened += 1
                logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} (Signal: {candidate['signal']:.1f}, Vol: {volatility:.1%}) - Leveraged {RISK_PARAMS['leverage_ratio']}x")
    
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
        
        # Handle partial shares
        if position['shares'] > 0:
            exit_value = position['shares'] * current_price
            entry_value = position['shares'] * position['entry_price']
            profit = exit_value - entry_value
            profit_pct = (profit / entry_value) * 100
            
            # Return leveraged position
            portfolio['cash'] += exit_value / RISK_PARAMS['leverage_ratio']
            
            total_trades.append({
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

# Combine portfolio trades with total trades
portfolio['trades'] = total_trades

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
print("ULTRA AGGRESSIVE MTF BACKTEST RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return (Simple): {monthly_return:.2f}%")
print(f"Monthly Return (Compound): {compound_monthly:.2f}%")
print(f"Leverage Used: {RISK_PARAMS['leverage_ratio']}x")
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
    
    print("\nTop 10 Trades by Profit:")
    print(trades_df.nlargest(10, 'profit')[['symbol', 'entry_date', 'exit_date', 'profit', 'profit_pct', 'reason']])
    
    # Monthly breakdown
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly_profits = trades_df.groupby('exit_month')['profit'].sum()
    
    print("\nMonthly P&L:")
    cumulative_capital = INITIAL_CAPITAL
    for month, profit in monthly_profits.items():
        monthly_pct = (profit / cumulative_capital) * 100
        cumulative_capital += profit
        print(f"  {month}: ${profit:,.2f} ({monthly_pct:.2f}%)")

# Save detailed results
results_file = DATA_DIR.parent / 'ultra_aggressive_backtest_results.csv'
trades_df = pd.DataFrame(portfolio['trades'])
trades_df.to_csv(results_file, index=False)
logger.info(f"\nDetailed results saved to: {results_file}")

print("\n" + "="*80)
print("STRATEGY DETAILS:")
print(f"- Position Size: {RISK_PARAMS['max_position_pct']*100}% with {RISK_PARAMS['leverage_ratio']}x leverage")
print(f"- Max Positions: {RISK_PARAMS['max_positions']}")
print(f"- Initial Stop Loss: {RISK_PARAMS['initial_stop_loss']*100}%")
print(f"- Signal Threshold: {RISK_PARAMS['signal_threshold']}")
print("- Take Profit Levels: 3%, 5%, 8%, 12% (25% each)")
print("- Uses all 5 core indicators for signal generation")
print("="*80)