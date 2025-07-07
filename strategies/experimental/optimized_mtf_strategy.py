"""
Optimized MTF Trading Strategy
Combines best practices for consistent returns
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

logger.info("Optimized MTF Trading Strategy")
logger.info("="*80)

# Test parameters
START_DATE = "2024-01-01"  # Test on 2024 data instead
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 100000

# Optimized parameters for consistent returns
STRATEGY_PARAMS = {
    # Position sizing
    'base_position_pct': 0.15,      # Base %15 pozisyon
    'max_position_pct': 0.25,       # Max %25 (güçlü sinyallerde)
    'max_positions': 6,             # Max 6 pozisyon
    'max_portfolio_risk': 0.75,     # Portfolio'nun max %75'i risk altında
    
    # Risk management
    'stop_loss_levels': {
        'tight': 0.02,              # %2 sıkı stop (volatil piyasa)
        'normal': 0.03,             # %3 normal stop
        'wide': 0.04,               # %4 geniş stop (düşük volatilite)
    },
    
    # Profit management
    'quick_profit': 0.025,          # %2.5 hızlı kar al
    'trailing_activation': 0.04,    # %4'te trailing başlat
    'trailing_distance': 0.02,      # %2 trailing mesafesi
    
    # Signal thresholds
    'entry_signals': {
        'strong': 3.0,              # Güçlü giriş sinyali
        'normal': 2.0,              # Normal giriş
        'weak': 1.5,                # Zayıf giriş (küçük pozisyon)
    },
    'exit_signal': -1.0,            # Çıkış sinyali
    
    # Market conditions
    'trend_filter': True,           # Trend filtresi kullan
    'volume_filter': True,          # Hacim filtresi
    'volatility_adjust': True,      # Volatiliteye göre ayarla
}

# Portfolio state
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'entry_signals': {},
}

# Load comprehensive data
def load_full_data(symbol):
    """Load all timeframes and indicators"""
    data = {}
    
    # Load 1d data with all indicators
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    
    if not raw_file.exists():
        return None
        
    # Load raw data
    raw_df = pd.read_csv(raw_file)
    raw_df['datetime'] = pd.to_datetime(raw_df['datetime'])
    raw_df.set_index('datetime', inplace=True)
    if raw_df.index.tz is not None:
        raw_df.index = raw_df.index.tz_localize(None)
    
    # Calculate additional features
    raw_df['returns'] = raw_df['close'].pct_change()
    raw_df['volatility'] = raw_df['returns'].rolling(20).std()
    raw_df['volume_ma'] = raw_df['volume'].rolling(20).mean()
    raw_df['volume_ratio'] = raw_df['volume'] / raw_df['volume_ma']
    
    # Load all indicators
    indicators = ['supertrend', 'adx_di', 'squeeze_momentum', 'wavetrend', 'macd_custom']
    
    merged_df = raw_df.copy()
    
    for indicator in indicators:
        ind_file = DATA_DIR / 'indicators' / '1d' / f"{symbol}_1d_{indicator}.csv"
        if ind_file.exists():
            ind_df = pd.read_csv(ind_file)
            ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
            ind_df.set_index('datetime', inplace=True)
            if ind_df.index.tz is not None:
                ind_df.index = ind_df.index.tz_localize(None)
            
            # Prefix columns to avoid conflicts
            for col in ind_df.columns:
                if col != 'datetime':
                    merged_df[f"{indicator}_{col}"] = ind_df[col]
    
    data['1d'] = merged_df
    
    # Load 1h for intraday confirmation
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
        daily_1h = ind_df.resample('D').agg({
            'trend': 'last',
            'buy_signal': lambda x: x.any() if len(x) > 0 else False,
            'sell_signal': lambda x: x.any() if len(x) > 0 else False
        })
        
        # Add to main dataframe
        for col in daily_1h.columns:
            merged_df[f"intraday_{col}"] = daily_1h[col]
    
    return data

# Calculate market regime
def get_market_regime(df, lookback=50):
    """Determine market regime (trending/ranging/volatile)"""
    if len(df) < lookback:
        return 'unknown'
    
    recent = df.iloc[-lookback:]
    
    # Calculate trend strength
    if 'adx_di_adx' in recent.columns:
        avg_adx = recent['adx_di_adx'].mean()
        if avg_adx > 30:
            return 'trending'
        elif avg_adx < 20:
            return 'ranging'
    
    # Check volatility
    vol = recent['volatility'].iloc[-1] if 'volatility' in recent.columns else 0
    if vol > 0.03:
        return 'volatile'
    
    return 'normal'

# Enhanced signal calculation
def calculate_enhanced_signal(row, market_regime='normal'):
    """Calculate signal with market regime adjustment"""
    signal = 0
    signal_components = {}
    
    # 1. Supertrend (primary signal)
    if row.get('supertrend_buy_signal', False):
        signal += 2.0
        signal_components['supertrend'] = 'BUY'
    elif row.get('supertrend_sell_signal', False):
        signal -= 2.0
        signal_components['supertrend'] = 'SELL'
    elif row.get('supertrend_trend', 0) == 1:
        signal += 0.5
        signal_components['supertrend'] = 'UP'
    elif row.get('supertrend_trend', 0) == -1:
        signal -= 0.5
        signal_components['supertrend'] = 'DOWN'
    
    # 2. ADX/DI (trend strength)
    if 'adx_di_adx' in row and pd.notna(row['adx_di_adx']):
        adx_val = row['adx_di_adx']
        if adx_val > 25:
            if row.get('adx_di_plus_di', 0) > row.get('adx_di_minus_di', 0):
                signal += 1.0 if market_regime == 'trending' else 0.5
                signal_components['adx'] = 'STRONG_UP'
            else:
                signal -= 1.0 if market_regime == 'trending' else 0.5
                signal_components['adx'] = 'STRONG_DOWN'
    
    # 3. Squeeze Momentum (volatility breakout)
    if row.get('squeeze_momentum_squeeze', True) == False:  # Squeeze off
        mom_val = row.get('squeeze_momentum_momentum', 0)
        if mom_val > 0:
            signal += 1.0
            signal_components['squeeze'] = 'BULLISH'
        else:
            signal -= 1.0
            signal_components['squeeze'] = 'BEARISH'
    
    # 4. WaveTrend (momentum)
    if row.get('wavetrend_buy', False):
        signal += 1.0
        signal_components['wavetrend'] = 'BUY'
    elif row.get('wavetrend_sell', False):
        signal -= 1.0
        signal_components['wavetrend'] = 'SELL'
    
    # 5. MACD (confirmation)
    if row.get('macd_custom_buy_signal', False):
        signal += 0.5
        signal_components['macd'] = 'BUY'
    elif row.get('macd_custom_sell_signal', False):
        signal -= 0.5
        signal_components['macd'] = 'SELL'
    
    # 6. Intraday confirmation
    if row.get('intraday_buy_signal', False):
        signal += 0.5
        signal_components['intraday'] = 'BUY'
    elif row.get('intraday_sell_signal', False):
        signal -= 0.5
        signal_components['intraday'] = 'SELL'
    
    # 7. Volume confirmation
    if STRATEGY_PARAMS['volume_filter'] and 'volume_ratio' in row:
        if row['volume_ratio'] > 1.5 and signal > 0:
            signal += 0.5
            signal_components['volume'] = 'HIGH'
        elif row['volume_ratio'] < 0.5 and signal < 0:
            signal -= 0.5
            signal_components['volume'] = 'LOW'
    
    return signal, signal_components

# Dynamic position sizing
def calculate_dynamic_position_size(portfolio_value, cash, volatility, signal_strength, market_regime):
    """Calculate position size based on multiple factors"""
    
    # Base size based on signal strength
    if signal_strength >= STRATEGY_PARAMS['entry_signals']['strong']:
        base_pct = STRATEGY_PARAMS['max_position_pct']
    elif signal_strength >= STRATEGY_PARAMS['entry_signals']['normal']:
        base_pct = STRATEGY_PARAMS['base_position_pct']
    else:
        base_pct = STRATEGY_PARAMS['base_position_pct'] * 0.75
    
    base_size = portfolio_value * base_pct
    
    # Volatility adjustment
    if STRATEGY_PARAMS['volatility_adjust']:
        if volatility > 0.03:  # High volatility
            vol_adj = 0.7
        elif volatility < 0.015:  # Low volatility
            vol_adj = 1.2
        else:
            vol_adj = 1.0
        base_size *= vol_adj
    
    # Market regime adjustment
    if market_regime == 'volatile':
        base_size *= 0.8
    elif market_regime == 'trending':
        base_size *= 1.1
    
    # Ensure within limits
    base_size = min(base_size, cash * 0.95)
    base_size = min(base_size, portfolio_value * STRATEGY_PARAMS['max_position_pct'])
    
    return base_size

# Smart stop loss calculation
def calculate_smart_stop_loss(entry_price, volatility, market_regime):
    """Calculate stop loss based on market conditions"""
    
    if market_regime == 'volatile' or volatility > 0.025:
        stop_pct = STRATEGY_PARAMS['stop_loss_levels']['wide']
    elif market_regime == 'ranging' or volatility < 0.015:
        stop_pct = STRATEGY_PARAMS['stop_loss_levels']['tight']
    else:
        stop_pct = STRATEGY_PARAMS['stop_loss_levels']['normal']
    
    return entry_price * (1 - stop_pct)

# Load all data
logger.info("Loading comprehensive data for all symbols...")
all_symbol_data = {}
valid_symbols = []

for symbol in SACRED_SYMBOLS:
    data = load_full_data(symbol)
    if data and '1d' in data:
        all_symbol_data[symbol] = data
        valid_symbols.append(symbol)
        logger.debug(f"Loaded {symbol} with {len(data['1d'])} days")

logger.info(f"Loaded {len(valid_symbols)} symbols successfully")

# Get trading days
all_dates = pd.DatetimeIndex([])
for symbol_data in all_symbol_data.values():
    if '1d' in symbol_data:
        all_dates = all_dates.union(symbol_data['1d'].index)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
mask = (all_dates >= start_dt) & (all_dates <= end_dt)
trading_days = all_dates[mask]

logger.info(f"Running optimized strategy from {trading_days[0]} to {trading_days[-1]} ({len(trading_days)} days)")

# Run backtest
trade_count = 0
win_count = 0
consecutive_losses = 0
max_consecutive_losses = 0

for date in trading_days:
    daily_value = portfolio['cash']
    current_prices = {}
    
    # Get current prices and calculate portfolio value
    for symbol, symbol_data in all_symbol_data.items():
        if '1d' in symbol_data and date in symbol_data['1d'].index:
            current_prices[symbol] = symbol_data['1d'].loc[date]['close']
    
    # Update position values and check exits
    positions_to_close = []
    
    for symbol, position in portfolio['positions'].items():
        if symbol not in current_prices:
            continue
        
        current_price = current_prices[symbol]
        entry_price = position['entry_price']
        position['current_value'] = position['shares'] * current_price
        daily_value += position['current_value']
        
        # Calculate profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Check trailing stop
        if profit_pct >= STRATEGY_PARAMS['trailing_activation']:
            trailing_stop = current_price * (1 - STRATEGY_PARAMS['trailing_distance'])
            if symbol not in portfolio['trailing_stops']:
                portfolio['trailing_stops'][symbol] = trailing_stop
            else:
                portfolio['trailing_stops'][symbol] = max(portfolio['trailing_stops'][symbol], trailing_stop)
        
        # Get stop price
        stop_price = portfolio['stop_losses'].get(symbol, entry_price * 0.97)
        if symbol in portfolio['trailing_stops']:
            stop_price = max(stop_price, portfolio['trailing_stops'][symbol])
        
        # Check exit conditions
        should_close = False
        close_reason = ""
        
        # 1. Stop loss hit
        if current_price <= stop_price:
            should_close = True
            close_reason = "Stop Loss"
        
        # 2. Quick profit target
        elif profit_pct >= STRATEGY_PARAMS['quick_profit'] and position.get('quick_profit_taken', False) == False:
            # Take partial profit (50%)
            shares_to_sell = position['shares'] // 2
            if shares_to_sell > 0:
                exit_value = shares_to_sell * current_price
                entry_value = shares_to_sell * entry_price
                profit = exit_value - entry_value
                profit_pct_partial = (profit / entry_value) * 100
                
                position['shares'] -= shares_to_sell
                position['quick_profit_taken'] = True
                portfolio['cash'] += exit_value
                
                portfolio['trades'].append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares_to_sell,
                    'profit': profit,
                    'profit_pct': profit_pct_partial,
                    'reason': 'Quick Profit'
                })
                
                trade_count += 1
                win_count += 1
                
                logger.debug(f"{date.date()}: QUICK PROFIT {symbol} - {shares_to_sell} shares @ ${current_price:.2f} ({profit_pct_partial:.2f}%)")
        
        # 3. Exit signal
        elif '1d' in all_symbol_data[symbol]:
            row = all_symbol_data[symbol]['1d'].loc[date]
            market_regime = get_market_regime(all_symbol_data[symbol]['1d'].loc[:date])
            signal, components = calculate_enhanced_signal(row, market_regime)
            
            if signal <= STRATEGY_PARAMS['exit_signal']:
                should_close = True
                close_reason = "Exit Signal"
        
        if should_close and position['shares'] > 0:
            positions_to_close.append((symbol, close_reason))
    
    # Close positions
    for symbol, reason in positions_to_close:
        position = portfolio['positions'][symbol]
        current_price = current_prices[symbol]
        
        exit_value = position['shares'] * current_price
        entry_value = position['shares'] * position['entry_price']
        profit = exit_value - entry_value
        profit_pct = (profit / entry_value) * 100
        
        portfolio['cash'] += exit_value
        
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
        
        del portfolio['positions'][symbol]
        if symbol in portfolio['stop_losses']:
            del portfolio['stop_losses'][symbol]
        if symbol in portfolio['trailing_stops']:
            del portfolio['trailing_stops'][symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - P&L: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check for new entries (skip if too many consecutive losses)
    if consecutive_losses >= 3:
        logger.debug(f"{date.date()}: Skipping entries due to {consecutive_losses} consecutive losses")
        continue
    
    current_portfolio_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
    portfolio_risk_pct = current_portfolio_value / daily_value if daily_value > 0 else 0
    
    if len(portfolio['positions']) < STRATEGY_PARAMS['max_positions'] and portfolio_risk_pct < STRATEGY_PARAMS['max_portfolio_risk']:
        
        # Find entry candidates
        entry_candidates = []
        
        for symbol, symbol_data in all_symbol_data.items():
            if symbol in portfolio['positions']:
                continue
            
            if '1d' not in symbol_data or date not in symbol_data['1d'].index:
                continue
            
            row = symbol_data['1d'].loc[date]
            
            # Skip if no volume
            if row.get('volume', 0) == 0:
                continue
            
            # Get market regime
            market_regime = get_market_regime(symbol_data['1d'].loc[:date])
            
            # Calculate signal
            signal, components = calculate_enhanced_signal(row, market_regime)
            
            # Check entry threshold
            if signal >= STRATEGY_PARAMS['entry_signals']['weak']:
                volatility = row.get('volatility', 0.02)
                
                entry_candidates.append({
                    'symbol': symbol,
                    'signal': signal,
                    'components': components,
                    'volatility': volatility,
                    'price': current_prices.get(symbol, 0),
                    'market_regime': market_regime
                })
        
        # Sort by signal strength
        entry_candidates.sort(key=lambda x: x['signal'], reverse=True)
        
        # Enter top positions
        positions_opened = 0
        max_new_positions = 2 if consecutive_losses == 0 else 1
        
        for candidate in entry_candidates[:max_new_positions]:
            if portfolio_risk_pct >= STRATEGY_PARAMS['max_portfolio_risk']:
                break
            
            symbol = candidate['symbol']
            current_price = candidate['price']
            
            if current_price <= 0:
                continue
            
            # Calculate position size
            position_size = calculate_dynamic_position_size(
                daily_value,
                portfolio['cash'],
                candidate['volatility'],
                candidate['signal'],
                candidate['market_regime']
            )
            
            shares = int(position_size / current_price)
            
            if shares > 0 and shares * current_price <= portfolio['cash'] * 0.95:
                # Open position
                entry_value = shares * current_price
                portfolio['cash'] -= entry_value
                
                portfolio['positions'][symbol] = {
                    'shares': shares,
                    'entry_price': current_price,
                    'entry_value': entry_value,
                    'entry_date': date,
                    'current_value': entry_value,
                    'signal_strength': candidate['signal'],
                    'market_regime': candidate['market_regime']
                }
                
                # Set smart stop loss
                stop_loss = calculate_smart_stop_loss(
                    current_price,
                    candidate['volatility'],
                    candidate['market_regime']
                )
                portfolio['stop_losses'][symbol] = stop_loss
                
                # Store entry signal components
                portfolio['entry_signals'][symbol] = candidate['components']
                
                positions_opened += 1
                logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} "
                           f"(Signal: {candidate['signal']:.1f}, Vol: {candidate['volatility']:.1%}, "
                           f"Regime: {candidate['market_regime']})")
    
    # Record daily value
    portfolio['daily_values'].append({
        'date': date,
        'value': daily_value,
        'cash': portfolio['cash'],
        'positions': len(portfolio['positions'])
    })

# Close remaining positions
final_date = trading_days[-1]
for symbol, position in list(portfolio['positions'].items()):
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

# Calculate metrics
final_value = portfolio['cash']
total_return = final_value - INITIAL_CAPITAL
total_return_pct = (total_return / INITIAL_CAPITAL) * 100
months = 12  # Full year
monthly_return = total_return_pct / months
win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

# Compound monthly return
compound_monthly = ((final_value / INITIAL_CAPITAL) ** (1/months) - 1) * 100

# Sharpe ratio
daily_df = pd.DataFrame(portfolio['daily_values'])
daily_df['returns'] = daily_df['value'].pct_change()
sharpe = daily_df['returns'].mean() / daily_df['returns'].std() * np.sqrt(252) if daily_df['returns'].std() > 0 else 0

# Max drawdown
daily_df['cummax'] = daily_df['value'].cummax()
daily_df['drawdown'] = (daily_df['cummax'] - daily_df['value']) / daily_df['cummax']
max_drawdown = daily_df['drawdown'].max() * 100

# Print results
print("\n" + "="*80)
print("OPTIMIZED MTF STRATEGY RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE} (Full Year 2024)")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return (Simple): {monthly_return:.2f}%")
print(f"Monthly Return (Compound): {compound_monthly:.2f}%")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print(f"Max Consecutive Losses: {max_consecutive_losses}")
print("="*80)

# Detailed analysis
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    
    print("\nPerformance Metrics:")
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] < 0]
    
    if len(winning_trades) > 0:
        avg_win = winning_trades['profit_pct'].mean()
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Largest Win: {winning_trades['profit_pct'].max():.2f}%")
    
    if len(losing_trades) > 0:
        avg_loss = losing_trades['profit_pct'].mean()
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Largest Loss: {losing_trades['profit_pct'].min():.2f}%")
        
        if len(winning_trades) > 0:
            expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
            print(f"Expectancy: {expectancy:.2f}%")
    
    # Profit factor
    if len(winning_trades) > 0 and len(losing_trades) > 0:
        profit_factor = abs(winning_trades['profit'].sum() / losing_trades['profit'].sum())
        print(f"Profit Factor: {profit_factor:.2f}")
    
    # Exit reasons
    print("\nExit Reasons:")
    print(trades_df['reason'].value_counts())
    
    # Monthly breakdown
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly_profits = trades_df.groupby('exit_month')['profit'].sum()
    
    print("\nMonthly P&L Breakdown:")
    cumulative = INITIAL_CAPITAL
    for month, profit in monthly_profits.items():
        monthly_pct = (profit / cumulative) * 100
        cumulative += profit
        print(f"  {month}: ${profit:,.2f} ({monthly_pct:.2f}%)")
    
    # Best performers
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': ['sum', 'count'],
        'profit_pct': 'mean'
    }).round(2)
    symbol_performance.columns = ['Total Profit', 'Trade Count', 'Avg Return %']
    symbol_performance = symbol_performance.sort_values('Total Profit', ascending=False)
    
    print("\nTop 5 Performing Symbols:")
    print(symbol_performance.head())

# Save results
results_file = DATA_DIR.parent / 'optimized_mtf_strategy_results.csv'
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(results_file, index=False)
    logger.info(f"\nDetailed results saved to: {results_file}")

# Strategy summary
print("\n" + "="*80)
print("STRATEGY OPTIMIZATION SUMMARY:")
print("="*80)
print("Key Features:")
print("✓ Multi-timeframe signal confirmation (1h + 1d)")
print("✓ Dynamic position sizing based on signal strength and volatility")
print("✓ Smart stop loss levels adjusted for market conditions")
print("✓ Partial profit taking at 2.5% with trailing stops")
print("✓ Market regime detection (trending/ranging/volatile)")
print("✓ Volume confirmation for entries")
print("✓ Risk management with consecutive loss limits")
print("✓ Portfolio risk capping at 75%")
print("\nRecommendations for 5-6% Monthly Returns:")
print("1. Use leverage carefully (1.5-2x) only on strongest signals")
print("2. Increase position sizes in trending markets")
print("3. Add more symbols from different sectors")
print("4. Consider pairs trading for market-neutral returns")
print("5. Implement options strategies for additional income")
print("="*80)