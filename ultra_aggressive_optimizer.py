"""
Ultra Aggressive Portfolio Optimizer
Williams VIX Fix ile piyasa dip tespiti
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
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR

logger.info("Ultra Aggressive Portfolio Optimizer with VIX Fix")
logger.info("="*80)

# Test parameters
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"
INITIAL_CAPITAL = 100000

# Ultra agresif parametreler
ULTRA_PARAMS = {
    # Pozisyon yönetimi
    'base_position_pct': 0.30,      # Base %30 pozisyon
    'max_position_pct': 0.50,       # Max %50 (VIX sinyalinde)
    'min_position_pct': 0.10,       # Min %10
    'max_positions': 10,            # Max 10 pozisyon
    'max_portfolio_risk': 1.0,      # %100 yatırım yapılabilir
    
    # Giriş/çıkış
    'entry_threshold': 0.5,         # Çok düşük giriş eşiği
    'vix_buy_threshold': 0.85,      # VIX Fix alım sinyali
    'momentum_threshold': 2.0,      # Momentum için eşik
    'exit_threshold': -1.0,         # Çıkış eşiği
    
    # Risk/Kar yönetimi
    'stop_loss': 0.025,             # %2.5 sıkı stop
    'take_profit_levels': [         # Kademeli kar al
        {'level': 0.04, 'exit_pct': 0.25},   # %4'te %25 çık
        {'level': 0.06, 'exit_pct': 0.25},   # %6'da %25 çık
        {'level': 0.10, 'exit_pct': 0.50},   # %10'da kalan çık
    ],
    'trailing_start': 0.03,         # %3'te trailing başlat
    'trailing_distance': 0.015,     # %1.5 trailing mesafesi
    
    # Rotasyon
    'rotation_freq': 1,             # Her gün kontrol
    'min_score_diff': 2.0,          # Min skor farkı rotasyon için
    'force_rotation_days': 7,       # 7 günde zorla rotasyon
}

# Enhanced portfolio state
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'profit_targets': {},
    'entry_scores': {},
    'holding_days': {},
    'peak_values': {}
}

# Cache for symbol data
symbol_cache = {}

def calculate_williams_vix_fix(df, lookback=22):
    """Calculate Williams VIX Fix indicator"""
    # WVF = ((Highest(Close, lookback) - Low) / Highest(Close, lookback)) * 100
    highest_close = df['close'].rolling(lookback).max()
    wvf = ((highest_close - df['low']) / highest_close) * 100
    
    # Bollinger Bands on WVF
    bb_length = 20
    bb_mult = 2.0
    wvf_sma = wvf.rolling(bb_length).mean()
    wvf_std = wvf.rolling(bb_length).std()
    upper_band = wvf_sma + (bb_mult * wvf_std)
    
    # Percentile based levels
    percentile_lookback = 50
    range_high = wvf.rolling(percentile_lookback).quantile(0.85)
    
    # Buy signal when WVF exceeds upper band or range high
    buy_signal = (wvf >= upper_band) | (wvf >= range_high)
    
    return wvf, buy_signal, upper_band

def load_enhanced_data(symbol):
    """Load data with all indicators and VIX Fix"""
    if symbol in symbol_cache:
        return symbol_cache[symbol]
    
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    if not raw_file.exists():
        return None
    
    # Load raw data
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Basic calculations
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    
    # Williams VIX Fix
    df['vix_fix'], df['vix_buy_signal'], df['vix_upper'] = calculate_williams_vix_fix(df)
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    
    # Load indicators
    indicators = ['supertrend', 'adx_di', 'squeeze_momentum', 'wavetrend', 'macd_custom']
    
    for indicator in indicators:
        ind_file = DATA_DIR / 'indicators' / '1d' / f"{symbol}_1d_{indicator}.csv"
        if ind_file.exists():
            ind_df = pd.read_csv(ind_file)
            ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
            ind_df.set_index('datetime', inplace=True)
            if ind_df.index.tz is not None:
                ind_df.index = ind_df.index.tz_localize(None)
            
            for col in ind_df.columns:
                df[f"{indicator}_{col}"] = ind_df[col]
    
    symbol_cache[symbol] = df
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ultra_score(row, vix_active=False):
    """Calculate ultra aggressive opportunity score"""
    score = 0
    
    # 1. VIX Fix signal (highest priority)
    if vix_active or row.get('vix_buy_signal', False):
        score += 50  # Major boost for market bottom
    
    # 2. Trend signals
    if row.get('supertrend_buy_signal', False):
        score += 30
    elif row.get('supertrend_trend', 0) == 1:
        score += 15
    elif row.get('supertrend_sell_signal', False):
        score -= 30
    elif row.get('supertrend_trend', 0) == -1:
        score -= 15
    
    # 3. Momentum (multi-timeframe)
    mom_5 = row.get('momentum_5', 0) * 100
    mom_10 = row.get('momentum_10', 0) * 100
    mom_20 = row.get('momentum_20', 0) * 100
    
    # Weight recent momentum more
    weighted_momentum = (mom_5 * 0.5) + (mom_10 * 0.3) + (mom_20 * 0.2)
    score += weighted_momentum * 2
    
    # 4. RSI extremes
    rsi = row.get('rsi', 50)
    if rsi < 30:  # Oversold
        score += 20
    elif rsi < 40:
        score += 10
    elif rsi > 70:  # Overbought
        score -= 10
    elif rsi > 80:
        score -= 20
    
    # 5. Volume confirmation
    vol_ratio = row.get('volume_ratio', 1)
    if vol_ratio > 2.0 and score > 0:
        score *= 1.3  # 30% boost for high volume
    elif vol_ratio > 1.5 and score > 0:
        score *= 1.15
    
    # 6. Price vs moving average
    price_vs_ma = row.get('price_vs_sma20', 0)
    if price_vs_ma < -0.05:  # 5% below MA20
        score += 15
    elif price_vs_ma < -0.10:  # 10% below MA20
        score += 25
    
    # 7. ADX strength
    if 'adx_di_adx' in row and pd.notna(row['adx_di_adx']):
        if row['adx_di_adx'] > 30:  # Very strong trend
            if row.get('adx_di_plus_di', 0) > row.get('adx_di_minus_di', 0):
                score += 20
            else:
                score -= 20
    
    # 8. Other indicators
    if row.get('squeeze_momentum_squeeze', True) == False:
        if row.get('squeeze_momentum_momentum', 0) > 0:
            score += 10
    
    if row.get('wavetrend_buy', False):
        score += 10
    elif row.get('wavetrend_sell', False):
        score -= 10
    
    return score

def scan_all_opportunities(date):
    """Scan all symbols and rank by opportunity"""
    opportunities = []
    
    # Check for market-wide VIX signal
    vix_signals = 0
    total_symbols = 0
    
    for symbol in SACRED_SYMBOLS:
        df = load_enhanced_data(symbol)
        if df is None or date not in df.index:
            continue
        
        row = df.loc[date]
        if row.get('vix_buy_signal', False):
            vix_signals += 1
        total_symbols += 1
    
    # Market-wide VIX active if >30% of symbols show signal
    market_vix_active = (vix_signals / total_symbols > 0.3) if total_symbols > 0 else False
    
    # Evaluate each symbol
    for symbol in SACRED_SYMBOLS:
        df = load_enhanced_data(symbol)
        if df is None or date not in df.index:
            continue
        
        row = df.loc[date]
        
        # Skip if no volume
        if row.get('volume', 0) == 0:
            continue
        
        # Calculate score
        score = calculate_ultra_score(row, market_vix_active)
        
        # Get price and volatility
        price = row['close']
        volatility = row.get('volatility', 0.02)
        
        # Check if in position
        in_position = symbol in portfolio['positions']
        
        opportunities.append({
            'symbol': symbol,
            'score': score,
            'price': price,
            'volatility': volatility,
            'volume_ratio': row.get('volume_ratio', 1),
            'rsi': row.get('rsi', 50),
            'momentum': row.get('momentum_5', 0) * 100,
            'vix_signal': row.get('vix_buy_signal', False),
            'in_position': in_position
        })
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities, market_vix_active

def calculate_ultra_position_size(cash, score, volatility, vix_active=False):
    """Calculate position size for ultra aggressive strategy"""
    
    # Base on score
    if score >= 80 or vix_active:
        base_pct = ULTRA_PARAMS['max_position_pct']
    elif score >= 60:
        base_pct = 0.40
    elif score >= 40:
        base_pct = ULTRA_PARAMS['base_position_pct']
    elif score >= 20:
        base_pct = 0.20
    else:
        base_pct = ULTRA_PARAMS['min_position_pct']
    
    # Volatility adjustment (less adjustment for aggressive strategy)
    if volatility > 0.04:
        vol_adj = 0.85
    elif volatility < 0.02:
        vol_adj = 1.1
    else:
        vol_adj = 1.0
    
    position_size = cash * base_pct * vol_adj
    
    # Ensure we don't exceed available cash
    position_size = min(position_size, cash * 0.98)
    
    return position_size

def should_force_rotation(symbol, holding_days, current_score, position_profit):
    """Check if should force rotation"""
    # Force rotation if held too long without good profit
    if holding_days >= ULTRA_PARAMS['force_rotation_days']:
        if position_profit < 0.03:  # Less than 3% after 7 days
            return True
    
    # Force rotation if score becomes very negative
    if current_score < -30:
        return True
    
    return False

# Load all data
logger.info("Loading data for all symbols...")
for symbol in SACRED_SYMBOLS:
    df = load_enhanced_data(symbol)
    if df is not None:
        logger.debug(f"Loaded {symbol}: {len(df)} days")

# Get trading days
all_dates = pd.DatetimeIndex([])
for symbol in SACRED_SYMBOLS:
    df = load_enhanced_data(symbol)
    if df is not None:
        all_dates = all_dates.union(df.index)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
mask = (all_dates >= start_dt) & (all_dates <= end_dt)
trading_days = all_dates[mask]

logger.info(f"Running ultra aggressive strategy from {trading_days[0]} to {trading_days[-1]}")

# Run backtest
trade_count = 0
win_count = 0
rotation_count = 0
vix_trades = 0

for i, date in enumerate(trading_days):
    daily_value = portfolio['cash']
    
    # Scan all opportunities
    opportunities, market_vix_active = scan_all_opportunities(date)
    
    if market_vix_active:
        logger.info(f"{date.date()}: MARKET VIX SIGNAL ACTIVE!")
    
    # Get current prices
    current_prices = {opp['symbol']: opp['price'] for opp in opportunities}
    
    # Update position values and check exits
    positions_to_close = []
    
    for symbol, position in portfolio['positions'].items():
        if symbol not in current_prices:
            continue
        
        current_price = current_prices[symbol]
        entry_price = position['entry_price']
        position['current_value'] = position['shares'] * current_price
        daily_value += position['current_value']
        
        # Update holding days
        if symbol not in portfolio['holding_days']:
            portfolio['holding_days'][symbol] = 0
        portfolio['holding_days'][symbol] += 1
        
        # Calculate profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Track peak value
        if symbol not in portfolio['peak_values']:
            portfolio['peak_values'][symbol] = current_price
        else:
            portfolio['peak_values'][symbol] = max(portfolio['peak_values'][symbol], current_price)
        
        # Update trailing stop
        if profit_pct >= ULTRA_PARAMS['trailing_start']:
            trailing_stop = current_price * (1 - ULTRA_PARAMS['trailing_distance'])
            if symbol not in portfolio['trailing_stops']:
                portfolio['trailing_stops'][symbol] = trailing_stop
            else:
                portfolio['trailing_stops'][symbol] = max(portfolio['trailing_stops'][symbol], trailing_stop)
        
        # Get stop price
        stop_price = portfolio['stop_losses'].get(symbol, entry_price * (1 - ULTRA_PARAMS['stop_loss']))
        if symbol in portfolio['trailing_stops']:
            stop_price = max(stop_price, portfolio['trailing_stops'][symbol])
        
        # Check exit conditions
        should_close = False
        close_reason = ""
        remaining_shares = position['shares']
        
        # 1. Take profit levels
        if symbol not in portfolio['profit_targets']:
            portfolio['profit_targets'][symbol] = 0
        
        for j, tp_level in enumerate(ULTRA_PARAMS['take_profit_levels']):
            if j <= portfolio['profit_targets'][symbol]:
                continue
                
            if profit_pct >= tp_level['level']:
                # Partial exit
                shares_to_sell = int(position['original_shares'] * tp_level['exit_pct'])
                if shares_to_sell > 0 and shares_to_sell <= remaining_shares:
                    exit_value = shares_to_sell * current_price
                    entry_value = shares_to_sell * entry_price
                    profit = exit_value - entry_value
                    profit_pct_partial = (profit / entry_value) * 100
                    
                    position['shares'] -= shares_to_sell
                    remaining_shares -= shares_to_sell
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
                        'reason': f'TP Level {j+1}'
                    })
                    
                    portfolio['profit_targets'][symbol] = j
                    trade_count += 1
                    win_count += 1
                    
                    logger.debug(f"{date.date()}: PARTIAL EXIT {symbol} - Level {j+1} - "
                               f"{shares_to_sell} shares @ ${current_price:.2f} ({profit_pct_partial:.2f}%)")
        
        # 2. Stop loss
        if remaining_shares > 0 and current_price <= stop_price:
            should_close = True
            close_reason = "Stop Loss"
        
        # 3. Force rotation check
        elif remaining_shares > 0:
            # Find current score
            current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
            if current_opp:
                current_score = current_opp['score']
                holding_days = portfolio['holding_days'].get(symbol, 0)
                
                if should_force_rotation(symbol, holding_days, current_score, profit_pct):
                    should_close = True
                    close_reason = "Force Rotation"
        
        if should_close and remaining_shares > 0:
            position['shares'] = remaining_shares
            positions_to_close.append((symbol, close_reason))
    
    # Close positions
    for symbol, reason in positions_to_close:
        position = portfolio['positions'][symbol]
        if position['shares'] <= 0:
            continue
            
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
        for tracking_dict in [portfolio['stop_losses'], portfolio['trailing_stops'], 
                             portfolio['profit_targets'], portfolio['holding_days'], 
                             portfolio['peak_values'], portfolio['entry_scores']]:
            if symbol in tracking_dict:
                del tracking_dict[symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        if reason == "Force Rotation":
            rotation_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - P&L: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check for rotations (best opportunities not in portfolio)
    if i % ULTRA_PARAMS['rotation_freq'] == 0:
        for new_opp in opportunities[:10]:  # Top 10 opportunities
            if new_opp['in_position'] or new_opp['score'] < ULTRA_PARAMS['entry_threshold']:
                continue
            
            # Check if should replace any position
            for symbol, position in list(portfolio['positions'].items()):
                if len(portfolio['positions']) >= ULTRA_PARAMS['max_positions']:
                    current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
                    if current_opp and new_opp['score'] - current_opp['score'] > ULTRA_PARAMS['min_score_diff']:
                        # Execute rotation
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
                            'reason': 'Rotation'
                        })
                        
                        del portfolio['positions'][symbol]
                        trade_count += 1
                        if profit > 0:
                            win_count += 1
                        rotation_count += 1
                        
                        logger.debug(f"{date.date()}: ROTATE {symbol} -> {new_opp['symbol']} "
                                   f"(Score: {current_opp['score']:.1f} -> {new_opp['score']:.1f})")
                        break
    
    # Open new positions
    positions_opened = 0
    max_new = 5 if market_vix_active else 3
    
    for opp in opportunities:
        if positions_opened >= max_new:
            break
        
        if len(portfolio['positions']) >= ULTRA_PARAMS['max_positions']:
            break
        
        if opp['in_position'] or opp['score'] < ULTRA_PARAMS['entry_threshold']:
            continue
        
        symbol = opp['symbol']
        current_price = opp['price']
        
        # Calculate position size
        position_size = calculate_ultra_position_size(
            portfolio['cash'],
            opp['score'],
            opp['volatility'],
            market_vix_active
        )
        
        shares = int(position_size / current_price)
        
        if shares > 0 and shares * current_price <= portfolio['cash'] * 0.98:
            # Open position
            entry_value = shares * current_price
            portfolio['cash'] -= entry_value
            
            portfolio['positions'][symbol] = {
                'shares': shares,
                'original_shares': shares,
                'entry_price': current_price,
                'entry_value': entry_value,
                'entry_date': date,
                'current_value': entry_value
            }
            
            portfolio['stop_losses'][symbol] = current_price * (1 - ULTRA_PARAMS['stop_loss'])
            portfolio['entry_scores'][symbol] = opp['score']
            portfolio['holding_days'][symbol] = 0
            
            positions_opened += 1
            if opp['vix_signal'] or market_vix_active:
                vix_trades += 1
            
            logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} "
                       f"(Score: {opp['score']:.1f}, RSI: {opp['rsi']:.0f}, "
                       f"Mom: {opp['momentum']:.1f}%, VIX: {'YES' if opp['vix_signal'] else 'NO'})")
    
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

# Calculate metrics
final_value = portfolio['cash']
total_return = final_value - INITIAL_CAPITAL
total_return_pct = (total_return / INITIAL_CAPITAL) * 100
monthly_return = total_return_pct / 6
compound_monthly = ((final_value / INITIAL_CAPITAL) ** (1/6) - 1) * 100
win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

# Advanced metrics
daily_df = pd.DataFrame(portfolio['daily_values'])
daily_df['returns'] = daily_df['value'].pct_change()
sharpe = daily_df['returns'].mean() / daily_df['returns'].std() * np.sqrt(252) if daily_df['returns'].std() > 0 else 0

daily_df['cummax'] = daily_df['value'].cummax()
daily_df['drawdown'] = (daily_df['cummax'] - daily_df['value']) / daily_df['cummax']
max_drawdown = daily_df['drawdown'].max() * 100

# Print results
print("\n" + "="*80)
print("ULTRA AGGRESSIVE PORTFOLIO OPTIMIZER RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return (Simple): {monthly_return:.2f}%")
print(f"Monthly Return (Compound): {compound_monthly:.2f}%")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Rotations: {rotation_count}")
print(f"VIX-based Trades: {vix_trades}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("="*80)

# Detailed analysis
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    
    print("\nPerformance Metrics:")
    winning_trades = trades_df[trades_df['profit'] > 0]
    losing_trades = trades_df[trades_df['profit'] < 0]
    
    if len(winning_trades) > 0:
        print(f"Average Win: {winning_trades['profit_pct'].mean():.2f}%")
        print(f"Largest Win: {winning_trades['profit_pct'].max():.2f}%")
    
    if len(losing_trades) > 0:
        print(f"Average Loss: {losing_trades['profit_pct'].mean():.2f}%")
        print(f"Largest Loss: {losing_trades['profit_pct'].min():.2f}%")
    
    # Exit reasons
    print("\nExit Reasons:")
    print(trades_df['reason'].value_counts())
    
    # Monthly breakdown
    trades_df['exit_month'] = pd.to_datetime(trades_df['exit_date']).dt.to_period('M')
    monthly_profits = trades_df.groupby('exit_month')['profit'].sum()
    
    print("\nMonthly P&L:")
    cumulative = INITIAL_CAPITAL
    for month, profit in monthly_profits.items():
        monthly_pct = (profit / cumulative) * 100
        cumulative += profit
        print(f"  {month}: ${profit:,.2f} ({monthly_pct:.2f}%)")
    
    # Top performers
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': 'sum',
        'profit_pct': 'mean'
    }).round(2)
    
    print("\nTop 5 Performers:")
    print(symbol_performance.nlargest(5, 'profit'))

# Save results
results_file = DATA_DIR.parent / 'ultra_aggressive_results.csv'
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")

print("\n" + "="*80)
print("ULTRA AGGRESSIVE FEATURES:")
print("✓ Williams VIX Fix for market bottom detection")
print("✓ Position sizes up to 50% with VIX signals")
print("✓ Multi-level momentum analysis (5/10/20 days)")
print("✓ Aggressive rotation every 7 days")
print("✓ Partial profit taking at 4%, 6%, 10%")
print("✓ Tight 2.5% stop loss with 1.5% trailing")
print("✓ RSI and volume-based signal enhancement")
print("✓ Market-wide VIX monitoring")
print("="*80)