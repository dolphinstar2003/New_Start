"""
Extreme Aggressive Strategy for 5-6% Monthly Returns
Daha agresif pozisyon boyutları ve risk yönetimi
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

logger.info("Extreme Aggressive Strategy - Target: 5-6% Monthly")
logger.info("="*80)

# Test parameters
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"
INITIAL_CAPITAL = 100000

# EXTREME AGGRESSIVE PARAMETERS
EXTREME_PARAMS = {
    # Position Management - VERY AGGRESSIVE
    'base_position_pct': 0.40,      # Base %40 pozisyon
    'max_position_pct': 0.80,       # Max %80 tek pozisyon (extreme!)
    'min_position_pct': 0.20,       # Min %20
    'max_positions': 3,             # Max 3 pozisyon (concentrated)
    'max_portfolio_risk': 1.0,      # %100 invested
    
    # Entry/Exit - MORE FLEXIBLE
    'entry_threshold': -10,         # Very low threshold (take more trades)
    'strong_signal': 40,            # Strong signal threshold
    'extreme_signal': 60,           # Extreme signal (max position)
    'exit_threshold': -30,          # Exit only on very bad signals
    
    # Risk Management - WIDER STOPS
    'stop_loss': 0.05,              # %5 stop loss (wider)
    'emergency_stop': 0.08,         # %8 emergency stop
    'take_profit_levels': [         
        {'level': 0.08, 'exit_pct': 0.33},   # %8'de %33 çık
        {'level': 0.15, 'exit_pct': 0.33},   # %15'te %33 çık
        {'level': 0.25, 'exit_pct': 0.34},   # %25'te kalan çık
    ],
    'trailing_start': 0.06,         # %6'da trailing başlat
    'trailing_distance': 0.03,      # %3 trailing mesafesi
    
    # Momentum Trading
    'momentum_weight': 2.0,         # Double weight for momentum
    'use_market_regime': True,      # Trade with market direction
    'pyramid_enabled': True,        # Add to winning positions
    'pyramid_threshold': 0.05,      # Add at 5% profit
}

# Portfolio state
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'profit_targets': {},
    'entry_scores': {},
    'pyramid_count': {},
    'market_regime': 'neutral'
}

# Data cache
data_cache = {}

def load_data_with_momentum(symbol):
    """Load data with momentum indicators"""
    if symbol in data_cache:
        return data_cache[symbol]
    
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    if not raw_file.exists():
        return None
    
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Price momentum (multiple periods)
    df['returns'] = df['close'].pct_change()
    df['mom_3'] = df['close'].pct_change(3)
    df['mom_5'] = df['close'].pct_change(5)
    df['mom_10'] = df['close'].pct_change(10)
    df['mom_20'] = df['close'].pct_change(20)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_5'] = df['returns'].rolling(5).std()
    
    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Technical indicators
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['rsi_5'] = calculate_rsi(df['close'], 5)  # Faster RSI
    
    # Price position
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])
    
    # Moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
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
    
    data_cache[symbol] = df
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_market_regime(date):
    """Detect overall market regime"""
    bullish_count = 0
    bearish_count = 0
    total_momentum = 0
    
    for symbol in SACRED_SYMBOLS[:10]:  # Top 10 symbols
        df = load_data_with_momentum(symbol)
        if df is None or date not in df.index:
            continue
        
        row = df.loc[date]
        momentum = row.get('mom_10', 0)
        total_momentum += momentum
        
        if momentum > 0.03:
            bullish_count += 1
        elif momentum < -0.03:
            bearish_count += 1
    
    avg_momentum = total_momentum / 10
    
    if avg_momentum > 0.02 or bullish_count >= 6:
        return 'bullish', avg_momentum
    elif avg_momentum < -0.02 or bearish_count >= 6:
        return 'bearish', avg_momentum
    else:
        return 'neutral', avg_momentum

def calculate_extreme_score(row, market_regime='neutral', market_momentum=0):
    """Calculate extreme opportunity score"""
    score = 0
    
    # 1. Momentum (40% weight) - MOST IMPORTANT
    mom_3 = row.get('mom_3', 0) * 100
    mom_5 = row.get('mom_5', 0) * 100
    mom_10 = row.get('mom_10', 0) * 100
    mom_20 = row.get('mom_20', 0) * 100
    
    # Weighted momentum (recent more important)
    weighted_mom = (mom_3 * 0.4) + (mom_5 * 0.3) + (mom_10 * 0.2) + (mom_20 * 0.1)
    momentum_score = weighted_mom * EXTREME_PARAMS['momentum_weight']
    
    # Momentum acceleration
    if mom_3 > mom_5 > mom_10:  # Accelerating
        momentum_score *= 1.5
    
    score += momentum_score
    
    # 2. Trend (25% weight)
    if row.get('supertrend_buy_signal', False):
        score += 30
    elif row.get('supertrend_trend', 0) == 1:
        score += 15
    elif row.get('supertrend_sell_signal', False):
        score -= 30
    elif row.get('supertrend_trend', 0) == -1:
        score -= 15
    
    # 3. Price position (15% weight)
    price_pos = row.get('price_position', 0.5)
    if price_pos > 0.8:  # Near 20-day high
        score += 15 if momentum_score > 0 else -10
    elif price_pos < 0.2:  # Near 20-day low
        score += 10 if row.get('rsi', 50) < 30 else -15
    
    # 4. RSI (10% weight)
    rsi = row.get('rsi', 50)
    rsi_5 = row.get('rsi_5', 50)
    
    if rsi < 30 and rsi_5 < 25:  # Extremely oversold
        score += 20
    elif rsi < 40:
        score += 10
    elif rsi > 70 and rsi_5 > 75:  # Extremely overbought
        score += 5 if momentum_score > 10 else -20
    
    # 5. Volume (10% weight)
    vol_ratio = row.get('volume_ratio', 1)
    if vol_ratio > 2.0:
        score *= 1.2 if score > 0 else 0.8
    elif vol_ratio > 1.5:
        score *= 1.1 if score > 0 else 0.9
    
    # 6. Market regime adjustment
    if market_regime == 'bullish':
        score *= 1.2 if score > 0 else 0.8
    elif market_regime == 'bearish':
        score *= 0.8 if score > 0 else 1.2
    
    # 7. Other indicators
    if row.get('adx_di_adx', 0) > 25:
        if row.get('adx_di_plus_di', 0) > row.get('adx_di_minus_di', 0):
            score += 10
        else:
            score -= 10
    
    return score

def scan_extreme_opportunities(date, market_regime, market_momentum):
    """Scan for extreme profit opportunities"""
    opportunities = []
    
    for symbol in SACRED_SYMBOLS:
        df = load_data_with_momentum(symbol)
        if df is None or date not in df.index:
            continue
        
        row = df.loc[date]
        
        # Skip if no volume
        if row.get('volume', 0) == 0:
            continue
        
        # Calculate score
        score = calculate_extreme_score(row, market_regime, market_momentum)
        
        # Get data
        price = row['close']
        volatility = row.get('volatility', 0.02)
        
        in_position = symbol in portfolio['positions']
        
        opportunities.append({
            'symbol': symbol,
            'score': score,
            'price': price,
            'volatility': volatility,
            'momentum_3': row.get('mom_3', 0) * 100,
            'momentum_10': row.get('mom_10', 0) * 100,
            'rsi': row.get('rsi', 50),
            'volume_ratio': row.get('volume_ratio', 1),
            'in_position': in_position
        })
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities

def calculate_extreme_position_size(cash, portfolio_value, score, volatility):
    """Calculate extreme position size"""
    
    # Base on score
    if score >= EXTREME_PARAMS['extreme_signal']:
        base_pct = EXTREME_PARAMS['max_position_pct']
    elif score >= EXTREME_PARAMS['strong_signal']:
        base_pct = 0.60
    elif score >= 20:
        base_pct = EXTREME_PARAMS['base_position_pct']
    elif score >= 0:
        base_pct = 0.30
    else:
        base_pct = EXTREME_PARAMS['min_position_pct']
    
    # Less volatility adjustment (we want big positions)
    if volatility > 0.04:
        vol_adj = 0.9
    else:
        vol_adj = 1.0
    
    position_size = portfolio_value * base_pct * vol_adj
    
    # Ensure we don't exceed cash
    position_size = min(position_size, cash * 0.98)
    
    return position_size

def should_pyramid(symbol, position, current_price, current_score):
    """Check if should add to position"""
    if not EXTREME_PARAMS['pyramid_enabled']:
        return False
    
    if symbol not in portfolio['pyramid_count']:
        portfolio['pyramid_count'][symbol] = 0
    
    # Max 2 pyramids
    if portfolio['pyramid_count'][symbol] >= 2:
        return False
    
    # Check profit
    profit_pct = (current_price - position['entry_price']) / position['entry_price']
    
    # Pyramid if profitable and score still good
    if profit_pct >= EXTREME_PARAMS['pyramid_threshold'] and current_score > 20:
        return True
    
    return False

# Load all data
logger.info("Loading data for extreme strategy...")
for symbol in SACRED_SYMBOLS:
    df = load_data_with_momentum(symbol)
    if df is not None:
        logger.debug(f"Loaded {symbol}: {len(df)} days")

# Get trading days
all_dates = pd.DatetimeIndex([])
for symbol in SACRED_SYMBOLS:
    df = load_data_with_momentum(symbol)
    if df is not None:
        all_dates = all_dates.union(df.index)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
mask = (all_dates >= start_dt) & (all_dates <= end_dt)
trading_days = all_dates[mask]

logger.info(f"Running extreme strategy from {trading_days[0]} to {trading_days[-1]}")

# Run backtest
trade_count = 0
win_count = 0
pyramid_trades = 0

for i, date in enumerate(trading_days):
    daily_value = portfolio['cash']
    
    # Detect market regime
    market_regime, market_momentum = detect_market_regime(date)
    portfolio['market_regime'] = market_regime
    
    if i % 20 == 0:  # Log every 20 days
        logger.info(f"{date.date()}: Market regime: {market_regime} (momentum: {market_momentum:.2%})")
    
    # Scan opportunities
    opportunities = scan_extreme_opportunities(date, market_regime, market_momentum)
    
    # Get current prices
    current_prices = {opp['symbol']: opp['price'] for opp in opportunities}
    
    # Update positions and check exits
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
        
        # Update trailing stop
        if profit_pct >= EXTREME_PARAMS['trailing_start']:
            trailing_stop = current_price * (1 - EXTREME_PARAMS['trailing_distance'])
            if symbol not in portfolio['trailing_stops']:
                portfolio['trailing_stops'][symbol] = trailing_stop
            else:
                portfolio['trailing_stops'][symbol] = max(portfolio['trailing_stops'][symbol], trailing_stop)
        
        # Get stop price
        if profit_pct < -EXTREME_PARAMS['stop_loss']:
            stop_price = entry_price * (1 - EXTREME_PARAMS['emergency_stop'])
        else:
            stop_price = entry_price * (1 - EXTREME_PARAMS['stop_loss'])
        
        if symbol in portfolio['trailing_stops']:
            stop_price = max(stop_price, portfolio['trailing_stops'][symbol])
        
        # Check exits
        should_close = False
        close_reason = ""
        remaining_shares = position['shares']
        
        # 1. Take profit levels
        if symbol not in portfolio['profit_targets']:
            portfolio['profit_targets'][symbol] = 0
        
        for j, tp_level in enumerate(EXTREME_PARAMS['take_profit_levels']):
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
        
        # 3. Exit signal (only on very bad signals)
        elif remaining_shares > 0:
            current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
            if current_opp and current_opp['score'] < EXTREME_PARAMS['exit_threshold']:
                should_close = True
                close_reason = "Exit Signal"
        
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
                             portfolio['profit_targets'], portfolio['entry_scores'],
                             portfolio['pyramid_count']]:
            if symbol in tracking_dict:
                del tracking_dict[symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - P&L: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Check for pyramiding opportunities
    for symbol, position in list(portfolio['positions'].items()):
        current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
        if current_opp and should_pyramid(symbol, position, current_opp['price'], current_opp['score']):
            # Add to position
            additional_size = calculate_extreme_position_size(
                portfolio['cash'],
                daily_value,
                current_opp['score'],
                current_opp['volatility']
            ) * 0.5  # Half size for pyramid
            
            additional_shares = int(additional_size / current_opp['price'])
            
            if additional_shares > 0 and additional_shares * current_opp['price'] <= portfolio['cash'] * 0.95:
                entry_value = additional_shares * current_opp['price']
                portfolio['cash'] -= entry_value
                
                # Update position (average price)
                total_shares = position['shares'] + additional_shares
                total_value = (position['shares'] * position['entry_price']) + entry_value
                position['entry_price'] = total_value / total_shares
                position['shares'] = total_shares
                position['original_shares'] = total_shares
                
                portfolio['pyramid_count'][symbol] += 1
                pyramid_trades += 1
                
                logger.debug(f"{date.date()}: PYRAMID {symbol} - {additional_shares} @ ${current_opp['price']:.2f} "
                           f"(Score: {current_opp['score']:.1f})")
    
    # Open new positions
    if len(portfolio['positions']) < EXTREME_PARAMS['max_positions']:
        positions_opened = 0
        
        for opp in opportunities:
            if positions_opened >= 1:  # Max 1 new position per day
                break
            
            if len(portfolio['positions']) >= EXTREME_PARAMS['max_positions']:
                break
            
            if opp['in_position'] or opp['score'] < EXTREME_PARAMS['entry_threshold']:
                continue
            
            symbol = opp['symbol']
            current_price = opp['price']
            
            # Calculate position size
            position_size = calculate_extreme_position_size(
                portfolio['cash'],
                daily_value,
                opp['score'],
                opp['volatility']
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
                
                portfolio['stop_losses'][symbol] = current_price * (1 - EXTREME_PARAMS['stop_loss'])
                portfolio['entry_scores'][symbol] = opp['score']
                portfolio['pyramid_count'][symbol] = 0
                
                positions_opened += 1
                
                logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} "
                           f"(Score: {opp['score']:.1f}, Mom3: {opp['momentum_3']:.1f}%, "
                           f"RSI: {opp['rsi']:.0f}, Size: {position_size/daily_value:.1%})")
    
    # Record daily value
    portfolio['daily_values'].append({
        'date': date,
        'value': daily_value,
        'cash': portfolio['cash'],
        'positions': len(portfolio['positions']),
        'regime': market_regime
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
print("EXTREME AGGRESSIVE STRATEGY RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return (Simple): {monthly_return:.2f}%")
print(f"Monthly Return (Compound): {compound_monthly:.2f}%")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Pyramid Trades: {pyramid_trades}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("="*80)

# Performance analysis
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
    
    # Market regime analysis
    regime_df = daily_df.groupby('regime')['returns'].agg(['mean', 'count'])
    print("\nPerformance by Market Regime:")
    print(regime_df)
    
    # Top performers
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': 'sum',
        'profit_pct': 'mean'
    }).round(2)
    
    print("\nTop 5 Performers:")
    print(symbol_performance.nlargest(5, 'profit'))

# Save results
results_file = DATA_DIR.parent / 'extreme_aggressive_results.csv'
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")

print("\n" + "="*80)
print("EXTREME STRATEGY FEATURES:")
print("✓ Concentrated portfolio (max 3 positions)")
print("✓ Position sizes up to 80% for extreme signals")
print("✓ Focus on momentum (2x weight)")
print("✓ Market regime detection")
print("✓ Pyramid into winning positions")
print("✓ Wider stop losses (5% normal, 8% emergency)")
print("✓ Take profits at 8%, 15%, 25%")
print("✓ Trade with market direction")
print("="*80)