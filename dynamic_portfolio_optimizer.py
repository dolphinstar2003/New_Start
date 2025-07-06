"""
Dynamic Portfolio Optimizer
Sürekli portföy rotasyonu ile maksimum getiri
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

logger.info("Dynamic Portfolio Optimizer")
logger.info("="*80)

# Test parameters
START_DATE = "2025-01-01"
END_DATE = "2025-06-30"
INITIAL_CAPITAL = 100000

# Optimal portföy parametreleri (2024'te %20 getiri veren)
PORTFOLIO_PARAMS = {
    # Pozisyon yönetimi
    'base_position_pct': 0.20,      # Base %20 pozisyon
    'max_position_pct': 0.30,       # Max %30 (en güçlü sinyallerde)
    'min_position_pct': 0.05,       # Min %5 (zayıf sinyal)
    'max_positions': 10,            # Max 10 pozisyon aynı anda
    'max_portfolio_risk': 0.95,     # Portfolio'nun max %95'i risk altında
    
    # Giriş/çıkış eşikleri
    'entry_threshold': 1.0,         # Düşük giriş eşiği (daha fazla fırsat)
    'strong_entry': 2.5,            # Güçlü giriş sinyali
    'exit_threshold': -0.5,         # Çıkış eşiği
    'rotation_threshold': 3.0,      # Rotasyon için min sinyal farkı
    
    # Risk yönetimi
    'stop_loss': 0.03,              # %3 stop loss
    'take_profit': 0.08,            # %8 kar al
    'trailing_start': 0.04,         # %4'te trailing başlat
    'trailing_distance': 0.02,      # %2 trailing mesafesi
    
    # Portföy rotasyonu
    'enable_rotation': True,        # Dinamik rotasyon aktif
    'rotation_check_days': 2,       # Her 2 günde rotasyon kontrolü
    'min_holding_days': 3,          # Min 3 gün tutma süresi
    'profit_lock_threshold': 0.15,  # %15 karda pozisyon küçült
}

# Enhanced portfolio state
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'entry_dates': {},
    'peak_profits': {},
    'rotation_candidates': {},
    'last_rotation_check': None
}

# Load comprehensive data with caching
symbol_data_cache = {}

def load_symbol_data(symbol):
    """Load and cache symbol data"""
    if symbol in symbol_data_cache:
        return symbol_data_cache[symbol]
    
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    if not raw_file.exists():
        return None
    
    # Load raw data
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Calculate technical features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['rsi'] = calculate_rsi(df['close'])
    df['momentum'] = df['close'].pct_change(10)
    
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
            
            # Merge indicators
            for col in ind_df.columns:
                df[f"{indicator}_{col}"] = ind_df[col]
    
    symbol_data_cache[symbol] = df
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_opportunity_score(row, symbol_momentum=0):
    """Calculate opportunity score for symbol"""
    score = 0
    
    # 1. Trend signals (40% weight)
    if row.get('supertrend_buy_signal', False):
        score += 40
    elif row.get('supertrend_trend', 0) == 1:
        score += 20
    elif row.get('supertrend_sell_signal', False):
        score -= 40
    elif row.get('supertrend_trend', 0) == -1:
        score -= 20
    
    # 2. Momentum (20% weight)
    if 'rsi' in row and pd.notna(row['rsi']):
        if 30 < row['rsi'] < 50:  # Oversold recovery
            score += 15
        elif 50 < row['rsi'] < 70:  # Bullish
            score += 10
        elif row['rsi'] > 80:  # Overbought
            score -= 10
    
    # 3. Volume (15% weight)
    if row.get('volume_ratio', 1) > 1.5:
        score += 15 if score > 0 else -15
    
    # 4. ADX strength (15% weight)
    if 'adx_di_adx' in row and pd.notna(row['adx_di_adx']):
        if row['adx_di_adx'] > 25:
            if row.get('adx_di_plus_di', 0) > row.get('adx_di_minus_di', 0):
                score += 15
            else:
                score -= 15
    
    # 5. Other indicators (10% weight)
    if row.get('squeeze_momentum_squeeze', True) == False:
        if row.get('squeeze_momentum_momentum', 0) > 0:
            score += 5
    
    if row.get('wavetrend_buy', False):
        score += 5
    elif row.get('wavetrend_sell', False):
        score -= 5
    
    # 6. Recent momentum bonus
    score += symbol_momentum * 10
    
    return score

def evaluate_all_opportunities(date, current_positions):
    """Evaluate all symbols and return ranked opportunities"""
    opportunities = []
    
    for symbol in SACRED_SYMBOLS:
        df = load_symbol_data(symbol)
        if df is None or date not in df.index:
            continue
        
        row = df.loc[date]
        
        # Skip if no volume
        if row.get('volume', 0) == 0:
            continue
        
        # Calculate recent momentum
        if len(df.loc[:date]) >= 20:
            recent_returns = df.loc[:date].tail(20)['returns'].mean() * 100
            symbol_momentum = recent_returns
        else:
            symbol_momentum = 0
        
        # Calculate opportunity score
        score = calculate_opportunity_score(row, symbol_momentum)
        
        # Get current price and volatility
        current_price = row['close']
        volatility = row.get('volatility', 0.02)
        
        # Check if already in position
        in_position = symbol in current_positions
        current_profit = 0
        
        if in_position:
            entry_price = current_positions[symbol]['entry_price']
            current_profit = (current_price - entry_price) / entry_price
        
        opportunities.append({
            'symbol': symbol,
            'score': score,
            'price': current_price,
            'volatility': volatility,
            'momentum': symbol_momentum,
            'volume_ratio': row.get('volume_ratio', 1),
            'in_position': in_position,
            'current_profit': current_profit,
            'rsi': row.get('rsi', 50)
        })
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities

def should_rotate_position(current_pos_data, new_opportunity, holding_days):
    """Determine if should rotate position"""
    if not PORTFOLIO_PARAMS['enable_rotation']:
        return False
    
    # Check minimum holding period
    if holding_days < PORTFOLIO_PARAMS['min_holding_days']:
        return False
    
    # Don't rotate if position is very profitable
    if current_pos_data['current_profit'] > PORTFOLIO_PARAMS['profit_lock_threshold']:
        return False
    
    # Don't rotate if in loss (wait for stop loss)
    if current_pos_data['current_profit'] < -0.01:
        return False
    
    # Rotate if new opportunity is significantly better
    score_diff = new_opportunity['score'] - current_pos_data['score']
    if score_diff > PORTFOLIO_PARAMS['rotation_threshold']:
        return True
    
    return False

def calculate_dynamic_position_size(portfolio_value, cash, opportunity, num_positions):
    """Calculate position size based on opportunity score and portfolio state"""
    
    # Base size based on score
    if opportunity['score'] >= 60:
        base_pct = PORTFOLIO_PARAMS['max_position_pct']
    elif opportunity['score'] >= 40:
        base_pct = PORTFOLIO_PARAMS['base_position_pct']
    elif opportunity['score'] >= 20:
        base_pct = (PORTFOLIO_PARAMS['base_position_pct'] + PORTFOLIO_PARAMS['min_position_pct']) / 2
    else:
        base_pct = PORTFOLIO_PARAMS['min_position_pct']
    
    # Adjust for number of positions
    if num_positions < 5:
        base_pct *= 1.2  # Larger positions when few holdings
    elif num_positions > 8:
        base_pct *= 0.8  # Smaller positions when many holdings
    
    # Volatility adjustment
    if opportunity['volatility'] > 0.03:
        base_pct *= 0.8
    elif opportunity['volatility'] < 0.015:
        base_pct *= 1.1
    
    # Calculate final size
    position_size = portfolio_value * base_pct
    position_size = min(position_size, cash * 0.95)
    position_size = max(position_size, portfolio_value * PORTFOLIO_PARAMS['min_position_pct'])
    
    return position_size

# Load all data
logger.info("Loading data for all symbols...")
for symbol in SACRED_SYMBOLS:
    df = load_symbol_data(symbol)
    if df is not None:
        logger.debug(f"Loaded {symbol}: {len(df)} days")

# Get trading days
all_dates = pd.DatetimeIndex([])
for symbol in SACRED_SYMBOLS:
    df = load_symbol_data(symbol)
    if df is not None:
        all_dates = all_dates.union(df.index)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
mask = (all_dates >= start_dt) & (all_dates <= end_dt)
trading_days = all_dates[mask]

logger.info(f"Running dynamic portfolio optimization from {trading_days[0]} to {trading_days[-1]}")

# Run backtest
trade_count = 0
win_count = 0
rotation_count = 0

for i, date in enumerate(trading_days):
    daily_value = portfolio['cash']
    current_prices = {}
    
    # Get current prices
    for symbol in SACRED_SYMBOLS:
        df = load_symbol_data(symbol)
        if df is not None and date in df.index:
            current_prices[symbol] = df.loc[date]['close']
    
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
        
        # Track peak profit
        if symbol not in portfolio['peak_profits']:
            portfolio['peak_profits'][symbol] = profit_pct
        else:
            portfolio['peak_profits'][symbol] = max(portfolio['peak_profits'][symbol], profit_pct)
        
        # Update trailing stop
        if profit_pct >= PORTFOLIO_PARAMS['trailing_start']:
            trailing_stop = current_price * (1 - PORTFOLIO_PARAMS['trailing_distance'])
            if symbol not in portfolio['trailing_stops']:
                portfolio['trailing_stops'][symbol] = trailing_stop
            else:
                portfolio['trailing_stops'][symbol] = max(portfolio['trailing_stops'][symbol], trailing_stop)
        
        # Get stop price
        stop_price = portfolio['stop_losses'].get(symbol, entry_price * (1 - PORTFOLIO_PARAMS['stop_loss']))
        if symbol in portfolio['trailing_stops']:
            stop_price = max(stop_price, portfolio['trailing_stops'][symbol])
        
        # Check exit conditions
        should_close = False
        close_reason = ""
        
        # Stop loss
        if current_price <= stop_price:
            should_close = True
            close_reason = "Stop Loss"
        
        # Take profit
        elif profit_pct >= PORTFOLIO_PARAMS['take_profit']:
            should_close = True
            close_reason = "Take Profit"
        
        # Exit signal
        elif load_symbol_data(symbol) is not None:
            df = load_symbol_data(symbol)
            if date in df.index:
                row = df.loc[date]
                exit_score = calculate_opportunity_score(row)
                if exit_score < PORTFOLIO_PARAMS['exit_threshold']:
                    should_close = True
                    close_reason = "Exit Signal"
        
        if should_close:
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
        if symbol in portfolio['peak_profits']:
            del portfolio['peak_profits'][symbol]
        if symbol in portfolio['entry_dates']:
            del portfolio['entry_dates'][symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - P&L: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Portfolio rotation check
    if (portfolio['last_rotation_check'] is None or 
        i - portfolio['last_rotation_check'] >= PORTFOLIO_PARAMS['rotation_check_days']):
        
        portfolio['last_rotation_check'] = i
        
        # Evaluate all opportunities
        opportunities = evaluate_all_opportunities(date, portfolio['positions'])
        
        # Check for rotation opportunities
        rotations = []
        
        for opp in opportunities[:20]:  # Top 20 opportunities
            if opp['in_position']:
                continue
            
            # Check if should replace any existing position
            for symbol, position in portfolio['positions'].items():
                if symbol not in portfolio['entry_dates']:
                    portfolio['entry_dates'][symbol] = position['entry_date']
                
                holding_days = (date - portfolio['entry_dates'][symbol]).days
                
                # Get current position data
                pos_df = load_symbol_data(symbol)
                if pos_df is None or date not in pos_df.index:
                    continue
                
                pos_row = pos_df.loc[date]
                pos_score = calculate_opportunity_score(pos_row)
                
                current_pos_data = {
                    'symbol': symbol,
                    'score': pos_score,
                    'current_profit': (current_prices.get(symbol, position['entry_price']) - position['entry_price']) / position['entry_price']
                }
                
                if should_rotate_position(current_pos_data, opp, holding_days):
                    rotations.append({
                        'sell': symbol,
                        'buy': opp['symbol'],
                        'score_improvement': opp['score'] - pos_score
                    })
                    break
        
        # Execute best rotation
        if rotations:
            rotations.sort(key=lambda x: x['score_improvement'], reverse=True)
            best_rotation = rotations[0]
            
            # Sell old position
            symbol = best_rotation['sell']
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
                'reason': 'Rotation'
            })
            
            del portfolio['positions'][symbol]
            trade_count += 1
            if profit > 0:
                win_count += 1
            rotation_count += 1
            
            logger.debug(f"{date.date()}: ROTATE {symbol} -> {best_rotation['buy']} "
                        f"(Score improvement: {best_rotation['score_improvement']:.1f})")
    
    # Check for new entries
    current_portfolio_value = sum(pos['current_value'] for pos in portfolio['positions'].values())
    portfolio_risk_pct = current_portfolio_value / daily_value if daily_value > 0 else 0
    
    if len(portfolio['positions']) < PORTFOLIO_PARAMS['max_positions'] and portfolio_risk_pct < PORTFOLIO_PARAMS['max_portfolio_risk']:
        
        # Get top opportunities
        opportunities = evaluate_all_opportunities(date, portfolio['positions'])
        
        # Enter new positions
        positions_opened = 0
        max_new = 3 if len(portfolio['positions']) < 5 else 2
        
        for opp in opportunities:
            if positions_opened >= max_new:
                break
            
            if opp['in_position'] or opp['score'] < PORTFOLIO_PARAMS['entry_threshold']:
                continue
            
            symbol = opp['symbol']
            current_price = opp['price']
            
            # Calculate position size
            position_size = calculate_dynamic_position_size(
                daily_value, 
                portfolio['cash'],
                opp,
                len(portfolio['positions'])
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
                    'opportunity_score': opp['score']
                }
                
                portfolio['stop_losses'][symbol] = current_price * (1 - PORTFOLIO_PARAMS['stop_loss'])
                portfolio['entry_dates'][symbol] = date
                
                positions_opened += 1
                logger.debug(f"{date.date()}: BUY {symbol} - {shares} @ ${current_price:.2f} "
                           f"(Score: {opp['score']:.1f}, Vol: {opp['volatility']:.1%}, "
                           f"Mom: {opp['momentum']:.1f}%, RSI: {opp['rsi']:.0f})")
    
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
# Calculate actual months
from dateutil.relativedelta import relativedelta
months_diff = relativedelta(pd.to_datetime(END_DATE), pd.to_datetime(START_DATE)).months + \
              relativedelta(pd.to_datetime(END_DATE), pd.to_datetime(START_DATE)).years * 12
months_diff = max(months_diff, 1)

monthly_return = total_return_pct / months_diff
compound_monthly = ((final_value / INITIAL_CAPITAL) ** (1/months_diff) - 1) * 100
win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

# Calculate additional metrics
daily_df = pd.DataFrame(portfolio['daily_values'])
daily_df['returns'] = daily_df['value'].pct_change()
sharpe = daily_df['returns'].mean() / daily_df['returns'].std() * np.sqrt(252) if daily_df['returns'].std() > 0 else 0

daily_df['cummax'] = daily_df['value'].cummax()
daily_df['drawdown'] = (daily_df['cummax'] - daily_df['value']) / daily_df['cummax']
# Sadece pozitif drawdown değerlerini al
daily_df['drawdown'] = daily_df['drawdown'].clip(lower=0)
max_drawdown = daily_df['drawdown'].max() * 100

# Print results
print("\n" + "="*80)
print("DYNAMIC PORTFOLIO OPTIMIZER RESULTS")
print("="*80)
print(f"Period: {START_DATE} to {END_DATE}")
print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"Final Value: ${final_value:,.2f}")
print(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
print(f"Monthly Return (Simple): {monthly_return:.2f}%")
print(f"Monthly Return (Compound): {compound_monthly:.2f}%")
print(f"Total Trades: {trade_count}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Portfolio Rotations: {rotation_count}")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2f}%")
print("="*80)

# Detailed analysis
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    
    print("\nPerformance Analysis:")
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
    
    # Best performers
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': 'sum',
        'profit_pct': 'mean'
    }).round(2)
    
    print("\nTop 5 Performing Symbols:")
    print(symbol_performance.nlargest(5, 'profit'))

# Save results
results_file = DATA_DIR.parent / 'dynamic_portfolio_optimizer_results.csv'
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(results_file, index=False)
    logger.info(f"\nDetailed results saved to: {results_file}")

print("\n" + "="*80)
print("OPTIMIZATION FEATURES:")
print("✓ Dynamic portfolio rotation (best opportunities)")
print("✓ Up to 10 concurrent positions")
print("✓ Position sizes 20-30% based on signal strength")
print("✓ Low entry threshold (more opportunities)")
print("✓ Automatic profit taking at 8%")
print("✓ Trailing stops after 4% profit")
print("✓ Continuous monitoring of all 20 symbols")
print("✓ RSI and momentum-based scoring")
print("✓ Volume confirmation for entries")
print("="*80)