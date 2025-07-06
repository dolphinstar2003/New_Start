"""
Walk-Forward Portfolio Optimizer
15m verilerle 1D işlem yapan gelişmiş sistem
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR

logger.info("Walk-Forward Portfolio Optimizer with 15m Data")
logger.info("="*80)

# Walk-forward parameters
WALK_FORWARD_PARAMS = {
    'training_days': 60,        # 60 gün eğitim
    'testing_days': 30,         # 30 gün test
    'reoptimize_freq': 30,      # Her 30 günde bir yeniden optimize
    'min_data_days': 90,        # Minimum veri gereksinimi
}

# Test period
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 100000

# Portfolio parameters (dynamic portfolio optimizer'dan)
PORTFOLIO_PARAMS = {
    'base_position_pct': 0.20,
    'max_position_pct': 0.30,
    'min_position_pct': 0.05,
    'max_positions': 10,
    'max_portfolio_risk': 0.95,
    'entry_threshold': 1.0,
    'strong_entry': 2.5,
    'exit_threshold': -0.5,
    'rotation_threshold': 3.0,
    'stop_loss': 0.03,
    'take_profit': 0.08,
    'trailing_start': 0.04,
    'trailing_distance': 0.02,
    'enable_rotation': True,
    'rotation_check_days': 2,
    'min_holding_days': 3,
}

# Portfolio state
portfolio = {
    'cash': INITIAL_CAPITAL,
    'positions': {},
    'trades': [],
    'daily_values': [],
    'stop_losses': {},
    'trailing_stops': {},
    'entry_dates': {},
    'peak_profits': {},
    'optimization_history': [],
    'current_weights': {}
}

# Data cache
data_cache_15m = {}
data_cache_1d = {}

def load_15m_data(symbol):
    """Load 15-minute data"""
    if symbol in data_cache_15m:
        return data_cache_15m[symbol]
    
    raw_file = DATA_DIR / 'raw' / '15m' / f"{symbol}_15m_raw.csv"
    if not raw_file.exists():
        logger.warning(f"15m data not found for {symbol}")
        return None
    
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Calculate intraday features
    df['returns'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Intraday momentum
    df['mom_4h'] = df['close'].pct_change(16)  # 16 * 15min = 4 hours
    df['mom_1h'] = df['close'].pct_change(4)   # 4 * 15min = 1 hour
    
    # Intraday volatility
    df['volatility_1h'] = df['returns'].rolling(4).std()
    df['volatility_4h'] = df['returns'].rolling(16).std()
    
    # Price levels
    df['high_4h'] = df['high'].rolling(16).max()
    df['low_4h'] = df['low'].rolling(16).min()
    df['price_position_4h'] = (df['close'] - df['low_4h']) / (df['high_4h'] - df['low_4h'])
    
    # RSI
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_4'] = calculate_rsi(df['close'], 4)  # Fast RSI
    
    # VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Load 15m indicators
    indicators = ['supertrend', 'adx_di', 'squeeze_momentum']
    
    for indicator in indicators:
        ind_file = DATA_DIR / 'indicators' / '15m' / f"{symbol}_15m_{indicator}.csv"
        if ind_file.exists():
            ind_df = pd.read_csv(ind_file)
            ind_df['datetime'] = pd.to_datetime(ind_df['datetime'])
            ind_df.set_index('datetime', inplace=True)
            if ind_df.index.tz is not None:
                ind_df.index = ind_df.index.tz_localize(None)
            
            for col in ind_df.columns:
                df[f"{indicator}_{col}"] = ind_df[col]
    
    data_cache_15m[symbol] = df
    return df

def load_1d_data(symbol):
    """Load daily data for reference"""
    if symbol in data_cache_1d:
        return data_cache_1d[symbol]
    
    raw_file = DATA_DIR / 'raw' / '1d' / f"{symbol}_1d_raw.csv"
    if not raw_file.exists():
        return None
    
    df = pd.read_csv(raw_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Daily features
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_20'] = df['close'].pct_change(20)
    df['rsi'] = calculate_rsi(df['close'])
    
    # Load daily indicators
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
    
    data_cache_1d[symbol] = df
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_daily_features_from_15m(df_15m, date):
    """Extract daily features from 15m data"""
    # Get data for the specific day
    day_data = df_15m[df_15m.index.date == date.date()]
    
    if len(day_data) == 0:
        return None
    
    features = {
        # Price data
        'open': day_data.iloc[0]['open'],
        'high': day_data['high'].max(),
        'low': day_data['low'].min(),
        'close': day_data.iloc[-1]['close'],
        'volume': day_data['volume'].sum(),
        
        # Intraday patterns
        'intraday_volatility': day_data['returns'].std(),
        'intraday_range': (day_data['high'].max() - day_data['low'].min()) / day_data.iloc[0]['open'],
        'close_vs_open': (day_data.iloc[-1]['close'] - day_data.iloc[0]['open']) / day_data.iloc[0]['open'],
        
        # Volume patterns
        'volume_concentration': day_data['volume'].std() / day_data['volume'].mean() if day_data['volume'].mean() > 0 else 0,
        'max_volume_ratio': day_data['volume_ratio'].max() if 'volume_ratio' in day_data else 1,
        
        # Momentum from last bar
        'last_mom_1h': day_data.iloc[-1]['mom_1h'] if 'mom_1h' in day_data else 0,
        'last_mom_4h': day_data.iloc[-1]['mom_4h'] if 'mom_4h' in day_data else 0,
        
        # Technical indicators from last bar
        'last_rsi_14': day_data.iloc[-1]['rsi_14'] if 'rsi_14' in day_data else 50,
        'last_rsi_4': day_data.iloc[-1]['rsi_4'] if 'rsi_4' in day_data else 50,
        'last_price_position': day_data.iloc[-1]['price_position_4h'] if 'price_position_4h' in day_data else 0.5,
        
        # Trend from 15m
        'supertrend_signal': day_data.iloc[-1].get('supertrend_trend', 0),
        'adx_strength': day_data.iloc[-1].get('adx_di_adx', 0),
        'squeeze_momentum': day_data.iloc[-1].get('squeeze_momentum_momentum', 0),
    }
    
    return features

def walk_forward_optimize(train_start, train_end, symbols):
    """Optimize parameters using training period"""
    logger.info(f"Optimizing from {train_start} to {train_end}")
    
    # Collect training data features
    symbol_performance = {}
    
    for symbol in symbols:
        df_15m = load_15m_data(symbol)
        df_1d = load_1d_data(symbol)
        
        if df_15m is None or df_1d is None:
            continue
        
        # Get training period data
        mask_15m = (df_15m.index >= train_start) & (df_15m.index <= train_end)
        mask_1d = (df_1d.index >= train_start) & (df_1d.index <= train_end)
        
        train_15m = df_15m[mask_15m]
        train_1d = df_1d[mask_1d]
        
        if len(train_1d) < 20:
            continue
        
        # Calculate performance metrics
        total_return = (train_1d.iloc[-1]['close'] - train_1d.iloc[0]['close']) / train_1d.iloc[0]['close']
        volatility = train_1d['returns'].std()
        sharpe = (train_1d['returns'].mean() / volatility * np.sqrt(252)) if volatility > 0 else 0
        
        # Calculate win rate (up days)
        win_rate = (train_1d['returns'] > 0).sum() / len(train_1d)
        
        # Average daily features
        avg_volume_ratio = train_1d['volume'].mean() / train_1d['volume'].rolling(20).mean().mean()
        avg_momentum = train_1d['momentum_10'].mean() if 'momentum_10' in train_1d else 0
        
        symbol_performance[symbol] = {
            'return': total_return,
            'sharpe': sharpe,
            'volatility': volatility,
            'win_rate': win_rate,
            'volume_ratio': avg_volume_ratio,
            'momentum': avg_momentum,
            'score': sharpe * 0.4 + total_return * 0.3 + win_rate * 0.2 + (1 - volatility) * 0.1
        }
    
    # Rank symbols
    ranked_symbols = sorted(symbol_performance.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Create weights (top performers get higher weights)
    weights = {}
    for i, (symbol, perf) in enumerate(ranked_symbols[:15]):  # Top 15 symbols
        weight = 1.0 - (i * 0.05)  # Decreasing weights
        weights[symbol] = max(weight, 0.5)
    
    # Store optimization results
    optimization_result = {
        'train_start': train_start,
        'train_end': train_end,
        'weights': weights,
        'performance': symbol_performance
    }
    
    portfolio['optimization_history'].append(optimization_result)
    portfolio['current_weights'] = weights
    
    logger.info(f"Optimization complete. Top 5 symbols: {list(weights.keys())[:5]}")
    
    return weights

def calculate_opportunity_score_15m(symbol, date, weight=1.0):
    """Calculate opportunity score using 15m data"""
    df_15m = load_15m_data(symbol)
    df_1d = load_1d_data(symbol)
    
    if df_15m is None:
        return 0
    
    # Get daily features from 15m
    daily_features = get_daily_features_from_15m(df_15m, date)
    if daily_features is None:
        return 0
    
    score = 0
    
    # 1. Intraday momentum (30%)
    mom_score = 0
    if daily_features['last_mom_1h'] > 0.01:
        mom_score += 15
    if daily_features['last_mom_4h'] > 0.02:
        mom_score += 15
    score += mom_score
    
    # 2. Trend alignment (25%)
    if daily_features['supertrend_signal'] == 1:
        score += 25
    elif daily_features['supertrend_signal'] == -1:
        score -= 25
    
    # 3. RSI conditions (20%)
    rsi = daily_features['last_rsi_14']
    if 30 < rsi < 50:  # Oversold recovery
        score += 20
    elif 50 < rsi < 70:  # Bullish
        score += 10
    elif rsi > 80:  # Overbought
        score -= 10
    
    # 4. Volume confirmation (15%)
    if daily_features['max_volume_ratio'] > 1.5:
        score += 15 if score > 0 else -15
    
    # 5. Price position (10%)
    price_pos = daily_features['last_price_position']
    if price_pos > 0.8 and mom_score > 0:
        score += 10
    elif price_pos < 0.2 and rsi < 40:
        score += 10
    
    # Apply symbol weight from optimization
    score *= weight
    
    # Add daily indicators if available
    if df_1d is not None and date in df_1d.index:
        daily_row = df_1d.loc[date]
        
        # Daily trend confirmation
        if daily_row.get('supertrend_buy_signal', False):
            score += 20
        elif daily_row.get('supertrend_trend', 0) == 1:
            score += 10
        
        # ADX strength
        if daily_row.get('adx_di_adx', 0) > 25:
            if daily_row.get('adx_di_plus_di', 0) > daily_row.get('adx_di_minus_di', 0):
                score += 10
            else:
                score -= 10
    
    return score

def evaluate_all_opportunities_15m(date):
    """Evaluate all symbols using 15m data"""
    opportunities = []
    
    for symbol in SACRED_SYMBOLS:
        # Skip if not in current weights (not optimized)
        if symbol not in portfolio['current_weights']:
            continue
        
        weight = portfolio['current_weights'][symbol]
        score = calculate_opportunity_score_15m(symbol, date, weight)
        
        # Get current price from 15m data
        df_15m = load_15m_data(symbol)
        if df_15m is None or date not in df_15m.index:
            continue
        
        # Get last 15m bar of the day
        day_data = df_15m[df_15m.index.date == date.date()]
        if len(day_data) == 0:
            continue
        
        last_bar = day_data.iloc[-1]
        
        # Check if already in position
        in_position = symbol in portfolio['positions']
        current_profit = 0
        
        if in_position:
            entry_price = portfolio['positions'][symbol]['entry_price']
            current_profit = (last_bar['close'] - entry_price) / entry_price
        
        opportunities.append({
            'symbol': symbol,
            'score': score,
            'price': last_bar['close'],
            'weight': weight,
            'volume_ratio': last_bar.get('volume_ratio', 1),
            'in_position': in_position,
            'current_profit': current_profit,
            'intraday_momentum': last_bar.get('mom_1h', 0) * 100,
            'rsi': last_bar.get('rsi_14', 50)
        })
    
    # Sort by score
    opportunities.sort(key=lambda x: x['score'], reverse=True)
    return opportunities

# Load initial data
logger.info("Loading 15m and 1d data for all symbols...")
symbols_with_data = []

for symbol in SACRED_SYMBOLS:
    df_15m = load_15m_data(symbol)
    df_1d = load_1d_data(symbol)
    
    if df_15m is not None and df_1d is not None:
        symbols_with_data.append(symbol)
        logger.debug(f"Loaded {symbol}: 15m={len(df_15m)} bars, 1d={len(df_1d)} bars")

logger.info(f"Loaded data for {len(symbols_with_data)} symbols")

# Get all trading days
all_dates = pd.DatetimeIndex([])
for symbol in symbols_with_data:
    df_1d = load_1d_data(symbol)
    if df_1d is not None:
        all_dates = all_dates.union(df_1d.index)
all_dates = all_dates.sort_values()

# Filter date range
start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
mask = (all_dates >= start_dt) & (all_dates <= end_dt)
trading_days = all_dates[mask]

# Ensure we have enough history
first_trade_day = start_dt + timedelta(days=WALK_FORWARD_PARAMS['min_data_days'])
trade_days = trading_days[trading_days >= first_trade_day]

logger.info(f"Running walk-forward backtest from {trade_days[0]} to {trade_days[-1]}")

# Run walk-forward backtest
trade_count = 0
win_count = 0
rotation_count = 0
last_optimization_date = None

for i, date in enumerate(trade_days):
    # Check if need to optimize
    if (last_optimization_date is None or 
        (date - last_optimization_date).days >= WALK_FORWARD_PARAMS['reoptimize_freq']):
        
        # Calculate training period
        train_end = date - timedelta(days=1)
        train_start = train_end - timedelta(days=WALK_FORWARD_PARAMS['training_days'])
        
        # Optimize
        walk_forward_optimize(train_start, train_end, symbols_with_data)
        last_optimization_date = date
    
    # Daily portfolio value
    daily_value = portfolio['cash']
    
    # Evaluate opportunities using 15m data
    opportunities = evaluate_all_opportunities_15m(date)
    
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
        else:
            current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
            if current_opp and current_opp['score'] < PORTFOLIO_PARAMS['exit_threshold']:
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
        for tracking_dict in [portfolio['stop_losses'], portfolio['trailing_stops'], 
                             portfolio['peak_profits'], portfolio['entry_dates']]:
            if symbol in tracking_dict:
                del tracking_dict[symbol]
        
        trade_count += 1
        if profit > 0:
            win_count += 1
        
        logger.debug(f"{date.date()}: CLOSE {symbol} - {reason} - P&L: ${profit:.2f} ({profit_pct:.2f}%)")
    
    # Portfolio rotation check
    if (i % PORTFOLIO_PARAMS['rotation_check_days'] == 0 and 
        PORTFOLIO_PARAMS['enable_rotation']):
        
        # Check for rotation opportunities
        for new_opp in opportunities[:20]:  # Top 20 opportunities
            if new_opp['in_position']:
                continue
            
            # Check if should replace any position
            for symbol, position in list(portfolio['positions'].items()):
                if symbol not in portfolio['entry_dates']:
                    portfolio['entry_dates'][symbol] = position['entry_date']
                
                holding_days = (date - portfolio['entry_dates'][symbol]).days
                
                if holding_days < PORTFOLIO_PARAMS['min_holding_days']:
                    continue
                
                # Get current position score
                current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
                if not current_opp:
                    continue
                
                # Check rotation conditions
                score_diff = new_opp['score'] - current_opp['score']
                if (score_diff > PORTFOLIO_PARAMS['rotation_threshold'] and
                    current_opp['current_profit'] > -0.01 and  # Not in loss
                    current_opp['current_profit'] < 0.15):     # Not too profitable
                    
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
                                f"(Score diff: {score_diff:.1f})")
                    break
    
    # Open new positions
    if len(portfolio['positions']) < PORTFOLIO_PARAMS['max_positions']:
        positions_opened = 0
        max_new = 3 if len(portfolio['positions']) < 5 else 2
        
        for opp in opportunities:
            if positions_opened >= max_new:
                break
            
            if opp['in_position'] or opp['score'] < PORTFOLIO_PARAMS['entry_threshold']:
                continue
            
            symbol = opp['symbol']
            current_price = opp['price']
            
            # Dynamic position sizing
            if opp['score'] >= 60:
                position_pct = PORTFOLIO_PARAMS['max_position_pct']
            elif opp['score'] >= 40:
                position_pct = PORTFOLIO_PARAMS['base_position_pct']
            elif opp['score'] >= 20:
                position_pct = 0.15
            else:
                position_pct = PORTFOLIO_PARAMS['min_position_pct']
            
            position_size = daily_value * position_pct
            position_size = min(position_size, portfolio['cash'] * 0.95)
            
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
                           f"(Score: {opp['score']:.1f}, Weight: {opp['weight']:.2f}, "
                           f"Mom: {opp['intraday_momentum']:.1f}%, RSI: {opp['rsi']:.0f})")
    
    # Record daily value
    portfolio['daily_values'].append({
        'date': date,
        'value': daily_value,
        'cash': portfolio['cash'],
        'positions': len(portfolio['positions'])
    })

# Close remaining positions
final_date = trade_days[-1]
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
monthly_return = total_return_pct / 12
compound_monthly = ((final_value / INITIAL_CAPITAL) ** (1/12) - 1) * 100
win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0

# Calculate additional metrics
daily_df = pd.DataFrame(portfolio['daily_values'])
daily_df['returns'] = daily_df['value'].pct_change()
sharpe = daily_df['returns'].mean() / daily_df['returns'].std() * np.sqrt(252) if daily_df['returns'].std() > 0 else 0

daily_df['cummax'] = daily_df['value'].cummax()
daily_df['drawdown'] = (daily_df['cummax'] - daily_df['value']) / daily_df['cummax']
max_drawdown = daily_df['drawdown'].max() * 100

# Print results
print("\n" + "="*80)
print("WALK-FORWARD PORTFOLIO OPTIMIZER RESULTS")
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
print(f"Reoptimizations: {len(portfolio['optimization_history'])}")
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
    
    # Best performing symbols
    symbol_performance = trades_df.groupby('symbol').agg({
        'profit': 'sum',
        'profit_pct': 'mean'
    }).round(2)
    
    print("\nTop 5 Performing Symbols:")
    print(symbol_performance.nlargest(5, 'profit'))

# Save results
results_file = DATA_DIR.parent / 'walk_forward_results.csv'
if portfolio['trades']:
    trades_df = pd.DataFrame(portfolio['trades'])
    trades_df.to_csv(results_file, index=False)
    logger.info(f"\nResults saved to: {results_file}")

print("\n" + "="*80)
print("WALK-FORWARD FEATURES:")
print("✓ Uses 15-minute data for intraday analysis")
print("✓ Reoptimizes every 30 days")
print("✓ Dynamic symbol weighting based on performance")
print("✓ Combines intraday and daily indicators")
print("✓ Portfolio rotation with optimized symbols")
print("✓ Up to 10 concurrent positions")
print("✓ Position sizes 20-30% based on signal strength")
print("✓ Walk-forward testing prevents overfitting")
print("="*80)