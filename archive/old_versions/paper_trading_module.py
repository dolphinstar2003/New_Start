"""
Paper Trading Module for Dynamic Portfolio Optimizer
Uses AlgoLab as data source and implements the exact trading logic
With Telegram integration for notifications and control
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
import pickle
import json
import asyncio
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from core.algolab_api import AlgoLabAPI
from core.algolab_socket import AlgoLabSocket
from utils.algolab_auth import AlgoLabAuth
from utils.telegram_utils import escape_markdown_v1, format_currency, format_percentage
# Telegram imports will be done dynamically in initialize()
from telegram_integration import TELEGRAM_CONFIG

logger.info("Paper Trading Module - Dynamic Portfolio Optimizer")
logger.info("="*80)

class PaperTradingModule:
    """Paper trading implementation of Dynamic Portfolio Optimizer"""
    
    def __init__(self, initial_capital=100000):
        # Portfolio parameters (from optimized strategy)
        self.PORTFOLIO_PARAMS = {
            'base_position_pct': 0.20,      # Base %20 position
            'max_position_pct': 0.30,       # Max %30 (strong signals)
            'min_position_pct': 0.05,       # Min %5 (weak signal)
            'max_positions': 10,            # Max 10 positions
            'max_portfolio_risk': 0.95,     # Max %95 portfolio risk
            'entry_threshold': 1.0,         # Low entry threshold
            'strong_entry': 2.5,            # Strong signal threshold
            'exit_threshold': -0.5,         # Exit threshold
            'rotation_threshold': 3.0,      # Rotation score difference
            'stop_loss': 0.03,              # %3 stop loss
            'take_profit': 0.08,            # %8 take profit
            'trailing_start': 0.04,         # %4 trailing start
            'trailing_distance': 0.02,      # %2 trailing distance
            'enable_rotation': True,        # Dynamic rotation
            'rotation_check_days': 2,       # Check rotation every 2 days
            'min_holding_days': 3,          # Min 3 days hold
            'profit_lock_threshold': 0.15,  # %15 profit lock
        }
        
        # Portfolio state
        self.portfolio = {
            'cash': initial_capital,
            'initial_capital': initial_capital,
            'positions': {},
            'trades': [],
            'pending_orders': {},
            'stop_losses': {},
            'trailing_stops': {},
            'entry_dates': {},
            'peak_profits': {},
            'last_rotation_check': None,
            'daily_values': [],
            'current_prices': {},
            'last_update': None
        }
        
        # AlgoLab connections
        self.algolab_api = None
        self.algolab_socket = None
        self.auth = None
        
        # Market data cache
        self.market_data = {}
        self.indicators = {}
        self.last_data_update = {}
        
        # Control flags
        self.is_running = False
        self.auto_trade_enabled = False
        self.telegram_notifications = True
        self.require_confirmation = True
        self._force_check_flag = False  # For telegram force check command
        
        # Telegram bot
        self.telegram_bot = None
        
    async def initialize(self):
        """Initialize AlgoLab connections"""
        try:
            # Initialize authentication
            self.auth = AlgoLabAuth()
            
            # Authenticate and get API instance
            self.algolab_api = self.auth.authenticate()
            
            if not self.algolab_api:
                logger.error("Failed to authenticate with AlgoLab")
                return False
            
            # Initialize WebSocket
            self.algolab_socket = AlgoLabSocket(
                api_key=self.auth.api_key,
                hash_token=self.algolab_api.hash
            )
            
            # Try to connect WebSocket (optional - continue if fails)
            try:
                self.algolab_socket.connect()
                
                # Subscribe to all sacred symbols
                for symbol in SACRED_SYMBOLS:
                    # Remove .IS suffix for AlgoLab
                    clean_symbol = symbol.replace('.IS', '')
                    self.algolab_socket.subscribe(
                        clean_symbol,
                        ["price", "depth", "trade"]
                    )
                logger.info("WebSocket connected and subscribed to symbols")
            except Exception as e:
                logger.warning(f"WebSocket connection failed: {e}")
                logger.warning("Continuing without real-time data - will use polling")
                self.algolab_socket = None
            
            # Initialize Telegram bot
            try:
                from telegram_bot_integrated import IntegratedTelegramBot
                self.telegram_bot = IntegratedTelegramBot(self)
                self.telegram_bot.start()
                logger.info("Integrated Telegram bot initialized with full system access")
            except Exception as e:
                logger.warning(f"Telegram bot initialization failed: {e}")
                self.telegram_bot = None
                self.telegram_notifications = False
            
            logger.info("Paper Trading Module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_opportunity_score(self, symbol, current_data):
        """Calculate opportunity score for symbol"""
        score = 0
        
        # Get latest market data
        market_data = self.market_data.get(symbol, {})
        if not market_data:
            return 0
        
        # 1. Price momentum (30%)
        price_change_1h = market_data.get('price_change_1h', 0)
        price_change_day = market_data.get('price_change_day', 0)
        
        if price_change_1h > 0.01:  # %1 hourly
            score += 15
        if price_change_day > 0.02:  # %2 daily
            score += 15
        
        # 2. Volume activity (20%)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.2:
            score += 10
        
        # 3. Technical indicators from cached data (30%)
        indicators = self.indicators.get(symbol, {})
        
        # Supertrend
        if indicators.get('supertrend_trend', 0) == 1:
            score += 20
        elif indicators.get('supertrend_buy_signal', False):
            score += 30
        
        # RSI
        rsi = indicators.get('rsi', 50)
        if 30 < rsi < 50:  # Oversold recovery
            score += 15
        elif 50 < rsi < 70:  # Bullish
            score += 10
        
        # 4. Market depth (10%)
        bid_ask_ratio = market_data.get('bid_ask_ratio', 1.0)
        if bid_ask_ratio > 1.1:  # More buyers
            score += 10
        
        # 5. Recent trades momentum (10%)
        buy_sell_ratio = market_data.get('buy_sell_ratio', 1.0)
        if buy_sell_ratio > 1.2:
            score += 10
        
        return score
    
    def evaluate_all_opportunities(self):
        """Evaluate all symbols and return ranked opportunities"""
        opportunities = []
        
        for symbol in SACRED_SYMBOLS:
            # Skip if no recent data
            if symbol not in self.market_data:
                continue
            
            # Get current market data
            market_data = self.market_data[symbol]
            current_price = market_data.get('last_price', 0)
            
            if current_price == 0:
                continue
            
            # Calculate opportunity score
            score = self.calculate_opportunity_score(symbol, market_data)
            
            # Check if already in position
            in_position = symbol in self.portfolio['positions']
            current_profit = 0
            
            if in_position:
                position = self.portfolio['positions'][symbol]
                entry_price = position['entry_price']
                current_profit = (current_price - entry_price) / entry_price
            
            opportunities.append({
                'symbol': symbol,
                'score': score,
                'price': current_price,
                'volume_ratio': market_data.get('volume_ratio', 1.0),
                'in_position': in_position,
                'current_profit': current_profit,
                'momentum_1h': market_data.get('price_change_1h', 0) * 100,
                'momentum_day': market_data.get('price_change_day', 0) * 100,
            })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities
    
    def should_rotate_position(self, current_pos_data, new_opportunity, holding_days):
        """Determine if should rotate position"""
        if not self.PORTFOLIO_PARAMS['enable_rotation']:
            return False
        
        # Check minimum holding period
        if holding_days < self.PORTFOLIO_PARAMS['min_holding_days']:
            return False
        
        # Don't rotate if position is very profitable
        if current_pos_data['current_profit'] > self.PORTFOLIO_PARAMS['profit_lock_threshold']:
            return False
        
        # Don't rotate if in loss (wait for stop loss)
        if current_pos_data['current_profit'] < -0.01:
            return False
        
        # Rotate if new opportunity is significantly better
        score_diff = new_opportunity['score'] - current_pos_data['score']
        if score_diff > self.PORTFOLIO_PARAMS['rotation_threshold']:
            return True
        
        return False
    
    def calculate_position_size(self, portfolio_value, cash, opportunity, num_positions):
        """Calculate dynamic position size"""
        # Base size based on score
        if opportunity['score'] >= 60:
            base_pct = self.PORTFOLIO_PARAMS['max_position_pct']
        elif opportunity['score'] >= 40:
            base_pct = self.PORTFOLIO_PARAMS['base_position_pct']
        elif opportunity['score'] >= 20:
            base_pct = (self.PORTFOLIO_PARAMS['base_position_pct'] + 
                       self.PORTFOLIO_PARAMS['min_position_pct']) / 2
        else:
            base_pct = self.PORTFOLIO_PARAMS['min_position_pct']
        
        # Adjust for number of positions
        if num_positions < 5:
            base_pct *= 1.2  # Larger positions when few holdings
        elif num_positions > 8:
            base_pct *= 0.8  # Smaller positions when many holdings
        
        # Calculate final size
        position_size = portfolio_value * base_pct
        position_size = min(position_size, cash * 0.95)
        position_size = max(position_size, portfolio_value * self.PORTFOLIO_PARAMS['min_position_pct'])
        
        return position_size
    
    async def execute_trade(self, action, symbol, shares, price, reason=""):
        """Execute a paper trade with Telegram confirmation if required"""
        trade_time = datetime.now()
        
        # Check if confirmation required
        if self.require_confirmation and self.telegram_bot:
            # Request confirmation via Telegram
            trade_id = await self.telegram_bot.request_trade_confirmation(
                action, symbol, shares, price, reason
            )
            logger.info(f"Trade confirmation requested via Telegram (ID: {trade_id})")
            
            # Wait for confirmation (max 60 seconds)
            max_wait = 60
            wait_interval = 1
            elapsed = 0
            
            while elapsed < max_wait:
                if trade_id not in self.telegram_bot.pending_confirmations:
                    # Either confirmed or cancelled
                    break
                await asyncio.sleep(wait_interval)
                elapsed += wait_interval
            
            if trade_id in self.telegram_bot.pending_confirmations:
                # Timeout - cancel trade
                del self.telegram_bot.pending_confirmations[trade_id]
                logger.warning(f"Trade confirmation timeout for {symbol}")
                if self.telegram_bot:
                    timeout_msg = f"⏰ Trade timeout: {action} {shares} {escape_markdown_v1(symbol)}"
                    await self.telegram_bot.send_notification(
                        timeout_msg,
                        "warning"
                    )
                return False
        
        if action == "BUY":
            # Check if enough cash
            total_cost = shares * price
            if total_cost > self.portfolio['cash']:
                logger.warning(f"Insufficient cash for {symbol}. Need: ${total_cost:.2f}, Have: ${self.portfolio['cash']:.2f}")
                return False
            
            # Execute buy
            self.portfolio['cash'] -= total_cost
            self.portfolio['positions'][symbol] = {
                'shares': shares,
                'entry_price': price,
                'entry_value': total_cost,
                'entry_date': trade_time,
                'current_value': total_cost,
                'opportunity_score': self.calculate_opportunity_score(symbol, self.market_data.get(symbol, {}))
            }
            
            # Set stop loss
            self.portfolio['stop_losses'][symbol] = price * (1 - self.PORTFOLIO_PARAMS['stop_loss'])
            self.portfolio['entry_dates'][symbol] = trade_time
            
            logger.info(f"BUY {symbol}: {shares} shares @ ${price:.2f} = ${total_cost:.2f} | Reason: {reason}")
            
            # Send Telegram notification
            if self.telegram_notifications and self.telegram_bot:
                await self.telegram_bot.send_trade_notification(
                    action, symbol, shares, price, reason
                )
            
        elif action == "SELL":
            if symbol not in self.portfolio['positions']:
                logger.warning(f"No position in {symbol} to sell")
                return False
            
            position = self.portfolio['positions'][symbol]
            exit_value = shares * price
            entry_value = shares * position['entry_price']
            profit = exit_value - entry_value
            profit_pct = (profit / entry_value) * 100
            
            # Execute sell
            self.portfolio['cash'] += exit_value
            
            # Record trade
            self.portfolio['trades'].append({
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': trade_time,
                'entry_price': position['entry_price'],
                'exit_price': price,
                'shares': shares,
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': reason
            })
            
            # Remove position
            del self.portfolio['positions'][symbol]
            for tracking_dict in [self.portfolio['stop_losses'], 
                                self.portfolio['trailing_stops'], 
                                self.portfolio['peak_profits'],
                                self.portfolio['entry_dates']]:
                if symbol in tracking_dict:
                    del tracking_dict[symbol]
            
            logger.info(f"SELL {symbol}: {shares} shares @ ${price:.2f} = ${exit_value:.2f} | "
                       f"P&L: ${profit:.2f} ({profit_pct:.2f}%) | Reason: {reason}")
            
            # Send Telegram notification
            if self.telegram_notifications and self.telegram_bot:
                await self.telegram_bot.send_trade_notification(
                    action, symbol, shares, price, reason, profit
                )
        
        return True
    
    async def check_positions_for_exit(self):
        """Check all positions for exit conditions"""
        positions_to_close = []
        
        for symbol, position in self.portfolio['positions'].items():
            market_data = self.market_data.get(symbol, {})
            current_price = market_data.get('last_price', position['entry_price'])
            
            if current_price == 0:
                continue
            
            entry_price = position['entry_price']
            profit_pct = (current_price - entry_price) / entry_price
            
            # Track peak profit
            if symbol not in self.portfolio['peak_profits']:
                self.portfolio['peak_profits'][symbol] = profit_pct
            else:
                self.portfolio['peak_profits'][symbol] = max(
                    self.portfolio['peak_profits'][symbol], 
                    profit_pct
                )
            
            # Update trailing stop
            if profit_pct >= self.PORTFOLIO_PARAMS['trailing_start']:
                trailing_stop = current_price * (1 - self.PORTFOLIO_PARAMS['trailing_distance'])
                if symbol not in self.portfolio['trailing_stops']:
                    self.portfolio['trailing_stops'][symbol] = trailing_stop
                else:
                    self.portfolio['trailing_stops'][symbol] = max(
                        self.portfolio['trailing_stops'][symbol], 
                        trailing_stop
                    )
            
            # Get stop price
            stop_price = self.portfolio['stop_losses'].get(
                symbol, 
                entry_price * (1 - self.PORTFOLIO_PARAMS['stop_loss'])
            )
            if symbol in self.portfolio['trailing_stops']:
                stop_price = max(stop_price, self.portfolio['trailing_stops'][symbol])
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            
            # Stop loss
            if current_price <= stop_price:
                should_close = True
                close_reason = "Stop Loss"
            
            # Take profit
            elif profit_pct >= self.PORTFOLIO_PARAMS['take_profit']:
                should_close = True
                close_reason = "Take Profit"
            
            # Exit signal
            else:
                exit_score = self.calculate_opportunity_score(symbol, market_data)
                if exit_score < self.PORTFOLIO_PARAMS['exit_threshold']:
                    should_close = True
                    close_reason = "Exit Signal"
            
            if should_close:
                positions_to_close.append((symbol, position['shares'], current_price, close_reason))
        
        # Execute closes
        for symbol, shares, price, reason in positions_to_close:
            await self.execute_trade("SELL", symbol, shares, price, reason)
    
    async def check_for_new_entries(self):
        """Check for new entry opportunities"""
        # Calculate portfolio value
        portfolio_value = self.portfolio['cash']
        for symbol, position in self.portfolio['positions'].items():
            market_data = self.market_data.get(symbol, {})
            current_price = market_data.get('last_price', position['entry_price'])
            portfolio_value += position['shares'] * current_price
        
        # Check if can add more positions
        current_portfolio_value = portfolio_value - self.portfolio['cash']
        portfolio_risk_pct = current_portfolio_value / portfolio_value if portfolio_value > 0 else 0
        
        if (len(self.portfolio['positions']) >= self.PORTFOLIO_PARAMS['max_positions'] or 
            portfolio_risk_pct >= self.PORTFOLIO_PARAMS['max_portfolio_risk']):
            return
        
        # Get opportunities
        opportunities = self.evaluate_all_opportunities()
        
        # Enter new positions
        positions_opened = 0
        max_new = 3 if len(self.portfolio['positions']) < 5 else 2
        
        for opp in opportunities:
            if positions_opened >= max_new:
                break
            
            if opp['in_position'] or opp['score'] < self.PORTFOLIO_PARAMS['entry_threshold']:
                continue
            
            symbol = opp['symbol']
            current_price = opp['price']
            
            # Calculate position size
            position_size = self.calculate_position_size(
                portfolio_value,
                self.portfolio['cash'],
                opp,
                len(self.portfolio['positions'])
            )
            
            shares = int(position_size / current_price)
            
            if shares > 0 and shares * current_price <= self.portfolio['cash'] * 0.95:
                # Open position
                success = await self.execute_trade(
                    "BUY", 
                    symbol, 
                    shares, 
                    current_price,
                    f"Entry Signal (Score: {opp['score']:.1f})"
                )
                
                if success:
                    positions_opened += 1
    
    async def check_for_rotation(self):
        """Check for portfolio rotation opportunities"""
        if not self.PORTFOLIO_PARAMS['enable_rotation']:
            return
        
        current_time = datetime.now()
        
        # Check if it's time for rotation check
        if (self.portfolio['last_rotation_check'] is not None and
            (current_time - self.portfolio['last_rotation_check']).days < 
            self.PORTFOLIO_PARAMS['rotation_check_days']):
            return
        
        self.portfolio['last_rotation_check'] = current_time
        
        # Get all opportunities
        opportunities = self.evaluate_all_opportunities()
        
        # Find best rotation candidates
        rotations = []
        
        for new_opp in opportunities[:20]:  # Top 20 opportunities
            if new_opp['in_position']:
                continue
            
            # Check against existing positions
            for symbol, position in self.portfolio['positions'].items():
                holding_days = (current_time - self.portfolio['entry_dates'][symbol]).days
                
                # Get current position data
                current_opp = next((o for o in opportunities if o['symbol'] == symbol), None)
                if not current_opp:
                    continue
                
                if self.should_rotate_position(current_opp, new_opp, holding_days):
                    rotations.append({
                        'sell': symbol,
                        'buy': new_opp['symbol'],
                        'score_improvement': new_opp['score'] - current_opp['score']
                    })
                    break
        
        # Execute best rotation
        if rotations:
            rotations.sort(key=lambda x: x['score_improvement'], reverse=True)
            best_rotation = rotations[0]
            
            # Sell old position
            symbol = best_rotation['sell']
            position = self.portfolio['positions'][symbol]
            market_data = self.market_data.get(symbol, {})
            current_price = market_data.get('last_price', position['entry_price'])
            
            await self.execute_trade(
                "SELL",
                symbol,
                position['shares'],
                current_price,
                f"Rotation to {best_rotation['buy']}"
            )
            
            logger.info(f"ROTATION: {symbol} -> {best_rotation['buy']} "
                       f"(Score improvement: {best_rotation['score_improvement']:.1f})")
    
    async def update_market_data(self, symbol, data):
        """Update market data from WebSocket"""
        clean_symbol = symbol + ".IS"  # Add .IS suffix for internal use
        
        if clean_symbol not in self.market_data:
            self.market_data[clean_symbol] = {
                'last_price': 0,
                'open_price': 0,
                'high_price': 0,
                'low_price': 0,
                'volume': 0,
                'volume_ma': 0,
                'price_history': [],
                'volume_history': [],
                'last_update': datetime.now()
            }
        
        market_data = self.market_data[clean_symbol]
        
        # Update based on message type
        if data.get('type') == 'price':
            market_data['last_price'] = data.get('price', 0)
            market_data['volume'] = data.get('volume', 0)
            market_data['last_update'] = datetime.now()
            
            # Update price history
            market_data['price_history'].append({
                'time': datetime.now(),
                'price': data.get('price', 0)
            })
            
            # Keep only last hour of data
            cutoff_time = datetime.now() - timedelta(hours=1)
            market_data['price_history'] = [
                p for p in market_data['price_history'] 
                if p['time'] > cutoff_time
            ]
            
            # Calculate price changes
            if len(market_data['price_history']) > 0:
                hour_ago_price = market_data['price_history'][0]['price']
                market_data['price_change_1h'] = (
                    (market_data['last_price'] - hour_ago_price) / hour_ago_price
                    if hour_ago_price > 0 else 0
                )
            
            # Calculate volume ratio
            if len(market_data['volume_history']) >= 20:
                market_data['volume_ma'] = np.mean([v['volume'] for v in market_data['volume_history'][-20:]])
                market_data['volume_ratio'] = (
                    market_data['volume'] / market_data['volume_ma']
                    if market_data['volume_ma'] > 0 else 1.0
                )
        
        elif data.get('type') == 'depth':
            # Update bid/ask ratio
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            if bids and asks:
                total_bid_volume = sum(b.get('volume', 0) for b in bids[:5])
                total_ask_volume = sum(a.get('volume', 0) for a in asks[:5])
                
                market_data['bid_ask_ratio'] = (
                    total_bid_volume / total_ask_volume
                    if total_ask_volume > 0 else 1.0
                )
        
        elif data.get('type') == 'trade':
            # Track buy/sell ratio
            if 'trades' not in market_data:
                market_data['trades'] = []
            
            market_data['trades'].append({
                'time': datetime.now(),
                'side': data.get('side', 'unknown'),
                'volume': data.get('volume', 0)
            })
            
            # Keep only last 15 minutes
            cutoff_time = datetime.now() - timedelta(minutes=15)
            market_data['trades'] = [
                t for t in market_data['trades']
                if t['time'] > cutoff_time
            ]
            
            # Calculate buy/sell ratio
            buy_volume = sum(t['volume'] for t in market_data['trades'] if t['side'] == 'buy')
            sell_volume = sum(t['volume'] for t in market_data['trades'] if t['side'] == 'sell')
            
            market_data['buy_sell_ratio'] = (
                buy_volume / sell_volume
                if sell_volume > 0 else 1.0
            )
        
        # Update portfolio current prices
        self.portfolio['current_prices'][clean_symbol] = market_data['last_price']
        self.portfolio['last_update'] = datetime.now()
    
    async def update_market_data_via_api(self):
        """Update market data using API calls (fallback when WebSocket not available)"""
        try:
            for symbol in SACRED_SYMBOLS:
                clean_symbol = symbol.replace('.IS', '')
                
                # Get symbol info
                stock_info = self.algolab_api.get_symbol_info(clean_symbol)
                
                if stock_info.get('success') and stock_info.get('content'):
                    data = stock_info['content']
                    
                    if symbol not in self.market_data:
                        self.market_data[symbol] = {
                            'last_price': 0,
                            'open_price': 0,
                            'high_price': 0,
                            'low_price': 0,
                            'volume': 0,
                            'price_history': [],
                            'last_update': datetime.now()
                        }
                    
                    # Update market data
                    self.market_data[symbol]['last_price'] = float(data.get('lastPrice', 0))
                    self.market_data[symbol]['open_price'] = float(data.get('open', 0))
                    self.market_data[symbol]['high_price'] = float(data.get('high', 0))
                    self.market_data[symbol]['low_price'] = float(data.get('low', 0))
                    self.market_data[symbol]['volume'] = int(data.get('volume', 0))
                    self.market_data[symbol]['last_update'] = datetime.now()
                    
                    # Calculate price changes
                    if self.market_data[symbol]['open_price'] > 0:
                        self.market_data[symbol]['price_change_day'] = (
                            (self.market_data[symbol]['last_price'] - self.market_data[symbol]['open_price']) / 
                            self.market_data[symbol]['open_price']
                        )
                    
                    # Update current prices
                    self.portfolio['current_prices'][symbol] = self.market_data[symbol]['last_price']
                
                # Small delay between API calls
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error updating market data via API: {e}")
    
    async def trading_loop(self):
        """Main trading loop"""
        logger.info("Starting paper trading loop...")
        self.is_running = True
        
        # Set up WebSocket message handler if available
        if self.algolab_socket:
            async def message_handler(message):
                if 'symbol' in message and 'data' in message:
                    await self.update_market_data(message['symbol'], message['data'])
            
            self.algolab_socket.on_message = message_handler
        else:
            logger.warning("WebSocket not available - using API polling for market data")
        
        # Main loop
        check_interval = 60  # Check every minute
        last_check = datetime.now()
        last_api_update = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Only trade during market hours (10:00 - 18:00 Istanbul time)
                if 10 <= current_time.hour < 18 and current_time.weekday() < 5:
                    
                    # Update market data via API if WebSocket not available
                    if not self.algolab_socket and (current_time - last_api_update).seconds >= 60:
                        await self.update_market_data_via_api()
                        last_api_update = current_time
                    
                    # Check positions every interval or if force check requested
                    if ((current_time - last_check).seconds >= check_interval) or self._force_check_flag:
                        if self._force_check_flag:
                            logger.info("Force check requested via Telegram")
                            self._force_check_flag = False
                        
                        if self.auto_trade_enabled:
                            # Check exit conditions
                            await self.check_positions_for_exit()
                            
                            # Check for rotations
                            await self.check_for_rotation()
                            
                            # Check for new entries
                            await self.check_for_new_entries()
                        
                        # Update portfolio value
                        await self.update_portfolio_value()
                        
                        # Send hourly updates via Telegram
                        if self.telegram_bot and current_time.minute == 0:
                            status = self.get_portfolio_status()
                            hourly_msg = (
                                f"📊 Hourly Update\n"
                                f"Value: {format_currency(status['portfolio_value'])}\n"
                                f"Return: {format_percentage(status['total_return_pct'])}\n"
                                f"Positions: {status['num_positions']}"
                            )
                            await self.telegram_bot.send_notification(
                                hourly_msg,
                                "info"
                            )
                        
                        last_check = current_time
                
                # Send daily summary at 18:30
                if (current_time.hour == 18 and current_time.minute == 30 and 
                    self.telegram_bot and not hasattr(self, '_daily_summary_sent')):
                    await self.telegram_bot.send_daily_summary()
                    self._daily_summary_sent = True
                elif current_time.hour != 18:
                    self._daily_summary_sent = False
                
                # Sleep briefly
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def update_portfolio_value(self):
        """Update and log portfolio value"""
        portfolio_value = self.portfolio['cash']
        
        for symbol, position in self.portfolio['positions'].items():
            market_data = self.market_data.get(symbol, {})
            current_price = market_data.get('last_price', position['entry_price'])
            position_value = position['shares'] * current_price
            portfolio_value += position_value
            
            # Update position current value
            position['current_value'] = position_value
        
        # Calculate returns
        total_return = portfolio_value - self.portfolio['initial_capital']
        total_return_pct = (total_return / self.portfolio['initial_capital']) * 100
        
        # Log portfolio status
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f} | "
                   f"Return: ${total_return:,.2f} ({total_return_pct:.2f}%) | "
                   f"Cash: ${self.portfolio['cash']:,.2f} | "
                   f"Positions: {len(self.portfolio['positions'])}")
        
        # Save daily value
        self.portfolio['daily_values'].append({
            'datetime': datetime.now(),
            'value': portfolio_value,
            'cash': self.portfolio['cash'],
            'positions': len(self.portfolio['positions']),
            'return_pct': total_return_pct
        })
    
    def get_portfolio_status(self):
        """Get current portfolio status"""
        portfolio_value = self.portfolio['cash']
        positions_detail = []
        
        for symbol, position in self.portfolio['positions'].items():
            market_data = self.market_data.get(symbol, {})
            current_price = market_data.get('last_price', position['entry_price'])
            position_value = position['shares'] * current_price
            portfolio_value += position_value
            
            profit = position_value - position['entry_value']
            profit_pct = (profit / position['entry_value']) * 100
            
            positions_detail.append({
                'symbol': symbol,
                'shares': position['shares'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'value': position_value,
                'profit': profit,
                'profit_pct': profit_pct,
                'holding_days': (datetime.now() - position['entry_date']).days
            })
        
        total_return = portfolio_value - self.portfolio['initial_capital']
        total_return_pct = (total_return / self.portfolio['initial_capital']) * 100
        
        # Calculate win rate
        if self.portfolio['trades']:
            wins = sum(1 for t in self.portfolio['trades'] if t['profit'] > 0)
            win_rate = (wins / len(self.portfolio['trades'])) * 100
        else:
            win_rate = 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': self.portfolio['cash'],
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'positions': positions_detail,
            'num_positions': len(self.portfolio['positions']),
            'total_trades': len(self.portfolio['trades']),
            'win_rate': win_rate,
            'last_update': self.portfolio['last_update']
        }
    
    def save_state(self):
        """Save portfolio state to file"""
        state_file = DATA_DIR / 'paper_trading_state.pkl'
        with open(state_file, 'wb') as f:
            pickle.dump({
                'portfolio': self.portfolio,
                'market_data': self.market_data,
                'indicators': self.indicators,
                'last_save': datetime.now()
            }, f)
        logger.info(f"Portfolio state saved to {state_file}")
    
    def load_state(self):
        """Load portfolio state from file"""
        state_file = DATA_DIR / 'paper_trading_state.pkl'
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                self.portfolio = state['portfolio']
                self.market_data = state['market_data']
                self.indicators = state['indicators']
                logger.info(f"Portfolio state loaded from {state_file}")
                return True
        return False
    
    def get_trade_history(self):
        """Get detailed trade history"""
        return pd.DataFrame(self.portfolio['trades'])
    
    def get_performance_metrics(self):
        """Calculate detailed performance metrics"""
        if not self.portfolio['trades']:
            return {}
        
        trades_df = pd.DataFrame(self.portfolio['trades'])
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['profit'] > 0]
        losing_trades = trades_df[trades_df['profit'] < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100
        
        # Profit metrics
        total_profit = trades_df['profit'].sum()
        avg_profit = trades_df['profit'].mean()
        avg_profit_pct = trades_df['profit_pct'].mean()
        
        # Risk metrics
        if len(winning_trades) > 0:
            avg_win = winning_trades['profit_pct'].mean()
            max_win = winning_trades['profit_pct'].max()
        else:
            avg_win = 0
            max_win = 0
        
        if len(losing_trades) > 0:
            avg_loss = losing_trades['profit_pct'].mean()
            max_loss = losing_trades['profit_pct'].min()
        else:
            avg_loss = 0
            max_loss = 0
        
        # Profit factor
        gross_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Daily returns
        if self.portfolio['daily_values']:
            daily_df = pd.DataFrame(self.portfolio['daily_values'])
            daily_df['returns'] = daily_df['value'].pct_change()
            
            # Sharpe ratio
            if len(daily_df) > 1 and daily_df['returns'].std() > 0:
                sharpe = (daily_df['returns'].mean() / daily_df['returns'].std()) * np.sqrt(252)
            else:
                sharpe = 0
            
            # Max drawdown
            daily_df['cummax'] = daily_df['value'].cummax()
            daily_df['drawdown'] = (daily_df['cummax'] - daily_df['value']) / daily_df['cummax']
            max_drawdown = daily_df['drawdown'].max() * 100
        else:
            sharpe = 0
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'avg_profit_pct': avg_profit_pct,
            'avg_win': avg_win,
            'max_win': max_win,
            'avg_loss': avg_loss,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    async def stop(self):
        """Stop paper trading"""
        self.is_running = False
        self.save_state()
        
        if self.algolab_socket:
            self.algolab_socket.disconnect()
        
        if self.telegram_bot:
            self.telegram_bot.stop()
        
        logger.info("Paper trading stopped")


# Command-line interface
async def main():
    """Main entry point for paper trading"""
    paper_trader = PaperTradingModule()
    
    # Try to load previous state
    paper_trader.load_state()
    
    # Initialize connections
    success = await paper_trader.initialize()
    if not success:
        logger.error("Failed to initialize paper trading module")
        return
    
    # Print initial status
    print("\n" + "="*80)
    print("PAPER TRADING MODULE - Dynamic Portfolio Optimizer")
    print("="*80)
    print("Commands:")
    print("  status  - Show portfolio status")
    print("  start   - Start auto trading")
    print("  stop    - Stop auto trading")
    print("  trades  - Show trade history")
    print("  perf    - Show performance metrics")
    print("  save    - Save current state")
    print("  telegram- Toggle Telegram notifications")
    print("  confirm - Toggle trade confirmations")
    print("  quit    - Exit program")
    print("="*80)
    
    # Start trading loop in background
    trading_task = asyncio.create_task(paper_trader.trading_loop())
    
    # Command loop
    try:
        while True:
            cmd = input("\nCommand: ").strip().lower()
            
            if cmd == "quit":
                break
                
            elif cmd == "status":
                status = paper_trader.get_portfolio_status()
                print(f"\nPortfolio Value: ${status['portfolio_value']:,.2f}")
                print(f"Total Return: ${status['total_return']:,.2f} ({status['total_return_pct']:.2f}%)")
                print(f"Cash: ${status['cash']:,.2f}")
                print(f"Positions: {status['num_positions']}")
                print(f"Total Trades: {status['total_trades']}")
                print(f"Win Rate: {status['win_rate']:.1f}%")
                
                if status['positions']:
                    print("\nCurrent Positions:")
                    for pos in status['positions']:
                        print(f"  {pos['symbol']}: {pos['shares']} @ ${pos['entry_price']:.2f} "
                              f"-> ${pos['current_price']:.2f} "
                              f"({pos['profit_pct']:+.2f}%) "
                              f"[{pos['holding_days']}d]")
                
            elif cmd == "start":
                paper_trader.auto_trade_enabled = True
                print("Auto trading started")
                
            elif cmd == "stop":
                paper_trader.auto_trade_enabled = False
                print("Auto trading stopped")
                
            elif cmd == "trades":
                trades_df = paper_trader.get_trade_history()
                if not trades_df.empty:
                    print("\nRecent Trades:")
                    print(trades_df.tail(10).to_string())
                else:
                    print("No trades yet")
                    
            elif cmd == "perf":
                metrics = paper_trader.get_performance_metrics()
                if metrics:
                    print("\nPerformance Metrics:")
                    print(f"Total Trades: {metrics['total_trades']}")
                    print(f"Win Rate: {metrics['win_rate']:.1f}%")
                    print(f"Avg Profit: ${metrics['avg_profit']:.2f} ({metrics['avg_profit_pct']:.2f}%)")
                    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
                else:
                    print("No trades to analyze")
                    
            elif cmd == "save":
                paper_trader.save_state()
                print("State saved")
                
            elif cmd == "telegram":
                paper_trader.telegram_notifications = not paper_trader.telegram_notifications
                status = "enabled" if paper_trader.telegram_notifications else "disabled"
                print(f"Telegram notifications {status}")
                if paper_trader.telegram_bot:
                    telegram_msg = f"Telegram notifications {escape_markdown_v1(status)}"
                    await paper_trader.telegram_bot.send_notification(
                        telegram_msg,
                        "info"
                    )
                
            elif cmd == "confirm":
                paper_trader.require_confirmation = not paper_trader.require_confirmation
                status = "enabled" if paper_trader.require_confirmation else "disabled"
                print(f"Trade confirmations {status}")
                if paper_trader.telegram_bot:
                    confirm_msg = f"Trade confirmations {escape_markdown_v1(status)}"
                    await paper_trader.telegram_bot.send_notification(
                        confirm_msg,
                        "info"
                    )
                
            else:
                print("Unknown command")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Stop trading
        await paper_trader.stop()
        trading_task.cancel()
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        status = paper_trader.get_portfolio_status()
        print(f"Final Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Total Return: ${status['total_return']:,.2f} ({status['total_return_pct']:.2f}%)")
        
        metrics = paper_trader.get_performance_metrics()
        if metrics:
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.1f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        print("="*80)


if __name__ == "__main__":
    asyncio.run(main())