"""
Enhanced Paper Trading Module with Realistic Limit Order Simulation
Integrates order book simulation for realistic execution
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
from telegram_integration import TELEGRAM_CONFIG

# Import order simulation components
from core.order_book_simulator import (
    OrderBookSimulator, Order, OrderType, OrderSide, 
    OrderStatus, TimeInForce, MarketDepth
)
from core.limit_order_manager import LimitOrderManager, OrderStrategy

logger.info("Enhanced Paper Trading Module V2 - With Realistic Order Simulation")
logger.info("="*80)

class EnhancedPaperTradingModule:
    """Enhanced paper trading with realistic limit order simulation"""
    
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
            'use_limit_orders': True,       # Use limit orders instead of market
            'limit_offset_pct': 0.005,      # Place limits 0.5% from market
        }
        
        # Portfolio state
        self.portfolio = {
            'cash': initial_capital,
            'initial_capital': initial_capital,
            'positions': {},
            'trades': [],
            'pending_orders': {},  # Now tracks actual limit orders
            'daily_order_plans': [],
            'stop_orders': {},
            'trailing_stops': {},
            'entry_dates': {},
            'peak_profits': {},
            'last_rotation_check': None,
            'daily_values': [],
            'current_prices': {},
            'last_update': None
        }
        
        # Order simulation components
        self.order_book_sim = OrderBookSimulator(commission_rate=0.0002)  # 2 basis points
        self.limit_order_manager = LimitOrderManager(self.order_book_sim)
        
        # Customize order strategy
        self.limit_order_manager.order_strategy = OrderStrategy(
            entry_offset_pct=0.005,  # 0.5% below market
            scale_in_levels=2,       # Split orders
            scale_in_spacing_pct=0.01,
            use_iceberg=True,        # Hide large orders
            adaptive_pricing=True,   # Smart pricing
            time_slice_orders=True   # TWAP for large orders
        )
        
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
        self._force_check_flag = False
        
        # Telegram bot
        self.telegram_bot = None
        
        # Performance tracking
        self.execution_metrics = {
            'orders_placed': 0,
            'orders_filled': 0,
            'orders_partial': 0,
            'orders_cancelled': 0,
            'avg_fill_time': [],
            'slippage_history': [],
            'rejection_reasons': {}
        }
        
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
                logger.info("Integrated Telegram bot initialized")
            except Exception as e:
                logger.warning(f"Telegram bot initialization failed: {e}")
                self.telegram_bot = None
                self.telegram_notifications = False
            
            logger.info("Enhanced Paper Trading Module initialized successfully")
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
        
        # 3. Technical indicators (30%)
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
        
        # 4. Order book analysis (10%)
        order_book_analysis = self.limit_order_manager.get_order_book_analysis(symbol)
        if order_book_analysis:
            # Positive imbalance (more buyers)
            if order_book_analysis.get('imbalance', 0) > 0.1:
                score += 10
            # Good liquidity
            if order_book_analysis.get('liquidity_score', 0) > 50:
                score += 5
        
        # 5. Spread analysis (10%)
        if order_book_analysis:
            spread_pct = order_book_analysis.get('spread_pct', 0.1)
            if spread_pct < 0.05:  # Tight spread
                score += 10
            elif spread_pct < 0.1:
                score += 5
        
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
            
            # Get order book info
            order_book_analysis = self.limit_order_manager.get_order_book_analysis(symbol)
            
            opportunities.append({
                'symbol': symbol,
                'score': score,
                'price': current_price,
                'volume_ratio': market_data.get('volume_ratio', 1.0),
                'in_position': in_position,
                'current_profit': current_profit,
                'momentum_1h': market_data.get('price_change_1h', 0) * 100,
                'momentum_day': market_data.get('price_change_day', 0) * 100,
                'spread_pct': order_book_analysis.get('spread_pct', 0) if order_book_analysis else 0,
                'liquidity_score': order_book_analysis.get('liquidity_score', 0) if order_book_analysis else 0,
            })
        
        # Sort by score
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        return opportunities
    
    async def create_and_execute_daily_orders(self):
        """Create and execute daily limit orders"""
        # Get opportunities
        opportunities = self.evaluate_all_opportunities()
        
        # Filter for new positions only
        new_opportunities = [o for o in opportunities if not o['in_position']]
        
        # Calculate portfolio value
        portfolio_value = self.portfolio['cash']
        for symbol, position in self.portfolio['positions'].items():
            market_data = self.market_data.get(symbol, {})
            current_price = market_data.get('last_price', position['entry_price'])
            portfolio_value += position['shares'] * current_price
        
        # Create daily order plan
        daily_plan = self.limit_order_manager.create_daily_order_plan(
            new_opportunities,
            portfolio_value,
            self.portfolio['positions']
        )
        
        # Execute the plan
        if daily_plan.orders:
            logger.info(f"Executing daily order plan with {len(daily_plan.orders)} orders")
            results = await self.limit_order_manager.execute_daily_plan(daily_plan)
            
            # Track execution metrics
            self.execution_metrics['orders_placed'] += results.get('orders_placed', 0)
            
            # Send Telegram notification
            if self.telegram_notifications and self.telegram_bot:
                summary_msg = (
                    f"📋 Daily Order Plan Executed\n"
                    f"Orders placed: {results.get('orders_placed', 0)}\n"
                    f"Total value: {format_currency(results.get('total_value', 0))}"
                )
                await self.telegram_bot.send_notification(summary_msg, "info")
    
    async def monitor_limit_orders(self):
        """Monitor and update active limit orders"""
        # Update order manager
        await self.limit_order_manager.monitor_and_update_orders()
        
        # Handle partial fills
        await self.limit_order_manager.handle_partial_fills()
        
        # Check filled orders and update positions
        for order_id, order in list(self.limit_order_manager.active_orders.items()):
            if order.status == OrderStatus.FILLED:
                # Update portfolio position
                await self._process_filled_order(order)
                
                # Remove from active orders
                del self.limit_order_manager.active_orders[order_id]
                
                # Update metrics
                self.execution_metrics['orders_filled'] += 1
                
                # Calculate fill time
                fill_time = (datetime.now() - order.created_at).seconds / 60
                self.execution_metrics['avg_fill_time'].append(fill_time)
                
                # Calculate slippage
                if order.average_fill_price and order.limit_price:
                    slippage = abs(order.average_fill_price - order.limit_price) / order.limit_price
                    self.execution_metrics['slippage_history'].append(slippage * 100)
            
            elif order.status == OrderStatus.PARTIALLY_FILLED:
                self.execution_metrics['orders_partial'] += 1
            
            elif order.status in [OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
                del self.limit_order_manager.active_orders[order_id]
                self.execution_metrics['orders_cancelled'] += 1
    
    async def _process_filled_order(self, order: Order):
        """Process a filled order and update portfolio"""
        symbol = order.symbol
        
        if order.side == OrderSide.BUY:
            # Add to positions
            if symbol in self.portfolio['positions']:
                # Add to existing position
                position = self.portfolio['positions'][symbol]
                total_shares = position['shares'] + order.filled_quantity
                total_value = (position['shares'] * position['entry_price'] + 
                             order.filled_quantity * order.average_fill_price)
                avg_price = total_value / total_shares
                
                position['shares'] = total_shares
                position['entry_price'] = avg_price
                position['entry_value'] = total_value
            else:
                # New position
                self.portfolio['positions'][symbol] = {
                    'shares': order.filled_quantity,
                    'entry_price': order.average_fill_price,
                    'entry_value': order.filled_quantity * order.average_fill_price,
                    'entry_date': order.created_at,
                    'current_value': order.filled_quantity * order.average_fill_price,
                    'opportunity_score': self.calculate_opportunity_score(
                        symbol, self.market_data.get(symbol, {})
                    )
                }
                
                # Set stop loss order
                stop_price = order.average_fill_price * (1 - self.PORTFOLIO_PARAMS['stop_loss'])
                await self._place_stop_order(symbol, order.filled_quantity, stop_price)
                
                self.portfolio['entry_dates'][symbol] = order.created_at
            
            # Update cash
            total_cost = order.filled_quantity * order.average_fill_price + order.commission
            self.portfolio['cash'] -= total_cost
            
            logger.info(f"BUY FILLED: {symbol} - {order.filled_quantity} @ "
                       f"{order.average_fill_price:.2f} (Commission: {order.commission:.2f})")
            
            # Send Telegram notification
            if self.telegram_notifications and self.telegram_bot:
                await self.telegram_bot.send_trade_notification(
                    "BUY", symbol, order.filled_quantity, 
                    order.average_fill_price, f"Limit order filled"
                )
        
        elif order.side == OrderSide.SELL:
            if symbol in self.portfolio['positions']:
                position = self.portfolio['positions'][symbol]
                exit_value = order.filled_quantity * order.average_fill_price - order.commission
                entry_value = order.filled_quantity * position['entry_price']
                profit = exit_value - entry_value
                profit_pct = (profit / entry_value) * 100
                
                # Update cash
                self.portfolio['cash'] += exit_value
                
                # Record trade
                self.portfolio['trades'].append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': datetime.now(),
                    'entry_price': position['entry_price'],
                    'exit_price': order.average_fill_price,
                    'shares': order.filled_quantity,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'commission': order.commission,
                    'reason': order.notes
                })
                
                # Update or remove position
                if order.filled_quantity >= position['shares']:
                    # Full exit
                    del self.portfolio['positions'][symbol]
                    for tracking_dict in [self.portfolio['stop_orders'], 
                                        self.portfolio['trailing_stops'], 
                                        self.portfolio['peak_profits'],
                                        self.portfolio['entry_dates']]:
                        if symbol in tracking_dict:
                            del tracking_dict[symbol]
                else:
                    # Partial exit
                    position['shares'] -= order.filled_quantity
                    position['entry_value'] = position['shares'] * position['entry_price']
                
                logger.info(f"SELL FILLED: {symbol} - {order.filled_quantity} @ "
                           f"{order.average_fill_price:.2f} | P&L: {profit:.2f} ({profit_pct:.2f}%)")
                
                # Send Telegram notification
                if self.telegram_notifications and self.telegram_bot:
                    await self.telegram_bot.send_trade_notification(
                        "SELL", symbol, order.filled_quantity,
                        order.average_fill_price, f"Limit order filled", profit
                    )
    
    async def _place_stop_order(self, symbol: str, quantity: int, stop_price: float):
        """Place a stop loss order"""
        import uuid
        
        stop_order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=quantity,
            limit_price=0,  # Will be set when triggered
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC,
            notes="Stop loss order"
        )
        
        success, message = self.order_book_sim.place_order(stop_order)
        if success:
            self.portfolio['stop_orders'][symbol] = stop_order.order_id
            logger.info(f"Stop loss placed for {symbol} at {stop_price:.2f}")
    
    async def check_positions_for_exit(self):
        """Check positions and place limit sell orders for exits"""
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
                
                # Update stop order if needed
                if symbol in self.portfolio['stop_orders']:
                    stop_order_id = self.portfolio['stop_orders'][symbol]
                    self.order_book_sim.update_order(
                        stop_order_id,
                        new_price=self.portfolio['trailing_stops'][symbol]
                    )
            
            # Check exit conditions
            should_close = False
            close_reason = ""
            limit_price = current_price
            
            # Take profit
            if profit_pct >= self.PORTFOLIO_PARAMS['take_profit']:
                should_close = True
                close_reason = "Take Profit"
                # Place limit slightly below market for quick fill
                limit_price = current_price * 0.998
            
            # Exit signal
            else:
                exit_score = self.calculate_opportunity_score(symbol, market_data)
                if exit_score < self.PORTFOLIO_PARAMS['exit_threshold']:
                    should_close = True
                    close_reason = "Exit Signal"
                    # More aggressive limit for exit signals
                    limit_price = current_price * 0.995
            
            if should_close:
                positions_to_close.append((
                    symbol, position['shares'], limit_price, close_reason
                ))
        
        # Place limit sell orders
        for symbol, shares, limit_price, reason in positions_to_close:
            await self._place_limit_sell_order(symbol, shares, limit_price, reason)
    
    async def _place_limit_sell_order(self, symbol: str, shares: int, 
                                     limit_price: float, reason: str):
        """Place a limit sell order"""
        import uuid
        
        sell_order = Order(
            order_id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=shares,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            notes=reason
        )
        
        success, message = self.order_book_sim.place_order(sell_order)
        
        if success:
            self.limit_order_manager.active_orders[sell_order.order_id] = sell_order
            logger.info(f"Limit sell order placed: {symbol} - {shares} @ {limit_price:.2f} ({reason})")
            
            # Send Telegram notification
            if self.telegram_notifications and self.telegram_bot:
                order_msg = (
                    f"📉 Sell Order Placed\n"
                    f"Symbol: {escape_markdown_v1(symbol)}\n"
                    f"Quantity: {shares}\n"
                    f"Limit: {format_currency(limit_price)}\n"
                    f"Reason: {escape_markdown_v1(reason)}"
                )
                await self.telegram_bot.send_notification(order_msg, "info")
        else:
            logger.error(f"Failed to place sell order for {symbol}: {message}")
    
    async def update_market_data(self, symbol, data):
        """Update market data and order book simulation"""
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
                'volatility': 0.02,  # Default 2% volatility
                'last_update': datetime.now()
            }
        
        market_data = self.market_data[clean_symbol]
        
        # Update based on message type
        if data.get('type') == 'price':
            old_price = market_data['last_price']
            market_data['last_price'] = data.get('price', 0)
            market_data['volume'] = data.get('volume', 0)
            market_data['last_update'] = datetime.now()
            
            # Calculate volatility
            if old_price > 0:
                price_change = abs(market_data['last_price'] - old_price) / old_price
                # Exponential moving average of volatility
                market_data['volatility'] = 0.9 * market_data['volatility'] + 0.1 * price_change
            
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
            
            # Update order book simulation
            if market_data['last_price'] > 0:
                self.order_book_sim.simulate_market_update(
                    clean_symbol,
                    market_data['last_price'],
                    market_data['volume'],
                    market_data['volatility']
                )
        
        # Update portfolio current prices
        self.portfolio['current_prices'][clean_symbol] = market_data['last_price']
        self.portfolio['last_update'] = datetime.now()
    
    async def update_market_data_via_api(self):
        """Update market data using API calls with order book simulation"""
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
                            'volatility': 0.02,
                            'price_history': [],
                            'last_update': datetime.now()
                        }
                    
                    # Update market data
                    old_price = self.market_data[symbol]['last_price']
                    self.market_data[symbol]['last_price'] = float(data.get('lastPrice', 0))
                    self.market_data[symbol]['open_price'] = float(data.get('open', 0))
                    self.market_data[symbol]['high_price'] = float(data.get('high', 0))
                    self.market_data[symbol]['low_price'] = float(data.get('low', 0))
                    self.market_data[symbol]['volume'] = int(data.get('volume', 0))
                    self.market_data[symbol]['last_update'] = datetime.now()
                    
                    # Calculate volatility from high/low
                    if self.market_data[symbol]['low_price'] > 0:
                        daily_range = (self.market_data[symbol]['high_price'] - 
                                     self.market_data[symbol]['low_price'])
                        self.market_data[symbol]['volatility'] = (
                            daily_range / self.market_data[symbol]['low_price']
                        )
                    
                    # Calculate price changes
                    if self.market_data[symbol]['open_price'] > 0:
                        self.market_data[symbol]['price_change_day'] = (
                            (self.market_data[symbol]['last_price'] - 
                             self.market_data[symbol]['open_price']) / 
                            self.market_data[symbol]['open_price']
                        )
                    
                    # Update order book simulation
                    if self.market_data[symbol]['last_price'] > 0:
                        self.order_book_sim.simulate_market_update(
                            symbol,
                            self.market_data[symbol]['last_price'],
                            self.market_data[symbol]['volume'],
                            self.market_data[symbol]['volatility']
                        )
                    
                    # Update current prices
                    self.portfolio['current_prices'][symbol] = self.market_data[symbol]['last_price']
                
                # Small delay between API calls
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error updating market data via API: {e}")
    
    async def trading_loop(self):
        """Main trading loop with limit order management"""
        logger.info("Starting enhanced paper trading loop...")
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
        last_daily_order = None
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Only trade during market hours (10:00 - 18:00 Istanbul time)
                if 10 <= current_time.hour < 18 and current_time.weekday() < 5:
                    
                    # Update market data via API if WebSocket not available
                    if not self.algolab_socket and (current_time - last_api_update).seconds >= 60:
                        await self.update_market_data_via_api()
                        last_api_update = current_time
                    
                    # Create daily orders at market open
                    if (current_time.hour == 10 and current_time.minute < 30 and
                        (last_daily_order is None or last_daily_order.date() != current_time.date())):
                        if self.auto_trade_enabled:
                            await self.create_and_execute_daily_orders()
                            last_daily_order = current_time
                    
                    # Monitor orders continuously
                    await self.monitor_limit_orders()
                    
                    # Check positions every interval or if force check requested
                    if ((current_time - last_check).seconds >= check_interval) or self._force_check_flag:
                        if self._force_check_flag:
                            logger.info("Force check requested via Telegram")
                            self._force_check_flag = False
                        
                        if self.auto_trade_enabled:
                            # Check exit conditions
                            await self.check_positions_for_exit()
                            
                            # Update portfolio value
                            await self.update_portfolio_value()
                        
                        # Send hourly updates via Telegram
                        if self.telegram_bot and current_time.minute == 0:
                            status = self.get_portfolio_status()
                            metrics = self.get_execution_metrics()
                            
                            hourly_msg = (
                                f"📊 Hourly Update\n"
                                f"Value: {format_currency(status['portfolio_value'])}\n"
                                f"Return: {format_percentage(status['total_return_pct'])}\n"
                                f"Positions: {status['num_positions']}\n"
                                f"Active Orders: {len(self.limit_order_manager.active_orders)}\n"
                                f"Fill Rate: {metrics['fill_rate']*100:.1f}%"
                            )
                            await self.telegram_bot.send_notification(
                                hourly_msg,
                                "info"
                            )
                        
                        last_check = current_time
                
                # End of day cleanup at 17:45
                if (current_time.hour == 17 and current_time.minute == 45 and
                    not hasattr(self, '_eod_cleanup_done')):
                    await self.limit_order_manager.end_of_day_cleanup()
                    self._eod_cleanup_done = True
                elif current_time.hour != 17:
                    self._eod_cleanup_done = False
                
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
                   f"Positions: {len(self.portfolio['positions'])} | "
                   f"Active Orders: {len(self.limit_order_manager.active_orders)}")
        
        # Save daily value
        self.portfolio['daily_values'].append({
            'datetime': datetime.now(),
            'value': portfolio_value,
            'cash': self.portfolio['cash'],
            'positions': len(self.portfolio['positions']),
            'active_orders': len(self.limit_order_manager.active_orders),
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
            'active_orders': len(self.limit_order_manager.active_orders),
            'last_update': self.portfolio['last_update']
        }
    
    def get_execution_metrics(self):
        """Get order execution metrics"""
        base_metrics = self.limit_order_manager.calculate_execution_metrics()
        
        # Add our tracking
        base_metrics['total_placed'] = self.execution_metrics['orders_placed']
        base_metrics['avg_slippage_bps'] = (
            np.mean(self.execution_metrics['slippage_history']) 
            if self.execution_metrics['slippage_history'] else 0
        )
        
        return base_metrics
    
    def save_state(self):
        """Save portfolio state to file"""
        state_file = DATA_DIR / 'enhanced_paper_trading_state.pkl'
        with open(state_file, 'wb') as f:
            pickle.dump({
                'portfolio': self.portfolio,
                'market_data': self.market_data,
                'indicators': self.indicators,
                'execution_metrics': self.execution_metrics,
                'order_performance': self.limit_order_manager.order_performance,
                'last_save': datetime.now()
            }, f)
        logger.info(f"Portfolio state saved to {state_file}")
    
    def load_state(self):
        """Load portfolio state from file"""
        state_file = DATA_DIR / 'enhanced_paper_trading_state.pkl'
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
                self.portfolio = state['portfolio']
                self.market_data = state['market_data']
                self.indicators = state['indicators']
                self.execution_metrics = state.get('execution_metrics', self.execution_metrics)
                if 'order_performance' in state:
                    self.limit_order_manager.order_performance = state['order_performance']
                logger.info(f"Portfolio state loaded from {state_file}")
                return True
        return False
    
    def get_trade_history(self):
        """Get detailed trade history"""
        return pd.DataFrame(self.portfolio['trades'])
    
    def get_order_history(self):
        """Get order execution history"""
        return pd.DataFrame(self.limit_order_manager.order_performance)
    
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
        
        # Commission impact
        total_commission = trades_df['commission'].sum() if 'commission' in trades_df else 0
        
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
        
        # Execution metrics
        exec_metrics = self.get_execution_metrics()
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_commission': total_commission,
            'net_profit': total_profit - total_commission,
            'avg_profit': avg_profit,
            'avg_profit_pct': avg_profit_pct,
            'avg_win': avg_win,
            'max_win': max_win,
            'avg_loss': avg_loss,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'fill_rate': exec_metrics.get('fill_rate', 0) * 100,
            'avg_slippage_bps': exec_metrics.get('avg_slippage_bps', 0),
            'avg_fill_time_min': exec_metrics.get('avg_fill_time', 0)
        }
    
    async def stop(self):
        """Stop paper trading"""
        self.is_running = False
        
        # Cancel all active orders
        await self.limit_order_manager.end_of_day_cleanup()
        
        # Save state
        self.save_state()
        
        if self.algolab_socket:
            self.algolab_socket.disconnect()
        
        if self.telegram_bot:
            self.telegram_bot.stop()
        
        logger.info("Enhanced paper trading stopped")


# Command-line interface
async def main():
    """Main entry point for enhanced paper trading"""
    paper_trader = EnhancedPaperTradingModule()
    
    # Try to load previous state
    paper_trader.load_state()
    
    # Initialize connections
    success = await paper_trader.initialize()
    if not success:
        logger.error("Failed to initialize enhanced paper trading module")
        return
    
    # Print initial status
    print("\n" + "="*80)
    print("ENHANCED PAPER TRADING MODULE V2 - With Realistic Order Simulation")
    print("="*80)
    print("Commands:")
    print("  status  - Show portfolio status")
    print("  orders  - Show active orders")
    print("  start   - Start auto trading")
    print("  stop    - Stop auto trading")
    print("  trades  - Show trade history")
    print("  exec    - Show execution metrics")
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
                print(f"Active Orders: {status['active_orders']}")
                print(f"Total Trades: {status['total_trades']}")
                print(f"Win Rate: {status['win_rate']:.1f}%")
                
                if status['positions']:
                    print("\nCurrent Positions:")
                    for pos in status['positions']:
                        print(f"  {pos['symbol']}: {pos['shares']} @ ${pos['entry_price']:.2f} "
                              f"-> ${pos['current_price']:.2f} "
                              f"({pos['profit_pct']:+.2f}%) "
                              f"[{pos['holding_days']}d]")
            
            elif cmd == "orders":
                active_orders = paper_trader.limit_order_manager.active_orders
                if active_orders:
                    print(f"\nActive Orders ({len(active_orders)}):")
                    for order_id, order in active_orders.items():
                        print(f"  {order.symbol} - {order.side.value} {order.quantity} @ "
                              f"{order.limit_price:.2f} ({order.status.value})")
                        if order.filled_quantity > 0:
                            print(f"    Filled: {order.filled_quantity}/{order.quantity}")
                else:
                    print("No active orders")
            
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
            
            elif cmd == "exec":
                metrics = paper_trader.get_execution_metrics()
                print("\nExecution Metrics:")
                print(f"Orders Placed: {metrics.get('total_placed', 0)}")
                print(f"Fill Rate: {metrics.get('fill_rate', 0)*100:.1f}%")
                print(f"Partial Fill Rate: {metrics.get('partial_fill_rate', 0)*100:.1f}%")
                print(f"Avg Fill Time: {metrics.get('avg_fill_time', 0):.1f} min")
                print(f"Avg Slippage: {metrics.get('avg_slippage', 0):.2f}%")
                print(f"Rejection Rate: {metrics.get('rejection_rate', 0)*100:.1f}%")
                    
            elif cmd == "perf":
                metrics = paper_trader.get_performance_metrics()
                if metrics:
                    print("\nPerformance Metrics:")
                    print(f"Total Trades: {metrics['total_trades']}")
                    print(f"Win Rate: {metrics['win_rate']:.1f}%")
                    print(f"Net Profit: ${metrics['net_profit']:.2f}")
                    print(f"Avg Profit: ${metrics['avg_profit']:.2f} ({metrics['avg_profit_pct']:.2f}%)")
                    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
                    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
                    print(f"Total Commission: ${metrics['total_commission']:.2f}")
                    print(f"Fill Rate: {metrics['fill_rate']:.1f}%")
                    print(f"Avg Slippage: {metrics['avg_slippage_bps']:.1f} bps")
                else:
                    print("No trades to analyze")
                    
            elif cmd == "save":
                paper_trader.save_state()
                print("State saved")
                
            elif cmd == "telegram":
                paper_trader.telegram_notifications = not paper_trader.telegram_notifications
                status = "enabled" if paper_trader.telegram_notifications else "disabled"
                print(f"Telegram notifications {status}")
                
            elif cmd == "confirm":
                paper_trader.require_confirmation = not paper_trader.require_confirmation
                status = "enabled" if paper_trader.require_confirmation else "disabled"
                print(f"Trade confirmations {status}")
                
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
            print(f"Fill Rate: {metrics['fill_rate']:.1f}%")
            print(f"Avg Slippage: {metrics['avg_slippage_bps']:.1f} bps")
        
        print("="*80)


if __name__ == "__main__":
    asyncio.run(main())