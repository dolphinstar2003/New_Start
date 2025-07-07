"""
Order Book and Limit Order Simulation for Paper Trading
Provides realistic order execution simulation with market microstructure
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio
from loguru import logger
import random
from collections import deque

class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    
class OrderStatus(Enum):
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TimeInForce(Enum):
    DAY = "DAY"  # Valid for the day
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

@dataclass
class Order:
    """Represents a limit order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    limit_price: float
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    created_at: datetime = field(default_factory=datetime.now)
    expire_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    average_fill_price: float = 0.0
    fills: List[Dict] = field(default_factory=list)
    commission: float = 0.0
    notes: str = ""
    
    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]

@dataclass
class MarketDepth:
    """Simulated market depth"""
    bids: List[Tuple[float, int]]  # [(price, quantity), ...]
    asks: List[Tuple[float, int]]  # [(price, quantity), ...]
    last_price: float
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0
    
    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0
    
    @property
    def spread_pct(self) -> float:
        mid = (self.best_bid + self.best_ask) / 2
        return (self.spread / mid) * 100 if mid > 0 else 0

class MarketMicrostructure:
    """Simulates realistic market microstructure behavior"""
    
    def __init__(self):
        # Market parameters
        self.base_spread_pct = 0.05  # Base spread %0.05
        self.volatility_spread_multiplier = 2.0  # Spread widens with volatility
        self.depth_levels = 10  # Number of depth levels
        self.base_depth_size = 1000  # Base lot size at each level
        self.depth_decay = 0.8  # Size decay per level
        self.tick_size = 0.01  # Minimum price increment
        
        # Liquidity parameters
        self.liquidity_cycles = {
            'morning_open': (10, 11, 1.5),    # hour_start, hour_end, liquidity_multiplier
            'midday': (11, 14, 0.7),
            'afternoon': (14, 16, 1.2),
            'close': (16, 18, 1.8)
        }
        
        # HFT and market maker simulation
        self.hft_participation = 0.3  # 30% of volume from HFT
        self.quote_flicker_rate = 0.1  # 10% chance of quote change per update
        self.fake_liquidity_pct = 0.15  # 15% of displayed liquidity is fake
        
    def get_current_liquidity_multiplier(self) -> float:
        """Get liquidity multiplier based on time of day"""
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        current_time = current_hour + current_minute / 60
        
        for period, (start, end, multiplier) in self.liquidity_cycles.items():
            if start <= current_time < end:
                return multiplier
        return 1.0
    
    def generate_order_book(self, 
                           last_price: float, 
                           volatility: float = 0.02,
                           volume_imbalance: float = 0.0) -> MarketDepth:
        """Generate realistic order book depth"""
        liquidity_mult = self.get_current_liquidity_multiplier()
        
        # Calculate dynamic spread based on volatility
        spread_pct = self.base_spread_pct * (1 + volatility * self.volatility_spread_multiplier)
        spread = last_price * spread_pct / 100
        
        # Adjust for minimum tick size
        spread = max(spread, self.tick_size * 2)
        
        # Generate bid/ask prices
        best_bid = last_price - spread / 2
        best_ask = last_price + spread / 2
        
        # Round to tick size
        best_bid = round(best_bid / self.tick_size) * self.tick_size
        best_ask = round(best_ask / self.tick_size) * self.tick_size
        
        bids = []
        asks = []
        
        # Generate depth levels
        for i in range(self.depth_levels):
            # Price levels
            bid_price = best_bid - i * self.tick_size
            ask_price = best_ask + i * self.tick_size
            
            # Size at each level (with decay and randomness)
            base_size = self.base_depth_size * (self.depth_decay ** i) * liquidity_mult
            
            # Add volume imbalance
            bid_size = int(base_size * (1 + volume_imbalance) * random.uniform(0.7, 1.3))
            ask_size = int(base_size * (1 - volume_imbalance) * random.uniform(0.7, 1.3))
            
            # Simulate HFT layering (smaller orders at better prices)
            if i < 3 and random.random() < self.hft_participation:
                bid_size = int(bid_size * 0.3)
                ask_size = int(ask_size * 0.3)
            
            bids.append((bid_price, bid_size))
            asks.append((ask_price, ask_size))
        
        # Simulate quote flickering
        if random.random() < self.quote_flicker_rate:
            # Remove some liquidity temporarily
            bids = [(p, int(q * 0.5)) for p, q in bids[:3]] + bids[3:]
            asks = [(p, int(q * 0.5)) for p, q in asks[:3]] + asks[3:]
        
        return MarketDepth(bids=bids, asks=asks, last_price=last_price)
    
    def calculate_market_impact(self, 
                               order_size: int, 
                               avg_daily_volume: int,
                               volatility: float) -> float:
        """Calculate temporary market impact of large orders"""
        # Based on Almgren-Chriss model simplified
        participation_rate = order_size / (avg_daily_volume / 390)  # per minute
        
        # Linear impact for small orders, square root for large
        if participation_rate < 0.01:
            impact_pct = participation_rate * 10 * volatility
        else:
            impact_pct = np.sqrt(participation_rate) * 5 * volatility
        
        return min(impact_pct, 0.02)  # Cap at 2%
    
    def simulate_latency(self) -> float:
        """Simulate order routing latency"""
        # Normal distribution with occasional spikes
        if random.random() < 0.05:  # 5% chance of high latency
            return random.uniform(0.5, 2.0)
        return random.gauss(0.05, 0.02)  # 50ms average

class OrderBookSimulator:
    """Main order book and execution simulator"""
    
    def __init__(self, commission_rate: float = 0.0002):  # 2 basis points
        self.microstructure = MarketMicrostructure()
        self.commission_rate = commission_rate
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.order_books: Dict[str, MarketDepth] = {}
        self.market_data: Dict[str, Dict] = {}
        
        # Price-time priority queues per symbol
        self.buy_queues: Dict[str, deque] = {}
        self.sell_queues: Dict[str, deque] = {}
        
        # Fill history for analysis
        self.fill_history: List[Dict] = []
        
        # Market anomaly parameters
        self.gap_probability = 0.02  # 2% chance of gap on new day
        self.flash_crash_probability = 0.001  # 0.1% per hour
        self.circuit_breaker_threshold = 0.07  # 7% move triggers halt
        
    def place_order(self, order: Order) -> Tuple[bool, str]:
        """Place a new order with validation"""
        # Validate order
        if order.quantity <= 0:
            return False, "Invalid quantity"
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.limit_price <= 0:
            return False, "Invalid limit price"
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price <= 0:
            return False, "Invalid stop price"
        
        # Risk checks
        if order.symbol not in self.market_data:
            return False, "Symbol not found"
        
        market_data = self.market_data[order.symbol]
        last_price = market_data.get('last_price', 0)
        
        # Fat finger protection
        if order.order_type == OrderType.LIMIT:
            price_diff_pct = abs(order.limit_price - last_price) / last_price
            if price_diff_pct > 0.1:  # 10% away from market
                return False, "Price too far from market (fat finger protection)"
        
        # Check if market is halted
        if market_data.get('halted', False):
            return False, "Market is halted"
        
        # Accept order
        order.status = OrderStatus.PENDING
        self.orders[order.order_id] = order
        
        # Add to price-time priority queue
        if order.symbol not in self.buy_queues:
            self.buy_queues[order.symbol] = deque()
            self.sell_queues[order.symbol] = deque()
        
        if order.side == OrderSide.BUY:
            self.buy_queues[order.symbol].append(order.order_id)
        else:
            self.sell_queues[order.symbol].append(order.order_id)
        
        logger.info(f"Order placed: {order.order_id} - {order.side.value} {order.quantity} "
                   f"{order.symbol} @ {order.limit_price}")
        
        return True, "Order accepted"
    
    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an order"""
        if order_id not in self.orders:
            return False, "Order not found"
        
        order = self.orders[order_id]
        
        if not order.is_active:
            return False, f"Order already {order.status.value}"
        
        # Remove from queues
        if order.side == OrderSide.BUY and order_id in self.buy_queues[order.symbol]:
            self.buy_queues[order.symbol].remove(order_id)
        elif order.side == OrderSide.SELL and order_id in self.sell_queues[order.symbol]:
            self.sell_queues[order.symbol].remove(order_id)
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order_id}")
        
        return True, "Order cancelled"
    
    def update_order(self, order_id: str, new_price: Optional[float] = None, 
                    new_quantity: Optional[int] = None) -> Tuple[bool, str]:
        """Update an existing order (loses time priority)"""
        if order_id not in self.orders:
            return False, "Order not found"
        
        order = self.orders[order_id]
        
        if not order.is_active:
            return False, f"Order already {order.status.value}"
        
        # Remove from queue (loses priority)
        if order.side == OrderSide.BUY and order_id in self.buy_queues[order.symbol]:
            self.buy_queues[order.symbol].remove(order_id)
        elif order.side == OrderSide.SELL and order_id in self.sell_queues[order.symbol]:
            self.sell_queues[order.symbol].remove(order_id)
        
        # Update order
        if new_price is not None:
            order.limit_price = new_price
        if new_quantity is not None:
            if new_quantity <= order.filled_quantity:
                return False, "New quantity less than filled quantity"
            order.quantity = new_quantity
        
        # Re-add to back of queue
        if order.side == OrderSide.BUY:
            self.buy_queues[order.symbol].append(order_id)
        else:
            self.sell_queues[order.symbol].append(order_id)
        
        logger.info(f"Order updated: {order_id} - Price: {order.limit_price}, Qty: {order.quantity}")
        
        return True, "Order updated"
    
    def simulate_market_update(self, symbol: str, last_price: float, 
                             volume: int, volatility: float = 0.02) -> MarketDepth:
        """Simulate market data update and trigger order matching"""
        # Update market data
        if symbol not in self.market_data:
            self.market_data[symbol] = {
                'last_price': last_price,
                'prev_close': last_price,
                'volume': 0,
                'avg_daily_volume': 1000000,
                'volatility': volatility,
                'halted': False,
                'gap_open': False
            }
        
        market_data = self.market_data[symbol]
        prev_price = market_data['last_price']
        
        # Check for circuit breaker
        price_change_pct = abs(last_price - market_data['prev_close']) / market_data['prev_close']
        if price_change_pct > self.circuit_breaker_threshold:
            market_data['halted'] = True
            logger.warning(f"Circuit breaker triggered for {symbol}")
            return self.order_books.get(symbol, MarketDepth([], [], last_price))
        
        # Simulate flash crash (rare)
        if random.random() < self.flash_crash_probability:
            flash_magnitude = random.uniform(0.03, 0.05)  # 3-5% drop
            last_price *= (1 - flash_magnitude)
            logger.warning(f"Flash crash in {symbol}: {flash_magnitude*100:.1f}% drop")
        
        # Update market data
        market_data['last_price'] = last_price
        market_data['volume'] += volume
        market_data['volatility'] = volatility
        
        # Generate new order book
        volume_imbalance = (last_price - prev_price) / prev_price if prev_price > 0 else 0
        order_book = self.microstructure.generate_order_book(
            last_price, volatility, volume_imbalance
        )
        self.order_books[symbol] = order_book
        
        # Process stop orders
        self._process_stop_orders(symbol, last_price)
        
        # Match orders
        self._match_orders(symbol, order_book)
        
        # Check for expired orders
        self._expire_orders(symbol)
        
        return order_book
    
    def _process_stop_orders(self, symbol: str, last_price: float):
        """Convert stop orders to limit/market orders when triggered"""
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            
            if (order.symbol != symbol or 
                order.order_type not in [OrderType.STOP, OrderType.STOP_LIMIT] or
                not order.is_active):
                continue
            
            triggered = False
            
            if order.side == OrderSide.BUY and last_price >= order.stop_price:
                triggered = True
            elif order.side == OrderSide.SELL and last_price <= order.stop_price:
                triggered = True
            
            if triggered:
                logger.info(f"Stop order triggered: {order_id} at {last_price}")
                if order.order_type == OrderType.STOP:
                    order.order_type = OrderType.MARKET
                else:
                    order.order_type = OrderType.LIMIT
    
    def _match_orders(self, symbol: str, order_book: MarketDepth):
        """Match orders against order book with realistic fill logic"""
        # Process buy orders
        for order_id in list(self.buy_queues.get(symbol, [])):
            if order_id not in self.orders:
                continue
                
            order = self.orders[order_id]
            if not order.is_active:
                continue
            
            # Market orders fill at best ask
            if order.order_type == OrderType.MARKET:
                fill_price = order_book.best_ask
            else:
                # Limit orders need price to cross
                if order.limit_price < order_book.best_ask:
                    continue
                fill_price = order.limit_price
            
            # Simulate fill probability and partial fills
            self._attempt_fill(order, fill_price, order_book.asks)
        
        # Process sell orders
        for order_id in list(self.sell_queues.get(symbol, [])):
            if order_id not in self.orders:
                continue
                
            order = self.orders[order_id]
            if not order.is_active:
                continue
            
            # Market orders fill at best bid
            if order.order_type == OrderType.MARKET:
                fill_price = order_book.best_bid
            else:
                # Limit orders need price to cross
                if order.limit_price > order_book.best_bid:
                    continue
                fill_price = order.limit_price
            
            # Simulate fill probability and partial fills
            self._attempt_fill(order, fill_price, order_book.bids)
    
    def _attempt_fill(self, order: Order, fill_price: float, 
                     depth_levels: List[Tuple[float, int]]):
        """Attempt to fill an order with realistic simulation"""
        remaining = order.remaining_quantity
        fills_this_update = []
        
        # Simulate latency
        latency = self.microstructure.simulate_latency()
        
        # Calculate fill probability based on queue position and liquidity
        for price_level, available_qty in depth_levels:
            if remaining == 0:
                break
            
            # Skip if price doesn't match our requirements
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price_level > order.limit_price:
                    continue
                if order.side == OrderSide.SELL and price_level < order.limit_price:
                    continue
            
            # Account for fake liquidity
            real_available = int(available_qty * (1 - self.microstructure.fake_liquidity_pct))
            
            # Simulate partial fills
            if order.time_in_force == TimeInForce.FOK and real_available < remaining:
                # Fill or Kill - cancel if can't fill completely
                order.status = OrderStatus.CANCELLED
                order.notes = "FOK - Insufficient liquidity"
                return
            
            # Calculate fill quantity
            fill_qty = min(remaining, real_available)
            
            # Apply market impact for large orders
            market_impact = self.microstructure.calculate_market_impact(
                fill_qty,
                self.market_data[order.symbol]['avg_daily_volume'],
                self.market_data[order.symbol]['volatility']
            )
            
            # Adjust fill price for market impact
            if order.side == OrderSide.BUY:
                adjusted_price = price_level * (1 + market_impact)
            else:
                adjusted_price = price_level * (1 - market_impact)
            
            # Simulate HFT front-running for large orders
            if fill_qty > 1000 and random.random() < self.microstructure.hft_participation:
                # HFT takes some liquidity
                fill_qty = int(fill_qty * 0.7)
                adjusted_price *= 1.001 if order.side == OrderSide.BUY else 0.999
            
            # Execute fill
            fills_this_update.append({
                'quantity': fill_qty,
                'price': adjusted_price,
                'timestamp': datetime.now() + timedelta(seconds=latency)
            })
            
            remaining -= fill_qty
            
            # IOC orders cancel remaining
            if order.time_in_force == TimeInForce.IOC:
                break
        
        # Apply fills to order
        for fill in fills_this_update:
            self._execute_fill(order, fill['quantity'], fill['price'], fill['timestamp'])
        
        # Cancel IOC remainder
        if order.time_in_force == TimeInForce.IOC and order.remaining_quantity > 0:
            order.status = OrderStatus.CANCELLED
            order.notes = "IOC - Partial fill, remainder cancelled"
    
    def _execute_fill(self, order: Order, quantity: int, price: float, 
                     timestamp: datetime):
        """Execute a fill on an order"""
        # Update order
        prev_filled = order.filled_quantity
        order.filled_quantity += quantity
        
        # Calculate weighted average price
        order.average_fill_price = (
            (prev_filled * order.average_fill_price + quantity * price) / 
            order.filled_quantity
        )
        
        # Calculate commission
        fill_value = quantity * price
        commission = fill_value * self.commission_rate
        order.commission += commission
        
        # Record fill
        fill_record = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': quantity,
            'price': price,
            'value': fill_value,
            'commission': commission,
            'timestamp': timestamp
        }
        
        order.fills.append(fill_record)
        self.fill_history.append(fill_record)
        
        # Update order status
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            # Remove from queues
            if order.side == OrderSide.BUY and order.order_id in self.buy_queues[order.symbol]:
                self.buy_queues[order.symbol].remove(order.order_id)
            elif order.side == OrderSide.SELL and order.order_id in self.sell_queues[order.symbol]:
                self.sell_queues[order.symbol].remove(order.order_id)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        logger.info(f"Fill executed: {order.order_id} - {quantity} @ {price:.2f} "
                   f"(Commission: {commission:.2f})")
    
    def _expire_orders(self, symbol: str):
        """Check and expire orders based on time in force"""
        current_time = datetime.now()
        
        for order_id in list(self.orders.keys()):
            order = self.orders[order_id]
            
            if order.symbol != symbol or not order.is_active:
                continue
            
            should_expire = False
            
            # Day orders expire at market close
            if order.time_in_force == TimeInForce.DAY:
                if current_time.hour >= 18:  # Market close
                    should_expire = True
            
            # GTD orders
            elif order.time_in_force == TimeInForce.GTD and order.expire_at:
                if current_time >= order.expire_at:
                    should_expire = True
            
            if should_expire:
                order.status = OrderStatus.EXPIRED
                # Remove from queues
                if order.side == OrderSide.BUY and order.order_id in self.buy_queues[order.symbol]:
                    self.buy_queues[order.symbol].remove(order.order_id)
                elif order.side == OrderSide.SELL and order.order_id in self.sell_queues[order.symbol]:
                    self.sell_queues[order.symbol].remove(order.order_id)
                
                logger.info(f"Order expired: {order_id}")
    
    def simulate_gap_open(self, symbol: str, gap_pct: float):
        """Simulate a gap opening"""
        if symbol in self.market_data:
            self.market_data[symbol]['gap_open'] = True
            prev_close = self.market_data[symbol]['prev_close']
            new_price = prev_close * (1 + gap_pct)
            self.market_data[symbol]['last_price'] = new_price
            logger.warning(f"Gap open in {symbol}: {gap_pct*100:.1f}% at {new_price:.2f}")
    
    def simulate_trading_halt(self, symbol: str, duration_minutes: int = 5):
        """Simulate a trading halt"""
        if symbol in self.market_data:
            self.market_data[symbol]['halted'] = True
            self.market_data[symbol]['halt_until'] = datetime.now() + timedelta(minutes=duration_minutes)
            logger.warning(f"Trading halted in {symbol} for {duration_minutes} minutes")
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders for a symbol"""
        open_orders = []
        for order in self.orders.values():
            if order.is_active:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order)
        return open_orders
    
    def get_fill_history(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get fill history as DataFrame"""
        fills = self.fill_history
        if symbol:
            fills = [f for f in fills if f['symbol'] == symbol]
        return pd.DataFrame(fills)
    
    def get_order_book_snapshot(self, symbol: str) -> Optional[MarketDepth]:
        """Get current order book snapshot"""
        return self.order_books.get(symbol)
    
    def calculate_slippage_cost(self, symbol: str, side: OrderSide, 
                               quantity: int) -> Tuple[float, float]:
        """Calculate expected slippage for a market order"""
        order_book = self.order_books.get(symbol)
        if not order_book:
            return 0, 0
        
        depth = order_book.asks if side == OrderSide.BUY else order_book.bids
        remaining = quantity
        total_cost = 0
        worst_price = 0
        
        for price, available in depth:
            fill_qty = min(remaining, available)
            total_cost += fill_qty * price
            worst_price = price
            remaining -= fill_qty
            
            if remaining == 0:
                break
        
        if remaining > 0:
            # Not enough liquidity
            return float('inf'), float('inf')
        
        avg_price = total_cost / quantity
        if side == OrderSide.BUY:
            slippage = (avg_price - order_book.best_ask) / order_book.best_ask
        else:
            slippage = (order_book.best_bid - avg_price) / order_book.best_bid
        
        return slippage, worst_price