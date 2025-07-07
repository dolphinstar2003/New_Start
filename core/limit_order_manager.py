"""
Limit Order Management System for Paper Trading
Handles daily order placement, monitoring, and execution
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import asyncio
import uuid

from core.order_book_simulator import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce,
    OrderBookSimulator, MarketDepth
)

@dataclass
class OrderStrategy:
    """Defines order placement strategy"""
    entry_offset_pct: float = 0.005  # Place limit 0.5% below market for buys
    scale_in_levels: int = 3  # Number of scale-in orders
    scale_in_spacing_pct: float = 0.01  # 1% spacing between levels
    use_iceberg: bool = True  # Split large orders
    iceberg_show_pct: float = 0.2  # Show only 20% of total size
    adaptive_pricing: bool = True  # Adjust based on order book
    time_slice_orders: bool = True  # TWAP execution for large orders
    
@dataclass 
class DailyOrderPlan:
    """Daily order execution plan"""
    date: datetime
    orders: List[Dict]  # List of planned orders
    total_capital: float
    risk_budget: float
    executed: bool = False
    
class LimitOrderManager:
    """Manages limit order placement and execution"""
    
    def __init__(self, order_book_simulator: OrderBookSimulator):
        self.order_book = order_book_simulator
        self.order_strategy = OrderStrategy()
        
        # Order tracking
        self.active_orders: Dict[str, Order] = {}
        self.daily_plans: List[DailyOrderPlan] = []
        self.order_performance: List[Dict] = []
        
        # Execution metrics
        self.fill_rates = {}  # Symbol -> fill rate
        self.avg_slippage = {}  # Symbol -> avg slippage
        self.rejection_reasons = {}
        
        # Market hours
        self.market_open = time(10, 0)  # 10:00 AM
        self.market_close = time(18, 0)  # 6:00 PM
        self.order_cutoff = time(17, 30)  # Stop placing new orders
        
    def create_daily_order_plan(self, 
                               opportunities: List[Dict],
                               portfolio_value: float,
                               existing_positions: Dict) -> DailyOrderPlan:
        """Create order plan for the day based on opportunities"""
        plan_date = datetime.now().date()
        orders = []
        
        # Calculate available capital
        position_values = sum(pos['current_value'] for pos in existing_positions.values())
        available_capital = portfolio_value - position_values
        risk_budget = available_capital * 0.95  # Keep 5% cash reserve
        
        for opp in opportunities:
            symbol = opp['symbol']
            score = opp['score']
            current_price = opp['price']
            
            # Skip if already in position
            if symbol in existing_positions:
                continue
            
            # Skip low score opportunities
            if score < 20:
                continue
            
            # Calculate position size
            position_size = self._calculate_position_size(
                score, risk_budget, portfolio_value
            )
            
            if position_size < 1000:  # Min position size
                continue
            
            # Create order entries based on strategy
            if self.order_strategy.scale_in_levels > 1:
                # Scale-in orders
                for i in range(self.order_strategy.scale_in_levels):
                    offset = self.order_strategy.entry_offset_pct + (
                        i * self.order_strategy.scale_in_spacing_pct
                    )
                    limit_price = current_price * (1 - offset)
                    order_size = position_size / self.order_strategy.scale_in_levels
                    
                    orders.append({
                        'symbol': symbol,
                        'side': 'BUY',
                        'quantity': int(order_size / limit_price),
                        'limit_price': round(limit_price, 2),
                        'order_type': 'LIMIT',
                        'time_in_force': 'DAY',
                        'priority': score,
                        'scale_level': i + 1,
                        'strategy_notes': f"Scale-in level {i+1}/{self.order_strategy.scale_in_levels}"
                    })
            else:
                # Single order
                limit_price = current_price * (1 - self.order_strategy.entry_offset_pct)
                orders.append({
                    'symbol': symbol,
                    'side': 'BUY', 
                    'quantity': int(position_size / limit_price),
                    'limit_price': round(limit_price, 2),
                    'order_type': 'LIMIT',
                    'time_in_force': 'DAY',
                    'priority': score,
                    'strategy_notes': "Single entry"
                })
            
            risk_budget -= position_size
            if risk_budget < 1000:
                break
        
        # Sort by priority
        orders.sort(key=lambda x: x['priority'], reverse=True)
        
        plan = DailyOrderPlan(
            date=datetime.now(),
            orders=orders,
            total_capital=portfolio_value,
            risk_budget=available_capital * 0.95,
            executed=False
        )
        
        self.daily_plans.append(plan)
        logger.info(f"Created daily order plan with {len(orders)} orders")
        
        return plan
    
    def _calculate_position_size(self, score: float, available: float, 
                               portfolio_value: float) -> float:
        """Calculate position size based on score and available capital"""
        # Base size on score
        if score >= 60:
            base_pct = 0.15  # 15% for high conviction
        elif score >= 40:
            base_pct = 0.10  # 10% for medium
        else:
            base_pct = 0.05  # 5% for low
        
        position_size = portfolio_value * base_pct
        position_size = min(position_size, available * 0.3)  # Max 30% of available
        
        return position_size
    
    async def execute_daily_plan(self, plan: DailyOrderPlan) -> Dict:
        """Execute the daily order plan"""
        if plan.executed:
            logger.warning("Plan already executed")
            return {'status': 'already_executed'}
        
        results = {
            'orders_placed': 0,
            'orders_rejected': 0,
            'total_value': 0,
            'errors': []
        }
        
        # Check market hours
        current_time = datetime.now().time()
        if current_time < self.market_open or current_time > self.order_cutoff:
            logger.warning("Outside order placement hours")
            return {'status': 'outside_hours'}
        
        # Place orders with smart execution
        for order_spec in plan.orders:
            try:
                if self.order_strategy.adaptive_pricing:
                    # Adjust price based on order book
                    adjusted_price = await self._get_adaptive_price(
                        order_spec['symbol'],
                        OrderSide.BUY,
                        order_spec['limit_price']
                    )
                    order_spec['limit_price'] = adjusted_price
                
                # Check if we should use iceberg orders
                if (self.order_strategy.use_iceberg and 
                    order_spec['quantity'] > 1000):
                    # Split into smaller visible chunks
                    success = await self._place_iceberg_order(order_spec)
                else:
                    # Place regular order
                    success = await self._place_single_order(order_spec)
                
                if success:
                    results['orders_placed'] += 1
                    results['total_value'] += (
                        order_spec['quantity'] * order_spec['limit_price']
                    )
                else:
                    results['orders_rejected'] += 1
                    
            except Exception as e:
                logger.error(f"Error placing order for {order_spec['symbol']}: {e}")
                results['errors'].append(str(e))
        
        plan.executed = True
        logger.info(f"Daily plan executed: {results['orders_placed']} orders placed")
        
        return results
    
    async def _place_single_order(self, order_spec: Dict) -> bool:
        """Place a single order"""
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=order_spec['symbol'],
            side=OrderSide[order_spec['side']],
            order_type=OrderType[order_spec['order_type']],
            quantity=order_spec['quantity'],
            limit_price=order_spec['limit_price'],
            time_in_force=TimeInForce[order_spec['time_in_force']],
            notes=order_spec.get('strategy_notes', '')
        )
        
        success, message = self.order_book.place_order(order)
        
        if success:
            self.active_orders[order.order_id] = order
            logger.info(f"Order placed: {order.symbol} - {order.quantity} @ {order.limit_price}")
        else:
            logger.warning(f"Order rejected: {message}")
            self.rejection_reasons[order.symbol] = message
            
        return success
    
    async def _place_iceberg_order(self, order_spec: Dict) -> bool:
        """Place an iceberg order (hidden quantity)"""
        total_quantity = order_spec['quantity']
        show_quantity = int(total_quantity * self.order_strategy.iceberg_show_pct)
        show_quantity = max(show_quantity, 100)  # Min show size
        
        orders_placed = 0
        remaining = total_quantity
        
        while remaining > 0:
            current_size = min(show_quantity, remaining)
            
            slice_order = order_spec.copy()
            slice_order['quantity'] = current_size
            slice_order['strategy_notes'] = f"Iceberg slice: {current_size}/{total_quantity}"
            
            if await self._place_single_order(slice_order):
                orders_placed += 1
            else:
                logger.warning(f"Failed to place iceberg slice for {order_spec['symbol']}")
                break
                
            remaining -= current_size
            
            # Small delay between slices
            await asyncio.sleep(0.1)
        
        return orders_placed > 0
    
    async def _get_adaptive_price(self, symbol: str, side: OrderSide, 
                                 base_price: float) -> float:
        """Get adaptive price based on order book"""
        order_book = self.order_book.get_order_book_snapshot(symbol)
        
        if not order_book:
            return base_price
        
        # For buy orders, look at ask depth
        if side == OrderSide.BUY:
            # Find a price level with good liquidity
            total_size = 0
            for ask_price, ask_size in order_book.asks:
                total_size += ask_size
                if total_size > 5000:  # Found enough liquidity
                    # Place order slightly below this level
                    return min(base_price, ask_price - 0.01)
            
        return base_price
    
    async def monitor_and_update_orders(self):
        """Monitor active orders and update if needed"""
        current_time = datetime.now()
        
        for order_id, order in list(self.active_orders.items()):
            if not order.is_active:
                del self.active_orders[order_id]
                continue
            
            # Get current market data
            market_data = self.order_book.market_data.get(order.symbol, {})
            last_price = market_data.get('last_price', 0)
            
            if last_price == 0:
                continue
            
            # Check if we should update the order
            if self.order_strategy.adaptive_pricing:
                # Update if price moved significantly
                price_diff_pct = abs(order.limit_price - last_price) / last_price
                
                if price_diff_pct > 0.02:  # More than 2% away
                    # Cancel and replace with better price
                    if order.side == OrderSide.BUY and order.limit_price < last_price * 0.98:
                        new_price = last_price * 0.995  # Update to 0.5% below market
                        await self._update_order_price(order_id, new_price)
            
            # Check if order is too old (for day orders)
            if order.time_in_force == TimeInForce.DAY:
                order_age = (current_time - order.created_at).seconds / 3600
                
                if order_age > 6 and order.filled_quantity == 0:
                    # Cancel stale orders
                    logger.info(f"Cancelling stale order: {order_id}")
                    self.order_book.cancel_order(order_id)
    
    async def _update_order_price(self, order_id: str, new_price: float):
        """Update order price (loses time priority)"""
        success, message = self.order_book.update_order(
            order_id, 
            new_price=new_price
        )
        
        if success:
            logger.info(f"Order {order_id} updated to {new_price}")
        else:
            logger.warning(f"Failed to update order {order_id}: {message}")
    
    async def handle_partial_fills(self):
        """Handle partially filled orders"""
        for order_id, order in self.active_orders.items():
            if order.status == OrderStatus.PARTIALLY_FILLED:
                fill_pct = order.filled_quantity / order.quantity
                
                # If mostly filled, let it complete
                if fill_pct > 0.8:
                    continue
                
                # If fill rate is very low after significant time
                order_age = (datetime.now() - order.created_at).seconds / 3600
                if order_age > 2 and fill_pct < 0.2:
                    # Consider cancelling and replacing
                    remaining = order.remaining_quantity
                    
                    # Cancel current order
                    self.order_book.cancel_order(order_id)
                    
                    # Place new order at better price
                    market_data = self.order_book.market_data.get(order.symbol, {})
                    last_price = market_data.get('last_price', order.limit_price)
                    
                    new_order_spec = {
                        'symbol': order.symbol,
                        'side': order.side.value,
                        'quantity': remaining,
                        'limit_price': last_price * 0.998,  # More aggressive
                        'order_type': 'LIMIT',
                        'time_in_force': 'DAY',
                        'strategy_notes': f"Replacement for partial fill {order_id}"
                    }
                    
                    await self._place_single_order(new_order_spec)
    
    def calculate_execution_metrics(self) -> Dict:
        """Calculate order execution performance metrics"""
        metrics = {
            'total_orders': len(self.order_performance),
            'fill_rate': 0,
            'avg_fill_time': 0,
            'avg_slippage': 0,
            'rejection_rate': 0,
            'partial_fill_rate': 0
        }
        
        if not self.order_performance:
            return metrics
        
        filled_orders = [o for o in self.order_performance if o['status'] == 'FILLED']
        partial_orders = [o for o in self.order_performance if o['status'] == 'PARTIALLY_FILLED']
        rejected_orders = [o for o in self.order_performance if o['status'] == 'REJECTED']
        
        metrics['fill_rate'] = len(filled_orders) / len(self.order_performance)
        metrics['partial_fill_rate'] = len(partial_orders) / len(self.order_performance)
        metrics['rejection_rate'] = len(rejected_orders) / len(self.order_performance)
        
        # Calculate average fill time
        fill_times = []
        for order in filled_orders:
            if 'fill_time' in order and 'create_time' in order:
                fill_time = (order['fill_time'] - order['create_time']).seconds / 60
                fill_times.append(fill_time)
        
        if fill_times:
            metrics['avg_fill_time'] = np.mean(fill_times)
        
        # Calculate average slippage
        slippages = []
        for order in filled_orders:
            if 'avg_fill_price' in order and 'limit_price' in order:
                slippage = (order['avg_fill_price'] - order['limit_price']) / order['limit_price']
                slippages.append(abs(slippage))
        
        if slippages:
            metrics['avg_slippage'] = np.mean(slippages) * 100  # In percentage
        
        return metrics
    
    async def end_of_day_cleanup(self):
        """Cancel remaining day orders and prepare for next day"""
        logger.info("Running end of day cleanup")
        
        cancelled_count = 0
        for order_id, order in list(self.active_orders.items()):
            if order.time_in_force == TimeInForce.DAY and order.is_active:
                self.order_book.cancel_order(order_id)
                cancelled_count += 1
                
                # Record performance
                self.order_performance.append({
                    'order_id': order_id,
                    'symbol': order.symbol,
                    'status': order.status.value,
                    'filled_qty': order.filled_quantity,
                    'total_qty': order.quantity,
                    'limit_price': order.limit_price,
                    'avg_fill_price': order.average_fill_price,
                    'create_time': order.created_at,
                    'commission': order.commission
                })
        
        logger.info(f"Cancelled {cancelled_count} day orders")
        
        # Calculate and log daily metrics
        daily_metrics = self.calculate_execution_metrics()
        logger.info(f"Daily execution metrics: {daily_metrics}")
        
        # Clear daily data
        self.active_orders.clear()
        
    def get_order_book_analysis(self, symbol: str) -> Dict:
        """Analyze order book for a symbol"""
        order_book = self.order_book.get_order_book_snapshot(symbol)
        
        if not order_book:
            return {}
        
        analysis = {
            'spread': order_book.spread,
            'spread_pct': order_book.spread_pct,
            'best_bid': order_book.best_bid,
            'best_ask': order_book.best_ask,
            'bid_depth': sum(size for _, size in order_book.bids[:5]),
            'ask_depth': sum(size for _, size in order_book.asks[:5]),
            'imbalance': 0,
            'liquidity_score': 0
        }
        
        # Calculate imbalance
        if analysis['ask_depth'] > 0:
            analysis['imbalance'] = (
                (analysis['bid_depth'] - analysis['ask_depth']) / 
                (analysis['bid_depth'] + analysis['ask_depth'])
            )
        
        # Simple liquidity score
        total_depth = analysis['bid_depth'] + analysis['ask_depth']
        analysis['liquidity_score'] = min(100, total_depth / 100)
        
        return analysis