"""
Simple Paper Trader for Portfolio Strategy
Provides basic paper trading functionality
"""
from datetime import datetime
from typing import Dict, Optional
from loguru import logger
import pandas as pd


class PaperTrader:
    """Simple paper trading simulator"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.available_balance = initial_capital
        self.total_equity = initial_capital
        
        # Positions tracking
        self.positions = {}
        self.closed_trades = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0
        
        logger.info(f"Paper Trader initialized with ${initial_capital:,.2f}")
    
    def open_position(self, symbol: str, quantity: int, price: float, 
                     position_type: str = 'LONG', stop_loss: float = None, 
                     take_profit: float = None) -> bool:
        """Open a new position"""
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return False
        
        position_value = quantity * price
        if position_value > self.available_balance:
            logger.warning(f"Insufficient balance for {symbol}. Need ${position_value:.2f}, have ${self.available_balance:.2f}")
            return False
        
        self.positions[symbol] = {
            'quantity': quantity,
            'entry_price': price,
            'entry_date': datetime.now(),
            'position_type': position_type,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'current_price': price,
            'unrealized_pnl': 0,
            'position_value': position_value
        }
        
        self.available_balance -= position_value
        self.total_trades += 1
        
        logger.info(f"Opened {position_type} position: {symbol} x{quantity} @ ${price:.2f}")
        return True
    
    def close_position(self, symbol: str, price: float) -> bool:
        """Close an existing position"""
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return False
        
        position = self.positions[symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']
        
        # Calculate P&L
        if position['position_type'] == 'LONG':
            pnl = (price - entry_price) * quantity
        else:
            pnl = (entry_price - price) * quantity
        
        pnl_pct = (pnl / position['position_value']) * 100
        
        # Update balance
        exit_value = quantity * price
        self.available_balance += exit_value
        self.total_pnl += pnl
        
        # Update stats
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Record trade
        trade_record = {
            'symbol': symbol,
            'entry_date': position['entry_date'],
            'exit_date': datetime.now(),
            'entry_price': entry_price,
            'exit_price': price,
            'quantity': quantity,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'position_type': position['position_type']
        }
        self.closed_trades.append(trade_record)
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} @ ${price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        return True
    
    def update_position_price(self, symbol: str, price: float):
        """Update current price and unrealized P&L"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position['current_price'] = price
        
        # Calculate unrealized P&L
        if position['position_type'] == 'LONG':
            position['unrealized_pnl'] = (price - position['entry_price']) * position['quantity']
        else:
            position['unrealized_pnl'] = (position['entry_price'] - price) * position['quantity']
    
    def check_stop_loss_take_profit(self, symbol: str, price: float) -> Optional[str]:
        """Check if stop loss or take profit is hit"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if position['stop_loss'] and price <= position['stop_loss']:
            return 'stop_loss'
        
        if position['take_profit'] and price >= position['take_profit']:
            return 'take_profit'
        
        return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position details"""
        return self.positions.get(symbol)
    
    def get_account_stats(self) -> Dict:
        """Get account statistics"""
        # Calculate total equity
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        total_equity = self.available_balance + sum(
            pos['position_value'] for pos in self.positions.values()
        ) + unrealized_pnl
        
        # Calculate metrics
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'total_equity': total_equity,
            'available_balance': self.available_balance,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': self.total_pnl,
            'total_return_pct': ((total_equity - self.initial_capital) / self.initial_capital * 100),
            'positions_count': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate
        }
    
    def get_positions_df(self) -> pd.DataFrame:
        """Get positions as DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = []
        for symbol, pos in self.positions.items():
            positions_data.append({
                'symbol': symbol,
                'quantity': pos['quantity'],
                'entry_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'unrealized_pnl_pct': (pos['unrealized_pnl'] / pos['position_value'] * 100),
                'position_type': pos['position_type'],
                'entry_date': pos['entry_date']
            })
        
        return pd.DataFrame(positions_data)
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get closed trades as DataFrame"""
        if not self.closed_trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.closed_trades)