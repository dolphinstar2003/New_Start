"""
Risk Management System
Implements God Mode risk rules for trading system
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import RISK_MANAGEMENT, SACRED_SYMBOLS


class RiskManager:
    """Risk management system following God Mode rules"""
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize risk manager
        
        Args:
            initial_capital: Initial trading capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # God Mode risk parameters
        self.max_position_percent = RISK_MANAGEMENT['max_position_percent']  # 10%
        self.max_daily_loss_percent = RISK_MANAGEMENT['max_daily_loss_percent']  # 5%
        self.stop_loss_mandatory = RISK_MANAGEMENT['stop_loss_mandatory']  # True
        self.max_open_positions = RISK_MANAGEMENT['max_open_positions']  # 5
        self.min_trades_per_day = RISK_MANAGEMENT['min_trades_per_day']  # 3
        self.target_win_rate = RISK_MANAGEMENT['target_win_rate']  # 0.55
        self.target_profit_factor = RISK_MANAGEMENT['target_profit_factor']  # 1.5
        self.monthly_target_percent = RISK_MANAGEMENT['monthly_target_percent']  # 8%
        self.max_drawdown_percent = RISK_MANAGEMENT['max_drawdown_percent']  # 15%
        self.target_sharpe_ratio = RISK_MANAGEMENT['target_sharpe_ratio']  # 1.5
        
        # Track trading state
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.max_capital_achieved = initial_capital
        self.daily_trades_history = []
        self.pnl_history = []
        
        # Risk alerts
        self.risk_alerts = []
        
        logger.info(f"Risk Manager initialized with ${initial_capital:,.2f} capital")
        logger.info(f"Max position: {self.max_position_percent}%, Max daily loss: {self.max_daily_loss_percent}%")
    
    def calculate_position_size(self, signal_strength: str, confidence: float,
                              current_price: float, stop_loss_price: float = None) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            signal_strength: Signal strength (STRONG_BUY, WEAK_BUY, etc.)
            confidence: Signal confidence (0-1)
            current_price: Current asset price
            stop_loss_price: Stop loss price
            
        Returns:
            Dictionary with position sizing information
        """
        # Base position size (percentage of capital)
        base_position_percent = self.max_position_percent / 100.0  # Convert to decimal
        
        # Adjust position size based on signal strength
        strength_multipliers = {
            'STRONG_BUY': 1.0,      # Full position size
            'WEAK_BUY': 0.6,        # Reduced position size
            'STRONG_SELL': 1.0,     # Full position size for shorts
            'WEAK_SELL': 0.6,       # Reduced position size for shorts
            'HOLD': 0.0             # No position
        }
        
        strength_multiplier = strength_multipliers.get(signal_strength, 0.0)
        
        # Adjust position size based on confidence
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0 range
        
        # Calculate final position percentage
        final_position_percent = base_position_percent * strength_multiplier * confidence_multiplier
        
        # Calculate dollar amount
        position_value = self.current_capital * final_position_percent
        
        # Calculate number of shares
        shares = int(position_value / current_price) if current_price > 0 else 0
        actual_position_value = shares * current_price
        
        # Calculate stop loss risk if provided
        risk_per_share = 0.0
        max_loss_amount = 0.0
        
        if stop_loss_price and current_price > 0:
            if signal_strength in ['STRONG_BUY', 'WEAK_BUY']:
                # Long position - stop loss below current price
                risk_per_share = max(0, current_price - stop_loss_price)
            else:
                # Short position - stop loss above current price
                risk_per_share = max(0, stop_loss_price - current_price)
            
            max_loss_amount = shares * risk_per_share
        
        # Validate position size doesn't exceed daily loss limit
        remaining_daily_loss_capacity = (self.max_daily_loss_percent / 100.0) * self.initial_capital - abs(self.daily_pnl)
        
        if max_loss_amount > remaining_daily_loss_capacity:
            # Reduce position size to fit within daily loss limit
            if risk_per_share > 0:
                max_shares = int(remaining_daily_loss_capacity / risk_per_share)
                shares = min(shares, max_shares)
                actual_position_value = shares * current_price
                max_loss_amount = shares * risk_per_share
        
        return {
            'shares': shares,
            'position_value': actual_position_value,
            'position_percent': (actual_position_value / self.current_capital) * 100,
            'max_loss_amount': max_loss_amount,
            'risk_per_share': risk_per_share,
            'stop_loss_price': stop_loss_price,
            'strength_multiplier': strength_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'can_trade': shares > 0 and len(self.open_positions) < self.max_open_positions
        }
    
    def validate_trade(self, symbol: str, signal_strength: str, position_size_info: Dict) -> Dict[str, Any]:
        """
        Validate if trade can be executed based on risk rules
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength
            position_size_info: Position sizing information
            
        Returns:
            Validation result with approval status and reasons
        """
        validation = {
            'approved': False,
            'reasons': [],
            'warnings': [],
            'risk_level': 'HIGH'
        }
        
        # Check if position size is valid
        if position_size_info['shares'] <= 0:
            validation['reasons'].append("Position size too small")
            return validation
        
        # Check maximum open positions
        if len(self.open_positions) >= self.max_open_positions:
            validation['reasons'].append(f"Maximum open positions reached ({self.max_open_positions})")
            return validation
        
        # Check if symbol already has open position
        if symbol in self.open_positions:
            validation['reasons'].append(f"Position already open for {symbol}")
            return validation
        
        # Check daily loss limit
        potential_daily_loss = abs(self.daily_pnl) + position_size_info['max_loss_amount']
        daily_loss_limit = (self.max_daily_loss_percent / 100.0) * self.initial_capital
        
        if potential_daily_loss > daily_loss_limit:
            validation['reasons'].append(f"Would exceed daily loss limit (${daily_loss_limit:,.2f})")
            return validation
        
        # Check position size limit
        if position_size_info['position_percent'] > self.max_position_percent:
            validation['reasons'].append(f"Position size exceeds limit ({self.max_position_percent}%)")
            return validation
        
        # Check stop loss requirement
        if self.stop_loss_mandatory and not position_size_info['stop_loss_price']:
            validation['reasons'].append("Stop loss is mandatory but not provided")
            return validation
        
        # Check current drawdown
        current_drawdown = (self.max_capital_achieved - self.current_capital) / self.max_capital_achieved
        if current_drawdown > (self.max_drawdown_percent / 100.0):
            validation['reasons'].append(f"Maximum drawdown exceeded ({self.max_drawdown_percent}%)")
            return validation
        
        # All checks passed
        validation['approved'] = True
        
        # Determine risk level
        if position_size_info['position_percent'] > 7:
            validation['risk_level'] = 'HIGH'
            validation['warnings'].append("High position size")
        elif position_size_info['position_percent'] > 4:
            validation['risk_level'] = 'MEDIUM'
        else:
            validation['risk_level'] = 'LOW'
        
        # Check signal strength
        if signal_strength in ['WEAK_BUY', 'WEAK_SELL']:
            validation['warnings'].append("Weak signal strength")
        
        return validation
    
    def open_position(self, symbol: str, signal_strength: str, shares: int,
                     entry_price: float, stop_loss_price: float = None,
                     take_profit_price: float = None, timestamp: datetime = None) -> bool:
        """
        Open a new trading position
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength
            shares: Number of shares
            entry_price: Entry price
            stop_loss_price: Stop loss price
            take_profit_price: Take profit price
            timestamp: Trade timestamp
            
        Returns:
            True if position opened successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Validate position
        position_size_info = self.calculate_position_size(signal_strength, 0.8, entry_price, stop_loss_price)
        validation = self.validate_trade(symbol, signal_strength, position_size_info)
        
        if not validation['approved']:
            logger.warning(f"Cannot open position for {symbol}: {', '.join(validation['reasons'])}")
            return False
        
        # Create position record
        position = {
            'symbol': symbol,
            'signal_strength': signal_strength,
            'shares': shares,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'position_value': shares * entry_price,
            'unrealized_pnl': 0.0,
            'risk_level': validation['risk_level']
        }
        
        # Add position to tracking
        self.open_positions[symbol] = position
        self.trades_today += 1
        
        # Update capital allocation
        position_value = shares * entry_price
        
        logger.info(f"✅ Opened {signal_strength} position: {symbol} x{shares} @ ${entry_price:.2f}")
        logger.info(f"   Position value: ${position_value:,.2f}, Risk level: {validation['risk_level']}")
        
        return True
    
    def close_position(self, symbol: str, exit_price: float, 
                      reason: str = "Manual", timestamp: datetime = None) -> Dict[str, Any]:
        """
        Close an existing position
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
            timestamp: Exit timestamp
            
        Returns:
            Position closing information
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if symbol not in self.open_positions:
            logger.warning(f"No open position found for {symbol}")
            return {}
        
        position = self.open_positions[symbol]
        
        # Calculate P&L
        shares = position['shares']
        entry_price = position['entry_price']
        
        # Determine if long or short position
        if position['signal_strength'] in ['STRONG_BUY', 'WEAK_BUY']:
            # Long position
            pnl = shares * (exit_price - entry_price)
        else:
            # Short position
            pnl = shares * (entry_price - exit_price)
        
        # Calculate metrics
        pnl_percent = (pnl / position['position_value']) * 100
        holding_period = (timestamp - position['entry_time']).total_seconds() / 3600  # hours
        
        # Update tracking
        self.daily_pnl += pnl
        self.current_capital += pnl
        self.max_capital_achieved = max(self.max_capital_achieved, self.current_capital)
        
        # Create trade record
        trade_record = {
            'symbol': symbol,
            'signal_strength': position['signal_strength'],
            'shares': shares,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'holding_period_hours': holding_period,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'reason': reason,
            'risk_level': position['risk_level']
        }
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        logger.info(f"✅ Closed position: {symbol} - P&L: ${pnl:,.2f} ({pnl_percent:.2f}%)")
        logger.info(f"   Reason: {reason}, Holding period: {holding_period:.1f}h")
        
        return trade_record
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check if any positions hit stop losses
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            List of positions that should be closed
        """
        positions_to_close = []
        
        for symbol, position in self.open_positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            stop_loss_price = position['stop_loss_price']
            
            if not stop_loss_price:
                continue
            
            should_close = False
            
            # Check stop loss trigger
            if position['signal_strength'] in ['STRONG_BUY', 'WEAK_BUY']:
                # Long position - close if price drops below stop loss
                if current_price <= stop_loss_price:
                    should_close = True
            else:
                # Short position - close if price rises above stop loss
                if current_price >= stop_loss_price:
                    should_close = True
            
            if should_close:
                positions_to_close.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'stop_loss_price': stop_loss_price,
                    'reason': 'Stop Loss'
                })
        
        return positions_to_close
    
    def check_take_profits(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check if any positions hit take profit levels
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            List of positions that should be closed
        """
        positions_to_close = []
        
        for symbol, position in self.open_positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            take_profit_price = position['take_profit_price']
            
            if not take_profit_price:
                continue
            
            should_close = False
            
            # Check take profit trigger
            if position['signal_strength'] in ['STRONG_BUY', 'WEAK_BUY']:
                # Long position - close if price rises above take profit
                if current_price >= take_profit_price:
                    should_close = True
            else:
                # Short position - close if price drops below take profit
                if current_price <= take_profit_price:
                    should_close = True
            
            if should_close:
                positions_to_close.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'take_profit_price': take_profit_price,
                    'reason': 'Take Profit'
                })
        
        return positions_to_close
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]) -> None:
        """Update unrealized P&L for open positions"""
        for symbol, position in self.open_positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                shares = position['shares']
                entry_price = position['entry_price']
                
                if position['signal_strength'] in ['STRONG_BUY', 'WEAK_BUY']:
                    # Long position
                    unrealized_pnl = shares * (current_price - entry_price)
                else:
                    # Short position
                    unrealized_pnl = shares * (entry_price - current_price)
                
                position['unrealized_pnl'] = unrealized_pnl
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        total_position_value = sum(pos['position_value'] for pos in self.open_positions.values())
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.open_positions.values())
        
        current_drawdown = (self.max_capital_achieved - self.current_capital) / self.max_capital_achieved * 100
        
        daily_loss_limit = (self.max_daily_loss_percent / 100.0) * self.initial_capital
        remaining_daily_capacity = daily_loss_limit - abs(self.daily_pnl)
        
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_return_percent': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': (self.daily_pnl / self.initial_capital) * 100,
            'remaining_daily_capacity': remaining_daily_capacity,
            'open_positions_count': len(self.open_positions),
            'max_open_positions': self.max_open_positions,
            'total_position_value': total_position_value,
            'position_utilization_percent': (total_position_value / self.current_capital) * 100,
            'total_unrealized_pnl': total_unrealized_pnl,
            'current_drawdown_percent': current_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'trades_today': self.trades_today,
            'min_trades_per_day': self.min_trades_per_day,
            'risk_alerts': self.risk_alerts
        }
    
    def reset_daily_metrics(self) -> None:
        """Reset daily tracking metrics"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.risk_alerts = []
        logger.info("Daily risk metrics reset")
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of open positions"""
        if not self.open_positions:
            return pd.DataFrame()
        
        positions_data = []
        for symbol, position in self.open_positions.items():
            positions_data.append({
                'symbol': symbol,
                'signal_strength': position['signal_strength'],
                'shares': position['shares'],
                'entry_price': position['entry_price'],
                'position_value': position['position_value'],
                'unrealized_pnl': position['unrealized_pnl'],
                'stop_loss_price': position['stop_loss_price'],
                'take_profit_price': position['take_profit_price'],
                'risk_level': position['risk_level'],
                'entry_time': position['entry_time']
            })
        
        return pd.DataFrame(positions_data)
    
    def can_open_position(self, portfolio_value: float) -> bool:
        """
        Check if we can open a new position based on risk limits
        
        Args:
            portfolio_value: Current portfolio value
            
        Returns:
            True if position can be opened
        """
        # Check max open positions
        if len(self.open_positions) >= self.max_open_positions:
            logger.debug(f"Cannot open position: max positions reached ({self.max_open_positions})")
            return False
        
        # Check daily loss limit
        daily_loss_limit = (self.max_daily_loss_percent / 100.0) * self.initial_capital
        if abs(self.daily_pnl) >= daily_loss_limit:
            logger.debug(f"Cannot open position: daily loss limit reached")
            return False
        
        # Check drawdown limit
        current_drawdown = (self.max_capital_achieved - portfolio_value) / self.max_capital_achieved
        if current_drawdown >= (self.max_drawdown_percent / 100.0):
            logger.debug(f"Cannot open position: max drawdown reached ({current_drawdown:.1%})")
            return False
        
        return True
    
    def calculate_position_size(self, portfolio_value: float, current_price: float, 
                               volatility: float = 0.02) -> float:
        """
        Calculate position size based on portfolio value and volatility
        
        Args:
            portfolio_value: Current portfolio value
            current_price: Current asset price
            volatility: Asset volatility (default 2%)
            
        Returns:
            Position size in currency
        """
        # Base position size as percentage of portfolio
        base_size = portfolio_value * (self.max_position_percent / 100.0)
        
        # Adjust for volatility (lower size for higher volatility)
        volatility_adjustment = 1.0 / (1.0 + volatility * 10)
        
        # Final position size
        position_size = base_size * volatility_adjustment
        
        return min(position_size, portfolio_value * 0.1)  # Max 10% per position
    
    def set_stop_loss(self, symbol: str, stop_price: float) -> None:
        """Set stop loss for a position"""
        if symbol in self.open_positions:
            self.open_positions[symbol]['stop_loss_price'] = stop_price
            logger.debug(f"Stop loss set for {symbol} at ${stop_price:.2f}")