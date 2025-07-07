#!/usr/bin/env python3
"""
Portfolio Manager for Paper Trading
Handles portfolio state, positions, and capital management
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Position:
    """Represents a single stock position"""
    
    def __init__(self, symbol: str, entry_price: float, shares: int, 
                 entry_date: datetime, strategy: str):
        self.symbol = symbol
        self.entry_price = entry_price
        self.shares = shares
        self.entry_date = entry_date
        self.strategy = strategy
        self.current_price = entry_price
        self.exit_price = None
        self.exit_date = None
        self.stop_loss = entry_price * 0.95  # %5 stop loss
        self.take_profit = entry_price * 1.15  # %15 take profit
        self.trailing_stop = None
        self.peak_price = entry_price
        
    def update_price(self, new_price: float):
        """Update current price and trailing stop"""
        self.current_price = new_price
        
        # Update peak price for trailing stop
        if new_price > self.peak_price:
            self.peak_price = new_price
            # Update trailing stop (%3 below peak)
            self.trailing_stop = self.peak_price * 0.97
        
        # Check if trailing stop is higher than initial stop loss
        if self.trailing_stop and self.trailing_stop > self.stop_loss:
            self.stop_loss = self.trailing_stop
    
    def get_value(self) -> float:
        """Get current position value"""
        return self.current_price * self.shares
    
    def get_pnl(self) -> float:
        """Get profit/loss"""
        return (self.current_price - self.entry_price) * self.shares
    
    def get_pnl_percentage(self) -> float:
        """Get profit/loss percentage"""
        return ((self.current_price / self.entry_price) - 1) * 100
    
    def should_exit(self) -> Tuple[bool, str]:
        """Check if position should be exited"""
        # Stop loss hit
        if self.current_price <= self.stop_loss:
            return True, "stop_loss"
        
        # Take profit hit
        if self.current_price >= self.take_profit:
            return True, "take_profit"
        
        # Position too old (>30 days)
        if (datetime.now() - self.entry_date).days > 30:
            return True, "time_limit"
        
        return False, ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'shares': self.shares,
            'entry_date': self.entry_date.isoformat(),
            'strategy': self.strategy,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'peak_price': self.peak_price,
            'pnl': self.get_pnl(),
            'pnl_percentage': self.get_pnl_percentage()
        }


class Portfolio:
    """Manages a trading portfolio"""
    
    def __init__(self, name: str, initial_capital: float = 50000, 
                 max_positions: int = 10, position_size_pct: float = 0.15):
        self.name = name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct  # Max 15% per position
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.transaction_history = []
        self.daily_values = []
        self.created_date = datetime.now()
        
        # Risk management
        self.max_portfolio_heat = 0.25  # Max 25% portfolio risk
        self.max_correlation = 0.7  # Max correlation between positions
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0
        
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(pos.get_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_portfolio_heat(self) -> float:
        """Calculate current portfolio risk (heat)"""
        if not self.positions:
            return 0
        
        total_risk = 0
        portfolio_value = self.get_portfolio_value()
        
        for pos in self.positions.values():
            # Risk = difference between current price and stop loss
            risk_per_share = pos.current_price - pos.stop_loss
            position_risk = risk_per_share * pos.shares
            total_risk += position_risk
        
        return total_risk / portfolio_value
    
    def can_open_position(self, symbol: str, price: float) -> Tuple[bool, str]:
        """Check if new position can be opened"""
        # Check if already have position in this symbol
        if symbol in self.positions:
            return False, "Already have position in this symbol"
        
        # Check max positions
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check available cash
        position_size = self.calculate_position_size(price)
        required_cash = position_size * price
        
        if required_cash > self.cash:
            return False, f"Insufficient cash. Required: {required_cash:.2f}, Available: {self.cash:.2f}"
        
        # Check portfolio heat
        current_heat = self.get_portfolio_heat()
        if current_heat > self.max_portfolio_heat * 0.8:  # 80% of max heat
            return False, f"Portfolio heat too high: {current_heat:.2%}"
        
        return True, "OK"
    
    def calculate_position_size(self, price: float) -> int:
        """Calculate number of shares to buy"""
        portfolio_value = self.get_portfolio_value()
        position_value = portfolio_value * self.position_size_pct
        
        # Adjust for risk
        risk_adjustment = 1.0
        current_heat = self.get_portfolio_heat()
        if current_heat > 0.15:  # If heat > 15%, reduce position size
            risk_adjustment = 0.7
        
        shares = int((position_value * risk_adjustment) / price)
        
        # Minimum 1 share, maximum to not exceed position limit
        return max(1, min(shares, int(self.cash / price)))
    
    def open_position(self, symbol: str, price: float, signal_date: datetime, 
                     strategy: str) -> Optional[Position]:
        """Open a new position"""
        can_open, reason = self.can_open_position(symbol, price)
        if not can_open:
            logger.warning(f"Cannot open position in {symbol}: {reason}")
            return None
        
        shares = self.calculate_position_size(price)
        cost = shares * price * 1.002  # Include 0.2% commission
        
        if cost > self.cash:
            shares = int(self.cash / (price * 1.002))
            cost = shares * price * 1.002
        
        # Create position
        position = Position(symbol, price, shares, signal_date, strategy)
        self.positions[symbol] = position
        self.cash -= cost
        
        # Record transaction
        self.transaction_history.append({
            'date': signal_date.isoformat(),
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'cost': cost,
            'strategy': strategy,
            'portfolio_value': self.get_portfolio_value()
        })
        
        logger.info(f"Opened position: {symbol} - {shares} shares @ {price:.2f}")
        return position
    
    def close_position(self, symbol: str, price: float, reason: str = "signal") -> bool:
        """Close an existing position"""
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        position.exit_price = price
        position.exit_date = datetime.now()
        
        # Calculate proceeds (minus commission)
        proceeds = position.shares * price * 0.998  # 0.2% commission
        self.cash += proceeds
        
        # Update stats
        pnl = position.get_pnl()
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        
        # Record transaction
        self.transaction_history.append({
            'date': position.exit_date.isoformat(),
            'type': 'SELL',
            'symbol': symbol,
            'shares': position.shares,
            'price': price,
            'proceeds': proceeds,
            'pnl': pnl,
            'pnl_pct': position.get_pnl_percentage(),
            'reason': reason,
            'hold_days': (position.exit_date - position.entry_date).days,
            'portfolio_value': self.get_portfolio_value()
        })
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(f"Closed position: {symbol} - PnL: {pnl:.2f} ({position.get_pnl_percentage():.1f}%) - Reason: {reason}")
        return True
    
    def update_positions(self, price_data: Dict[str, float]):
        """Update all position prices and check exits"""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol in price_data:
                # Update price
                position.update_price(price_data[symbol])
                
                # Check if should exit
                should_exit, reason = position.should_exit()
                if should_exit:
                    positions_to_close.append((symbol, reason))
        
        # Close positions that hit exit criteria
        for symbol, reason in positions_to_close:
            self.close_position(symbol, self.positions[symbol].current_price, reason)
    
    def rebalance_portfolio(self, new_signals: Dict[str, float], current_prices: Dict[str, float]):
        """Rebalance portfolio based on new signals"""
        # Update existing positions
        self.update_positions(current_prices)
        
        # If at max positions and have new signals, close weakest positions
        if len(self.positions) >= self.max_positions and new_signals:
            # Sort positions by performance
            positions_by_performance = sorted(
                self.positions.items(), 
                key=lambda x: x[1].get_pnl_percentage()
            )
            
            # Close bottom 20% to make room
            num_to_close = min(2, len(positions_by_performance) // 5)
            for i in range(num_to_close):
                symbol = positions_by_performance[i][0]
                if symbol in current_prices:
                    self.close_position(symbol, current_prices[symbol], "rebalance")
        
        # Open new positions
        for symbol, price in new_signals.items():
            if symbol not in self.positions:
                self.open_position(symbol, price, datetime.now(), self.name)
    
    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics"""
        portfolio_value = self.get_portfolio_value()
        total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate daily returns for Sharpe ratio
        if len(self.daily_values) > 1:
            daily_returns = pd.Series(self.daily_values).pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
            
            # Max drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_pnl': self.total_pnl,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'num_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_heat': self.get_portfolio_heat() * 100
        }
    
    def save_state(self, filepath: str):
        """Save portfolio state to JSON"""
        state = {
            'name': self.name,
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'created_date': self.created_date.isoformat(),
            'positions': {sym: pos.to_dict() for sym, pos in self.positions.items()},
            'performance': self.get_performance_metrics(),
            'transaction_history': self.transaction_history[-100:],  # Last 100 transactions
            'daily_values': self.daily_values[-252:]  # Last year
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load portfolio state from JSON"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.cash = state['cash']
        self.transaction_history = state.get('transaction_history', [])
        self.daily_values = state.get('daily_values', [])
        
        # Recreate positions
        self.positions = {}
        for sym, pos_data in state.get('positions', {}).items():
            position = Position(
                sym, 
                pos_data['entry_price'],
                pos_data['shares'],
                datetime.fromisoformat(pos_data['entry_date']),
                pos_data['strategy']
            )
            position.current_price = pos_data['current_price']
            position.stop_loss = pos_data['stop_loss']
            position.take_profit = pos_data['take_profit']
            position.peak_price = pos_data['peak_price']
            self.positions[sym] = position


class PortfolioManager:
    """Manages multiple portfolios for different strategies"""
    
    def __init__(self, data_dir: str = "paper_trading/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.portfolios: Dict[str, Portfolio] = {}
        
        # Load existing portfolios
        self.load_all_portfolios()
    
    def create_portfolio(self, name: str, initial_capital: float = 50000) -> Portfolio:
        """Create a new portfolio"""
        if name in self.portfolios:
            logger.warning(f"Portfolio {name} already exists")
            return self.portfolios[name]
        
        portfolio = Portfolio(name, initial_capital)
        self.portfolios[name] = portfolio
        
        # Save initial state
        portfolio.save_state(self.data_dir / f"{name}_portfolio.json")
        
        logger.info(f"Created portfolio: {name} with {initial_capital} TL")
        return portfolio
    
    def get_portfolio(self, name: str) -> Optional[Portfolio]:
        """Get portfolio by name"""
        return self.portfolios.get(name)
    
    def update_all_portfolios(self, price_data: Dict[str, float]):
        """Update all portfolios with new prices"""
        for name, portfolio in self.portfolios.items():
            portfolio.update_positions(price_data)
            
            # Record daily value
            portfolio.daily_values.append(portfolio.get_portfolio_value())
            
            # Save state
            portfolio.save_state(self.data_dir / f"{name}_portfolio.json")
    
    def load_all_portfolios(self):
        """Load all portfolios from disk"""
        for filepath in self.data_dir.glob("*_portfolio.json"):
            try:
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                portfolio = Portfolio(state['name'], state['initial_capital'])
                portfolio.load_state(filepath)
                self.portfolios[state['name']] = portfolio
                
                logger.info(f"Loaded portfolio: {state['name']}")
            except Exception as e:
                logger.error(f"Error loading portfolio from {filepath}: {e}")
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all portfolios"""
        summaries = []
        
        for name, portfolio in self.portfolios.items():
            metrics = portfolio.get_performance_metrics()
            metrics['name'] = name
            summaries.append(metrics)
        
        return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Test portfolio manager
    pm = PortfolioManager()
    
    # Create portfolios for each strategy
    aggressive = pm.create_portfolio("aggressive", 50000)
    balanced = pm.create_portfolio("balanced", 50000)
    conservative = pm.create_portfolio("conservative", 50000)
    
    # Test opening positions
    test_prices = {'GARAN': 45.5, 'AKBNK': 32.8, 'THYAO': 285.4}
    
    # Open some positions
    aggressive.open_position('GARAN', 45.5, datetime.now(), 'aggressive')
    aggressive.open_position('AKBNK', 32.8, datetime.now(), 'aggressive')
    
    # Update prices
    test_prices['GARAN'] = 47.2  # Price went up
    test_prices['AKBNK'] = 31.5  # Price went down
    
    pm.update_all_portfolios(test_prices)
    
    # Show summary
    print("\nPortfolio Summary:")
    print(pm.get_summary())