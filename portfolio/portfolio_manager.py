"""
Portfolio Management System
Manages portfolio allocation, rebalancing, and performance tracking
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

from config.settings import SACRED_SYMBOLS, RISK_MANAGEMENT
from trading.risk_manager import RiskManager


class PortfolioManager:
    """Portfolio management system for systematic trading"""
    
    def __init__(self, initial_capital: float = 100000.0, 
                 portfolio_dir: Path = None):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Initial portfolio capital
            portfolio_dir: Directory to save portfolio data
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.portfolio_dir = portfolio_dir or Path("portfolio")
        self.portfolio_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize risk manager
        self.risk_manager = RiskManager(initial_capital)
        
        # Portfolio allocation targets (God Mode symbols)
        self.target_allocations = self._calculate_target_allocations()
        
        # Portfolio state
        self.holdings = {}  # symbol -> holding info
        self.cash_balance = initial_capital
        self.portfolio_history = []
        self.rebalance_history = []
        
        # Performance tracking
        self.daily_returns = []
        self.benchmark_returns = []  # BIST100 returns for comparison
        
        # Portfolio constraints
        self.max_single_position = 0.15  # 15% max in single stock
        self.min_cash_reserve = 0.05     # 5% minimum cash
        self.rebalance_threshold = 0.05   # 5% deviation triggers rebalance
        
        logger.info(f"Portfolio Manager initialized with ${initial_capital:,.2f}")
        logger.info(f"Target allocations: {len(self.target_allocations)} symbols")
    
    def _calculate_target_allocations(self) -> Dict[str, float]:
        """
        Calculate target allocation percentages for sacred symbols
        Based on market cap weighting with equal weight backup
        """
        # For now, use equal weight allocation
        # In production, this could be based on market cap, volatility, etc.
        
        num_symbols = len(SACRED_SYMBOLS)
        equal_weight = 0.8 / num_symbols  # 80% allocated, 20% cash reserve
        
        allocations = {}
        
        # Sector-based allocation weights
        sector_weights = {
            # Banks (higher weight due to stability)
            'GARAN': 0.06, 'AKBNK': 0.05, 'ISCTR': 0.05, 'YKBNK': 0.04,
            # Holdings (medium-high weight)
            'SAHOL': 0.055, 'KCHOL': 0.045, 'SISE': 0.04,
            # Industry (medium weight)
            'EREGL': 0.05, 'KRDMD': 0.035, 'TUPRS': 0.045,
            # Tech/Defense (medium weight)
            'ASELS': 0.045, 'THYAO': 0.05, 'TCELL': 0.04,
            # Consumer (medium weight)
            'BIMAS': 0.04, 'MGROS': 0.035, 'ULKER': 0.035,
            # Energy/Infrastructure (medium weight)
            'AKSEN': 0.04, 'ENKAI': 0.035,
            # Opportunities (lower weight, higher risk)
            'PETKM': 0.03, 'KOZAL': 0.025
        }
        
        # Normalize to ensure total is 80%
        total_weight = sum(sector_weights.values())
        target_total = 0.8  # 80% allocated
        
        for symbol in SACRED_SYMBOLS:
            if symbol in sector_weights:
                allocations[symbol] = (sector_weights[symbol] / total_weight) * target_total
            else:
                allocations[symbol] = equal_weight
        
        return allocations
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash_balance
        
        for symbol, holding in self.holdings.items():
            if symbol in current_prices:
                shares = holding['shares']
                current_price = current_prices[symbol]
                position_value = shares * current_price
                total_value += position_value
        
        return total_value
    
    def get_current_allocations(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Get current allocation percentages
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary of symbol -> current allocation percentage
        """
        portfolio_value = self.calculate_portfolio_value(current_prices)
        current_allocations = {}
        
        for symbol, holding in self.holdings.items():
            if symbol in current_prices:
                shares = holding['shares']
                current_price = current_prices[symbol]
                position_value = shares * current_price
                allocation = position_value / portfolio_value
                current_allocations[symbol] = allocation
        
        # Add cash allocation
        current_allocations['CASH'] = self.cash_balance / portfolio_value
        
        return current_allocations
    
    def identify_rebalancing_needs(self, current_prices: Dict[str, float]) -> Dict[str, Dict]:
        """
        Identify which positions need rebalancing
        
        Args:
            current_prices: Dictionary of symbol -> current price
            
        Returns:
            Dictionary of rebalancing actions needed
        """
        current_allocations = self.get_current_allocations(current_prices)
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        rebalancing_actions = {}
        
        for symbol in SACRED_SYMBOLS:
            target_allocation = self.target_allocations.get(symbol, 0)
            current_allocation = current_allocations.get(symbol, 0)
            
            allocation_diff = abs(target_allocation - current_allocation)
            
            if allocation_diff > self.rebalance_threshold:
                target_value = target_allocation * portfolio_value
                current_value = current_allocation * portfolio_value
                
                if symbol in current_prices:
                    current_price = current_prices[symbol]
                    target_shares = int(target_value / current_price)
                    current_shares = self.holdings.get(symbol, {}).get('shares', 0)
                    
                    shares_diff = target_shares - current_shares
                    value_diff = target_value - current_value
                    
                    rebalancing_actions[symbol] = {
                        'action': 'BUY' if shares_diff > 0 else 'SELL',
                        'shares_diff': abs(shares_diff),
                        'value_diff': abs(value_diff),
                        'current_allocation': current_allocation,
                        'target_allocation': target_allocation,
                        'current_price': current_price,
                        'priority': allocation_diff  # Higher diff = higher priority
                    }
        
        # Sort by priority (highest allocation difference first)
        sorted_actions = dict(sorted(rebalancing_actions.items(), 
                                   key=lambda x: x[1]['priority'], reverse=True))
        
        return sorted_actions
    
    def execute_trade(self, symbol: str, action: str, shares: int, 
                     price: float, timestamp: datetime = None) -> bool:
        """
        Execute a trade and update portfolio
        
        Args:
            symbol: Trading symbol
            action: 'BUY' or 'SELL'
            shares: Number of shares
            price: Execution price
            timestamp: Trade timestamp
            
        Returns:
            True if trade executed successfully
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        trade_value = shares * price
        
        if action.upper() == 'BUY':
            # Check if we have enough cash
            if trade_value > self.cash_balance:
                logger.warning(f"Insufficient cash for {symbol} purchase: ${trade_value:,.2f} needed, ${self.cash_balance:,.2f} available")
                return False
            
            # Execute buy
            self.cash_balance -= trade_value
            
            if symbol in self.holdings:
                # Add to existing position
                old_shares = self.holdings[symbol]['shares']
                old_avg_price = self.holdings[symbol]['avg_price']
                
                new_total_shares = old_shares + shares
                new_avg_price = ((old_shares * old_avg_price) + (shares * price)) / new_total_shares
                
                self.holdings[symbol].update({
                    'shares': new_total_shares,
                    'avg_price': new_avg_price,
                    'last_update': timestamp
                })
            else:
                # Create new position
                self.holdings[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'first_purchase': timestamp,
                    'last_update': timestamp
                }
            
            logger.info(f"✅ BOUGHT {shares} shares of {symbol} @ ${price:.2f}")
            
        elif action.upper() == 'SELL':
            # Check if we have enough shares
            if symbol not in self.holdings or self.holdings[symbol]['shares'] < shares:
                available_shares = self.holdings.get(symbol, {}).get('shares', 0)
                logger.warning(f"Insufficient shares for {symbol} sale: {shares} needed, {available_shares} available")
                return False
            
            # Execute sell
            self.cash_balance += trade_value
            
            self.holdings[symbol]['shares'] -= shares
            self.holdings[symbol]['last_update'] = timestamp
            
            # Remove position if fully sold
            if self.holdings[symbol]['shares'] == 0:
                del self.holdings[symbol]
            
            logger.info(f"✅ SOLD {shares} shares of {symbol} @ ${price:.2f}")
        
        else:
            logger.error(f"Invalid trade action: {action}")
            return False
        
        return True
    
    def rebalance_portfolio(self, current_prices: Dict[str, float], 
                          max_trades: int = 10) -> List[Dict]:
        """
        Rebalance portfolio to target allocations
        
        Args:
            current_prices: Dictionary of symbol -> current price
            max_trades: Maximum number of trades to execute
            
        Returns:
            List of executed trades
        """
        logger.info("Starting portfolio rebalancing...")
        
        rebalancing_needs = self.identify_rebalancing_needs(current_prices)
        
        if not rebalancing_needs:
            logger.info("Portfolio is already balanced")
            return []
        
        executed_trades = []
        trades_executed = 0
        
        # Execute highest priority trades first
        for symbol, action_info in rebalancing_needs.items():
            if trades_executed >= max_trades:
                break
            
            action = action_info['action']
            shares = action_info['shares_diff']
            price = action_info['current_price']
            
            # Execute trade
            success = self.execute_trade(symbol, action, shares, price)
            
            if success:
                trade_record = {
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'value': shares * price,
                    'timestamp': datetime.now(),
                    'reason': 'Rebalancing'
                }
                executed_trades.append(trade_record)
                trades_executed += 1
                
                logger.info(f"Rebalance trade {trades_executed}: {action} {shares} {symbol} @ ${price:.2f}")
        
        # Record rebalancing event
        if executed_trades:
            rebalance_record = {
                'timestamp': datetime.now(),
                'trades_executed': len(executed_trades),
                'total_value_traded': sum(trade['value'] for trade in executed_trades),
                'portfolio_value_before': self.calculate_portfolio_value(current_prices)
            }
            self.rebalance_history.append(rebalance_record)
        
        logger.info(f"Rebalancing completed: {len(executed_trades)} trades executed")
        return executed_trades
    
    def update_portfolio_performance(self, current_prices: Dict[str, float],
                                   benchmark_return: float = 0.0) -> Dict[str, Any]:
        """
        Update portfolio performance metrics
        
        Args:
            current_prices: Dictionary of symbol -> current price
            benchmark_return: Benchmark return for comparison
            
        Returns:
            Performance metrics dictionary
        """
        current_value = self.calculate_portfolio_value(current_prices)
        
        # Calculate returns
        if self.portfolio_history:
            previous_value = self.portfolio_history[-1]['portfolio_value']
            daily_return = (current_value - previous_value) / previous_value
        else:
            daily_return = 0.0
        
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # Update history
        portfolio_record = {
            'timestamp': datetime.now(),
            'portfolio_value': current_value,
            'cash_balance': self.cash_balance,
            'daily_return': daily_return,
            'total_return': total_return,
            'benchmark_return': benchmark_return
        }
        
        self.portfolio_history.append(portfolio_record)
        self.daily_returns.append(daily_return)
        self.benchmark_returns.append(benchmark_return)
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics()
        
        return performance_metrics
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            return {}
        
        returns = np.array(self.daily_returns)
        
        # Basic metrics
        total_return = self.portfolio_history[-1]['total_return']
        current_value = self.portfolio_history[-1]['portfolio_value']
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        portfolio_values = [record['portfolio_value'] for record in self.portfolio_history]
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        positive_days = np.sum(returns > 0)
        total_days = len(returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Beta vs benchmark (if available)
        beta = 1.0
        if len(self.benchmark_returns) == len(returns) and len(returns) > 20:
            benchmark_returns = np.array(self.benchmark_returns)
            if np.std(benchmark_returns) > 0:
                beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        
        return {
            'current_value': current_value,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_drawdown': drawdown[-1] if len(drawdown) > 0 else 0,
            'win_rate': win_rate,
            'beta': beta,
            'total_trades': len(self.rebalance_history),
            'days_tracked': len(self.portfolio_history)
        }
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        current_allocations = self.get_current_allocations(current_prices)
        performance_metrics = self.calculate_performance_metrics()
        
        # Holdings summary
        holdings_summary = []
        for symbol, holding in self.holdings.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                shares = holding['shares']
                avg_price = holding['avg_price']
                current_value = shares * current_price
                unrealized_pnl = shares * (current_price - avg_price)
                unrealized_pnl_percent = (unrealized_pnl / (shares * avg_price)) * 100
                
                holdings_summary.append({
                    'symbol': symbol,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'current_value': current_value,
                    'allocation': current_allocations.get(symbol, 0) * 100,
                    'target_allocation': self.target_allocations.get(symbol, 0) * 100,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_percent': unrealized_pnl_percent
                })
        
        return {
            'portfolio_value': self.calculate_portfolio_value(current_prices),
            'cash_balance': self.cash_balance,
            'cash_allocation': current_allocations.get('CASH', 0) * 100,
            'holdings_count': len(self.holdings),
            'holdings_summary': holdings_summary,
            'performance_metrics': performance_metrics,
            'target_allocations': {k: v*100 for k, v in self.target_allocations.items()},
            'current_allocations': {k: v*100 for k, v in current_allocations.items()},
            'rebalancing_needed': len(self.identify_rebalancing_needs(current_prices)) > 0
        }
    
    def save_portfolio_state(self, filename: str = None) -> Path:
        """Save current portfolio state"""
        if filename is None:
            filename = f"portfolio_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.portfolio_dir / filename
        
        # Save portfolio history
        if self.portfolio_history:
            history_df = pd.DataFrame(self.portfolio_history)
            history_df.to_csv(filepath, index=False)
        
        logger.info(f"Portfolio state saved to {filepath}")
        return filepath
    
    def load_portfolio_state(self, filepath: Path) -> None:
        """Load portfolio state from file"""
        if not filepath.exists():
            raise FileNotFoundError(f"Portfolio file not found: {filepath}")
        
        history_df = pd.read_csv(filepath)
        self.portfolio_history = history_df.to_dict('records')
        
        # Extract returns
        self.daily_returns = history_df['daily_return'].tolist()
        self.benchmark_returns = history_df['benchmark_return'].tolist()
        
        # Update current state
        if self.portfolio_history:
            latest = self.portfolio_history[-1]
            self.current_capital = latest['portfolio_value']
            self.cash_balance = latest['cash_balance']
        
        logger.info(f"Portfolio state loaded from {filepath}")
    
    def export_performance_report(self) -> str:
        """Export detailed performance report"""
        if not self.portfolio_history:
            return "No portfolio history available"
        
        performance = self.calculate_performance_metrics()
        
        report = f"""
PORTFOLIO PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*50}

SUMMARY METRICS
Initial Capital: ${self.initial_capital:,.2f}
Current Value: ${performance['current_value']:,.2f}
Total Return: {performance['total_return_percent']:.2f}%
Annualized Return: {performance['annualized_return']*100:.2f}%

RISK METRICS
Volatility (Annualized): {performance['volatility']*100:.2f}%
Sharpe Ratio: {performance['sharpe_ratio']:.2f}
Maximum Drawdown: {performance['max_drawdown']*100:.2f}%
Current Drawdown: {performance['current_drawdown']*100:.2f}%

TRADING METRICS
Win Rate: {performance['win_rate']*100:.2f}%
Total Rebalances: {performance['total_trades']}
Days Tracked: {performance['days_tracked']}
Beta vs Benchmark: {performance['beta']:.2f}

RISK TARGETS (God Mode)
Target Win Rate: {RISK_MANAGEMENT['target_win_rate']*100:.1f}%
Target Sharpe Ratio: {RISK_MANAGEMENT['target_sharpe_ratio']:.1f}
Max Allowed Drawdown: {RISK_MANAGEMENT['max_drawdown_percent']:.1f}%
Monthly Target: {RISK_MANAGEMENT['monthly_target_percent']:.1f}%
"""
        
        return report