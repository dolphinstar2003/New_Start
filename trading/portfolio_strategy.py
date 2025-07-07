"""
Portfolio Strategy Manager
Combines multiple backtest strategies with weighted allocation
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from loguru import logger
import asyncio
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS
from backtest.realistic_backtest import IndicatorBacktest
from backtest.backtest_sirali import HierarchicalBacktest


@dataclass
class PortfolioAllocation:
    """Portfolio allocation configuration"""
    strategy_name: str
    weight: float
    max_positions: int
    description: str


class PortfolioStrategy:
    """Manage portfolio with multiple strategies"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Portfolio allocations
        self.allocations = [
            PortfolioAllocation(
                strategy_name="realistic",
                weight=0.60,
                max_positions=6,  # 60% / 10% per position
                description="Conservative indicator-based strategy"
            ),
            PortfolioAllocation(
                strategy_name="hier_4h_sl_trailing",
                weight=0.30,
                max_positions=3,  # 30% / 10% per position
                description="Aggressive 4H with trailing stop"
            ),
            PortfolioAllocation(
                strategy_name="cash_reserve",
                weight=0.10,
                max_positions=0,
                description="Cash reserve for opportunities"
            )
        ]
        
        # Strategy instances
        self.strategies = {
            "realistic": IndicatorBacktest(),
            "hier_4h_sl_trailing": HierarchicalBacktest('4h')
        }
        
        # Active positions by strategy
        self.positions = {
            "realistic": {},
            "hier_4h_sl_trailing": {},
            "total": {}
        }
        
        # Performance tracking
        self.performance = {
            "realistic": {"trades": 0, "profit": 0, "win_rate": 0},
            "hier_4h_sl_trailing": {"trades": 0, "profit": 0, "win_rate": 0},
            "portfolio": {"total_return": 0, "max_drawdown": 0, "sharpe_ratio": 0}
        }
        
        logger.info("Portfolio Strategy initialized")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")
        for alloc in self.allocations:
            logger.info(f"  {alloc.strategy_name}: {alloc.weight*100:.0f}% (${initial_capital*alloc.weight:,.2f})")
    
    def get_strategy_capital(self, strategy_name: str) -> float:
        """Get allocated capital for a strategy"""
        for alloc in self.allocations:
            if alloc.strategy_name == strategy_name:
                return self.capital * alloc.weight
        return 0
    
    def get_position_size(self, strategy_name: str) -> float:
        """Calculate position size for a strategy"""
        strategy_capital = self.get_strategy_capital(strategy_name)
        
        for alloc in self.allocations:
            if alloc.strategy_name == strategy_name:
                if alloc.max_positions > 0:
                    return strategy_capital / alloc.max_positions
                break
        
        return 0
    
    def can_open_position(self, strategy_name: str, symbol: str) -> bool:
        """Check if we can open a new position"""
        # Check if already have position in this symbol
        if symbol in self.positions["total"]:
            return False
        
        # Check strategy position limit
        strategy_positions = len(self.positions.get(strategy_name, {}))
        
        for alloc in self.allocations:
            if alloc.strategy_name == strategy_name:
                if strategy_positions >= alloc.max_positions:
                    return False
                break
        
        # Check available capital
        position_size = self.get_position_size(strategy_name)
        if position_size <= 0:
            return False
        
        return True
    
    def generate_portfolio_signals(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Generate signals from all strategies"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:20]
        
        all_signals = {}
        
        # Realistic strategy signals
        logger.info("Generating realistic strategy signals...")
        realistic_backtest = self.strategies["realistic"]
        for symbol in symbols:
            try:
                signals = realistic_backtest.generate_signals(symbol)
                if not signals.empty:
                    all_signals[f"realistic_{symbol}"] = signals
            except Exception as e:
                logger.error(f"Error generating realistic signals for {symbol}: {e}")
        
        # Hierarchical 4H trailing stop signals
        logger.info("Generating hier_4h_sl_trailing signals...")
        hier_backtest = self.strategies["hier_4h_sl_trailing"]
        hier_backtest.risk_strategy = 'sl_plus_trailing'
        
        for symbol in symbols:
            try:
                signals = hier_backtest.generate_signals_sequential(symbol, '4h')
                if not signals.empty:
                    all_signals[f"hier_4h_sl_trailing_{symbol}"] = signals
            except Exception as e:
                logger.error(f"Error generating hier signals for {symbol}: {e}")
        
        logger.info(f"Generated {len(all_signals)} signal sets")
        return all_signals
    
    def execute_portfolio_signals(self, signals_dict: Dict[str, pd.DataFrame], 
                                current_date: datetime) -> List[Dict]:
        """Execute signals based on portfolio allocation"""
        trades = []
        
        logger.debug(f"Executing signals for date: {current_date}")
        
        # Process each strategy
        for strategy_name in ["realistic", "hier_4h_sl_trailing"]:
            # Get signals for this strategy
            strategy_signals = {
                symbol.split('_', 1)[1]: df 
                for symbol, df in signals_dict.items() 
                if symbol.startswith(strategy_name)
            }
            
            # Check each symbol
            for symbol, signals_df in strategy_signals.items():
                # Try to get the last signal (most recent)
                if len(signals_df) == 0:
                    continue
                
                # Use the last available signal
                last_idx = signals_df.index[-1]
                signal = signals_df.loc[last_idx, 'signal']
                
                # Skip if we already have a position in this symbol
                if symbol in self.positions["total"]:
                    continue
                
                # Log signal status only for actionable signals
                can_open = self.can_open_position(strategy_name, symbol)
                if signal != 0 and can_open:
                    logger.debug(f"{strategy_name} - {symbol}: signal={signal}, can_open={can_open}")
                
                # Buy signal
                if signal == 1 and can_open:
                    position_size = self.get_position_size(strategy_name)
                    
                    trade = {
                        'date': current_date,
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'action': 'BUY',
                        'size': position_size,
                        'price': signals_df.loc[last_idx, 'close']
                    }
                    
                    trades.append(trade)
                    
                    # Update positions
                    self.positions[strategy_name][symbol] = trade
                    self.positions["total"][symbol] = trade
                
                # Sell signal
                elif signal == -1 and symbol in self.positions[strategy_name]:
                    position = self.positions[strategy_name][symbol]
                    
                    trade = {
                        'date': current_date,
                        'symbol': symbol,
                        'strategy': strategy_name,
                        'action': 'SELL',
                        'size': position['size'],
                        'price': signals_df.loc[last_idx, 'close'],
                        'entry_price': position['price'],
                        'return_pct': ((signals_df.loc[last_idx, 'close'] - position['price']) / position['price']) * 100
                    }
                    
                    trades.append(trade)
                    
                    # Update positions
                    del self.positions[strategy_name][symbol]
                    del self.positions["total"][symbol]
        
        return trades
    
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        total_value = self.capital
        
        # Calculate position values
        position_values = {}
        for strategy_name, positions in self.positions.items():
            if strategy_name == "total":
                continue
            
            strategy_value = 0
            for symbol, position in positions.items():
                # Would need current price here
                position_values[f"{strategy_name}_{symbol}"] = position['size']
                strategy_value += position['size']
            
            position_values[f"{strategy_name}_total"] = strategy_value
        
        # Calculate allocations
        current_allocations = {}
        for alloc in self.allocations:
            if alloc.strategy_name == "cash_reserve":
                current_allocations[alloc.strategy_name] = 1.0 - sum(
                    position_values.get(f"{s}_total", 0) / total_value 
                    for s in ["realistic", "hier_4h_sl_trailing"]
                )
            else:
                current_allocations[alloc.strategy_name] = position_values.get(f"{alloc.strategy_name}_total", 0) / total_value
        
        return {
            'total_value': total_value,
            'positions_count': len(self.positions["total"]),
            'current_allocations': current_allocations,
            'target_allocations': {a.strategy_name: a.weight for a in self.allocations},
            'position_values': position_values,
            'performance': self.performance
        }
    
    def rebalance_portfolio(self) -> List[Dict]:
        """Rebalance portfolio to target allocations"""
        trades = []
        status = self.get_portfolio_status()
        
        # Check each strategy allocation
        for alloc in self.allocations:
            if alloc.strategy_name == "cash_reserve":
                continue
            
            current_weight = status['current_allocations'][alloc.strategy_name]
            target_weight = alloc.weight
            
            # If deviation > 5%, rebalance
            if abs(current_weight - target_weight) > 0.05:
                logger.info(f"Rebalancing {alloc.strategy_name}: {current_weight:.1%} -> {target_weight:.1%}")
                
                # Calculate adjustment needed
                current_value = self.capital * current_weight
                target_value = self.capital * target_weight
                adjustment = target_value - current_value
                
                if adjustment > 0:
                    # Need to buy more
                    logger.info(f"  Need to add ${adjustment:,.2f} to {alloc.strategy_name}")
                else:
                    # Need to sell some
                    logger.info(f"  Need to reduce ${-adjustment:,.2f} from {alloc.strategy_name}")
                
                # Would implement actual rebalancing logic here
        
        return trades


def create_portfolio_config() -> Dict:
    """Create portfolio configuration file"""
    config = {
        "portfolio_name": "Conservative Growth Portfolio",
        "initial_capital": 100000,
        "allocations": [
            {
                "strategy": "realistic",
                "weight": 0.60,
                "description": "Core holding - indicator-based signals",
                "risk_level": "low",
                "expected_return": "4-6% monthly",
                "max_positions": 6
            },
            {
                "strategy": "hier_4h_sl_trailing", 
                "weight": 0.30,
                "description": "Growth component - 4H timeframe with trailing stops",
                "risk_level": "medium-high",
                "expected_return": "6-10% monthly",
                "max_positions": 3
            },
            {
                "strategy": "cash_reserve",
                "weight": 0.10,
                "description": "Cash reserve for opportunities and risk management",
                "risk_level": "none",
                "expected_return": "0%",
                "max_positions": 0
            }
        ],
        "risk_management": {
            "max_portfolio_drawdown": 0.15,  # 15% max drawdown
            "rebalance_frequency": "monthly",
            "position_sizing": "equal_weight",
            "stop_loss_default": 0.03,  # 3% stop loss
            "take_profit_default": 0.08  # 8% take profit
        },
        "trading_rules": {
            "max_positions_total": 10,
            "min_volume_filter": 1000000,  # $1M daily volume
            "no_trade_on_news": True,
            "avoid_earnings_week": True
        }
    }
    
    # Save configuration
    import json
    config_path = Path(__file__).parent / 'portfolio_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Portfolio configuration saved to {config_path}")
    
    return config


def main():
    """Test portfolio strategy"""
    # Create portfolio
    portfolio = PortfolioStrategy(initial_capital=100000)
    
    # Get current status
    status = portfolio.get_portfolio_status()
    
    print("\n" + "="*60)
    print("PORTFOLIO STATUS")
    print("="*60)
    print(f"Total Value: ${status['total_value']:,.2f}")
    print(f"Active Positions: {status['positions_count']}")
    
    print("\nCurrent Allocations:")
    for strategy, weight in status['current_allocations'].items():
        target = status['target_allocations'][strategy]
        print(f"  {strategy}: {weight:.1%} (target: {target:.1%})")
    
    # Create configuration file
    config = create_portfolio_config()
    
    print("\n✅ Portfolio strategy configured successfully!")
    print("📄 Configuration saved to portfolio_config.json")
    
    return portfolio


if __name__ == "__main__":
    main()