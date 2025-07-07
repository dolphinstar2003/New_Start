"""
Target Price Manager
Manages and stores daily target prices for each strategy
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TargetManager:
    """Manages target prices for trading strategies"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.targets_file = self.data_dir / "daily_targets.json"
        self.targets = self.load_targets()
        
    def load_targets(self) -> dict:
        """Load existing targets from file"""
        logger.info(f"Loading targets from: {self.targets_file}")
        if self.targets_file.exists():
            try:
                with open(self.targets_file, 'r') as f:
                    data = json.load(f)
                    # Check if targets are from today
                    if data.get('date') == datetime.now().strftime('%Y-%m-%d'):
                        logger.info(f"Loaded targets for today with {len(data.get('strategies', {}))} strategies")
                        return data
                    else:
                        logger.info(f"Targets are from {data.get('date')}, need to update")
            except Exception as e:
                logger.error(f"Error loading targets: {e}")
        else:
            logger.info(f"Targets file does not exist: {self.targets_file}")
        
        # Return empty structure if no valid targets
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'strategies': {}
        }
    
    def save_targets(self):
        """Save targets to file"""
        try:
            with open(self.targets_file, 'w') as f:
                json.dump(self.targets, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving targets: {e}")
    
    def calculate_target_price(self, symbol: str, current_price: float, 
                             strategy: str) -> float:
        """Calculate target buy price based on strategy"""
        
        # Different strategies have different entry criteria
        if strategy == "aggressive":
            # Aggressive: Buy closer to current price (1-2% pullback)
            if current_price > 100:
                return round(current_price * 0.98, 2)
            else:
                return round(current_price * 0.985, 2)
                
        elif strategy == "balanced":
            # Balanced: Moderate pullback (2-3%)
            if current_price > 100:
                return round(current_price * 0.97, 2)
            else:
                return round(current_price * 0.975, 2)
                
        elif strategy == "conservative":
            # Conservative: Wait for bigger pullback (3-4%)
            if current_price > 100:
                return round(current_price * 0.96, 2)
            else:
                return round(current_price * 0.97, 2)
        
        return current_price
    
    def update_targets(self, prices: Dict[str, float], force_update: bool = False):
        """Update target prices for all strategies"""
        
        # Check if we need to update (new day or forced)
        current_date = datetime.now().strftime('%Y-%m-%d')
        if not force_update and self.targets.get('date') == current_date:
            logger.info("Targets already set for today")
            return
        
        logger.info("Calculating new target prices...")
        
        # Reset targets for new day
        self.targets = {
            'date': current_date,
            'time': datetime.now().strftime('%H:%M:%S'),
            'strategies': {
                'aggressive': {},
                'balanced': {},
                'conservative': {}
            }
        }
        
        # Calculate targets for each strategy and symbol
        for symbol, price in prices.items():
            for strategy in ['aggressive', 'balanced', 'conservative']:
                target = self.calculate_target_price(symbol, price, strategy)
                self.targets['strategies'][strategy][symbol] = {
                    'target_price': target,
                    'current_price': price,
                    'distance': round((target - price) / price * 100, 2)
                }
        
        # Save to file
        self.save_targets()
        logger.info(f"Updated targets for {len(prices)} symbols")
    
    def get_target(self, symbol: str, strategy: str) -> Optional[float]:
        """Get target price for a symbol and strategy"""
        try:
            return self.targets['strategies'][strategy][symbol]['target_price']
        except KeyError:
            return None
    
    def get_all_targets(self, strategy: str) -> dict:
        """Get all targets for a strategy"""
        return self.targets['strategies'].get(strategy, {})
    
    def check_target_hit(self, symbol: str, current_price: float, 
                        strategy: str) -> bool:
        """Check if current price has hit target"""
        target = self.get_target(symbol, strategy)
        if target:
            return current_price <= target
        return False
    
    def get_status_summary(self, current_prices: dict) -> dict:
        """Get summary of target status for all strategies"""
        summary = {}
        
        for strategy in ['aggressive', 'balanced', 'conservative']:
            targets = self.get_all_targets(strategy)
            ready_count = 0
            total_distance = 0
            nearest = None
            nearest_distance = float('inf')
            
            for symbol, target_data in targets.items():
                if symbol in current_prices:
                    current = current_prices[symbol]
                    target = target_data['target_price']
                    distance = (target - current) / current * 100
                    
                    if current <= target:
                        ready_count += 1
                    
                    total_distance += abs(distance)
                    
                    if 0 < distance < nearest_distance:
                        nearest = symbol
                        nearest_distance = distance
            
            summary[strategy] = {
                'ready_count': ready_count,
                'total_symbols': len(targets),
                'avg_distance': round(total_distance / len(targets), 2) if targets else 0,
                'nearest_symbol': nearest,
                'nearest_distance': round(nearest_distance, 2) if nearest else None
            }
        
        return summary