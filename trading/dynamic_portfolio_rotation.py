"""
Dynamic Portfolio Rotation Strategy
- Keeps best 10 stocks
- Continuously scans for better opportunities
- Rotates out saturated positions for high potential ones
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS
from backtest.realistic_backtest import IndicatorBacktest
from backtest.backtest_sirali import HierarchicalBacktest


@dataclass
class StockScore:
    """Stock scoring data"""
    symbol: str
    score: float
    momentum_score: float
    trend_score: float
    volatility_score: float
    potential_score: float
    current_price: float
    signals: Dict


class DynamicRotationStrategy:
    """Dynamic portfolio rotation with top 10 stocks"""
    
    def __init__(self, initial_capital: float = 100000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.min_position_size = 0.08  # 8% minimum
        self.max_position_size = 0.15  # 15% maximum
        
        # Scoring weights
        self.weights = {
            'momentum': 0.30,    # Recent price momentum
            'trend': 0.25,       # Trend strength
            'volatility': 0.20,  # Volatility (inverse)
            'indicators': 0.25   # Technical indicators
        }
        
        # Strategies for signal generation
        self.realistic_strategy = IndicatorBacktest()
        self.hierarchical_strategy = HierarchicalBacktest('4h')
        
        # Current positions
        self.positions = {}
        
        # Rotation thresholds
        self.rotation_threshold = 0.15  # 15% score difference needed
        self.profit_lock_threshold = 0.12  # Lock profits above 12%
        self.saturation_threshold = 0.08   # Consider saturated above 8% with weakening momentum
        
        logger.info(f"Dynamic Rotation Strategy initialized with {max_positions} positions")
    
    def calculate_stock_scores(self, symbols: List[str]) -> List[StockScore]:
        """Calculate comprehensive scores for all symbols"""
        scores = []
        
        for symbol in symbols:
            try:
                # Get signals from both strategies
                realistic_signals = self.realistic_strategy.generate_signals(symbol)
                hier_signals = self.hierarchical_strategy.generate_signals_sequential(symbol, '4h')
                
                if realistic_signals.empty or hier_signals.empty:
                    continue
                
                # Get latest data
                latest_realistic = realistic_signals.iloc[-1]
                latest_hier = hier_signals.iloc[-1]
                current_price = latest_realistic.get('close', 0)
                
                # Calculate momentum score (price change over different periods)
                momentum_score = self._calculate_momentum_score(realistic_signals)
                
                # Calculate trend score (ADX, directional movement)
                trend_score = self._calculate_trend_score(latest_realistic)
                
                # Calculate volatility score (lower is better)
                volatility_score = self._calculate_volatility_score(realistic_signals)
                
                # Calculate indicator score
                indicator_score = self._calculate_indicator_score(latest_realistic, latest_hier)
                
                # Calculate potential score (room to grow)
                potential_score = self._calculate_potential_score(realistic_signals, latest_realistic)
                
                # Composite score
                total_score = (
                    self.weights['momentum'] * momentum_score +
                    self.weights['trend'] * trend_score +
                    self.weights['volatility'] * volatility_score +
                    self.weights['indicators'] * indicator_score
                )
                
                scores.append(StockScore(
                    symbol=symbol,
                    score=total_score,
                    momentum_score=momentum_score,
                    trend_score=trend_score,
                    volatility_score=volatility_score,
                    potential_score=potential_score,
                    current_price=current_price,
                    signals={
                        'realistic': latest_realistic.get('signal', 0),
                        'hierarchical': latest_hier.get('signal', 0)
                    }
                ))
                
            except Exception as e:
                logger.error(f"Error calculating score for {symbol}: {e}")
                continue
        
        # Sort by total score
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores
    
    def _calculate_momentum_score(self, signals_df: pd.DataFrame) -> float:
        """Calculate momentum based on price changes"""
        if len(signals_df) < 20:
            return 0.5
        
        try:
            prices = signals_df['close'].values
            
            # Calculate returns over different periods
            ret_5d = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
            ret_10d = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
            ret_20d = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
            
            # Weight recent performance more
            momentum = (ret_5d * 0.5 + ret_10d * 0.3 + ret_20d * 0.2)
            
            # Normalize to 0-1 scale
            return np.clip((momentum + 0.1) / 0.2, 0, 1)
            
        except:
            return 0.5
    
    def _calculate_trend_score(self, latest_data: pd.Series) -> float:
        """Calculate trend strength score"""
        try:
            adx = latest_data.get('adx', 0)
            plus_di = latest_data.get('plus_di', 0)
            minus_di = latest_data.get('minus_di', 0)
            
            # Strong uptrend: high ADX + plus_di > minus_di
            if adx > 25 and plus_di > minus_di:
                trend_strength = min(adx / 50, 1.0)  # Normalize ADX
                di_ratio = plus_di / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0.5
                return trend_strength * di_ratio
            else:
                return 0.3  # Weak or no trend
                
        except:
            return 0.5
    
    def _calculate_volatility_score(self, signals_df: pd.DataFrame) -> float:
        """Calculate volatility score (lower volatility = higher score)"""
        if len(signals_df) < 10:
            return 0.5
        
        try:
            returns = signals_df['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Inverse relationship - lower volatility gets higher score
            # Assuming 2% daily volatility is average
            vol_score = 1 - min(volatility / 0.04, 1.0)
            return vol_score
            
        except:
            return 0.5
    
    def _calculate_indicator_score(self, realistic_data: pd.Series, hier_data: pd.Series) -> float:
        """Calculate score based on technical indicators"""
        score = 0
        
        # Realistic strategy signal
        if realistic_data.get('signal', 0) == 1:
            score += 0.3
        
        # Hierarchical strategy signal
        if hier_data.get('signal', 0) == 1:
            score += 0.3
        
        # Supertrend
        if realistic_data.get('st_trend', 0) == 1:
            score += 0.1
        
        # MACD momentum
        if realistic_data.get('macd_hist', 0) > 0:
            score += 0.1
        
        # Squeeze momentum
        if not realistic_data.get('sq_on', True) and realistic_data.get('sq_mom', 0) > 0:
            score += 0.1
        
        # WaveTrend
        if realistic_data.get('wt1', 0) < 0 and realistic_data.get('wt_cross_up', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_potential_score(self, signals_df: pd.DataFrame, latest_data: pd.Series) -> float:
        """Calculate upside potential based on recent highs and support levels"""
        if len(signals_df) < 20:
            return 0.5
        
        try:
            current_price = latest_data.get('close', 0)
            
            # Recent high (20-day)
            recent_high = signals_df['close'].tail(20).max()
            
            # Distance from high (more room = higher potential)
            distance_from_high = (recent_high - current_price) / current_price
            
            # If we're at new highs, check longer-term potential
            if distance_from_high < 0.02:  # Within 2% of recent high
                # Look at 50-day high
                if len(signals_df) >= 50:
                    longer_high = signals_df['close'].tail(50).max()
                    distance_from_high = (longer_high - current_price) / current_price
            
            # Convert to score (0-20% upside = 0-1 score)
            potential = min(distance_from_high / 0.20, 1.0)
            
            # Adjust for oversold conditions
            wt1 = latest_data.get('wt1', 0)
            if wt1 < -60:  # Oversold
                potential = min(potential + 0.2, 1.0)
            
            return potential
            
        except:
            return 0.5
    
    def identify_rotation_candidates(self, current_scores: List[StockScore]) -> Tuple[List[str], List[str]]:
        """Identify which positions to sell and which to buy"""
        sell_candidates = []
        buy_candidates = []
        
        # Get top N stocks by score
        top_stocks = [s.symbol for s in current_scores[:self.max_positions]]
        
        # Check current positions for rotation
        for symbol, position in self.positions.items():
            current_return = position.get('return_pct', 0)
            holding_days = (datetime.now() - position.get('entry_date', datetime.now())).days
            
            # Find current score
            stock_score = next((s for s in current_scores if s.symbol == symbol), None)
            if not stock_score:
                continue
            
            # Reasons to sell:
            # 1. No longer in top N
            if symbol not in top_stocks:
                sell_candidates.append(symbol)
                logger.info(f"Sell candidate {symbol}: No longer in top {self.max_positions}")
                continue
            
            # 2. Profit taking on saturated positions
            if current_return > self.profit_lock_threshold:
                # Check if momentum is weakening
                if stock_score.momentum_score < 0.4:
                    sell_candidates.append(symbol)
                    logger.info(f"Sell candidate {symbol}: Profit {current_return:.1%} with weak momentum")
                    continue
            
            # 3. Stop loss
            if current_return < -0.05:  # -5% stop loss
                sell_candidates.append(symbol)
                logger.info(f"Sell candidate {symbol}: Stop loss at {current_return:.1%}")
                continue
            
            # 4. Better opportunity available
            # Check if there's a significantly better stock not in portfolio
            for better_stock in current_scores:
                if better_stock.symbol not in self.positions:
                    score_diff = better_stock.score - stock_score.score
                    if score_diff > self.rotation_threshold:
                        sell_candidates.append(symbol)
                        buy_candidates.append(better_stock.symbol)
                        logger.info(f"Rotation: {symbol} -> {better_stock.symbol} (score diff: {score_diff:.2f})")
                        break
        
        # Add top stocks not in portfolio to buy list
        for stock_score in current_scores[:self.max_positions]:
            if stock_score.symbol not in self.positions and stock_score.symbol not in buy_candidates:
                # Only buy if signals are positive
                if stock_score.signals['realistic'] == 1 or stock_score.signals['hierarchical'] == 1:
                    buy_candidates.append(stock_score.symbol)
        
        return sell_candidates[:3], buy_candidates[:3]  # Limit to 3 trades per cycle
    
    def calculate_position_size(self, score: float, available_capital: float) -> float:
        """Calculate position size based on score and available capital"""
        # Higher score = larger position
        size_factor = self.min_position_size + (self.max_position_size - self.min_position_size) * score
        
        # Ensure we don't exceed available capital
        position_size = min(available_capital * size_factor, available_capital / 2)
        
        return position_size
    
    def generate_rotation_signals(self, symbols: List[str] = None) -> Dict:
        """Generate buy/sell signals for portfolio rotation"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:20]
        
        # Calculate scores for all symbols
        logger.info("Calculating scores for all symbols...")
        scores = self.calculate_stock_scores(symbols)
        
        if not scores:
            logger.warning("No scores calculated")
            return {}
        
        # Log top 10
        logger.info("\nTop 10 stocks by score:")
        for i, score in enumerate(scores[:10]):
            logger.info(f"{i+1}. {score.symbol}: {score.score:.3f} "
                       f"(M:{score.momentum_score:.2f}, T:{score.trend_score:.2f}, "
                       f"V:{score.volatility_score:.2f}, P:{score.potential_score:.2f})")
        
        # Identify rotation candidates
        sell_candidates, buy_candidates = self.identify_rotation_candidates(scores)
        
        # Create signals
        signals = {
            'sell': sell_candidates,
            'buy': buy_candidates,
            'scores': {s.symbol: s for s in scores}
        }
        
        return signals


def create_rotation_portfolio_config() -> Dict:
    """Create configuration for rotation strategy"""
    from typing import Dict
    config = {
        "strategy_name": "Dynamic Top 10 Rotation",
        "max_positions": 10,
        "position_sizing": "score_based",
        "min_position_pct": 8,
        "max_position_pct": 15,
        "rotation_rules": {
            "check_frequency": "every_signal",
            "min_score_difference": 0.15,
            "profit_lock_threshold": 0.12,
            "stop_loss": -0.05,
            "max_trades_per_rotation": 3
        },
        "scoring_weights": {
            "momentum": 0.30,
            "trend": 0.25,
            "volatility": 0.20,
            "indicators": 0.25
        }
    }
    
    import json
    config_path = Path(__file__).parent / 'rotation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Rotation config saved to {config_path}")
    return config


if __name__ == "__main__":
    # Test the rotation strategy
    strategy = DynamicRotationStrategy()
    
    # Test positions for simulation
    strategy.positions = {
        'GARAN': {'entry_date': datetime.now() - timedelta(days=10), 'return_pct': 0.08},
        'AKBNK': {'entry_date': datetime.now() - timedelta(days=5), 'return_pct': 0.15},
        'THYAO': {'entry_date': datetime.now() - timedelta(days=20), 'return_pct': -0.03}
    }
    
    signals = strategy.generate_rotation_signals()
    
    print("\n" + "="*60)
    print("ROTATION SIGNALS")
    print("="*60)
    print(f"\nSELL: {signals.get('sell', [])}")
    print(f"BUY: {signals.get('buy', [])}")
    
    # Create config
    create_rotation_portfolio_config()