"""
Improved Rotation Strategy
Simplified version with key enhancements for better performance
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
class ImprovedStockScore:
    """Stock scoring with key metrics"""
    symbol: str
    total_score: float
    momentum_score: float
    trend_score: float
    volatility_adjusted_score: float
    relative_strength: float
    current_price: float
    volatility: float
    volume_ratio: float


class ImprovedRotationStrategy:
    """Improved rotation with essential enhancements"""
    
    def __init__(self, initial_capital: float = 100000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        
        # Position sizing
        self.base_position_size = 0.10  # 10% base
        self.min_position_size = 0.05   # 5% minimum
        self.max_position_size = 0.15   # 15% maximum
        
        # Improved scoring weights
        self.weights = {
            'momentum': 0.30,
            'trend': 0.25,
            'relative_strength': 0.25,
            'volatility': 0.20
        }
        
        # Risk management
        self.stop_loss_atr_multiplier = 2.0
        self.trailing_stop_activation = 0.10  # 10% profit
        self.trailing_stop_distance = 0.05    # 5% trailing
        self.time_exit_days = 30
        self.time_exit_min_return = 0.05      # 5% minimum
        
        # Rotation parameters
        self.rotation_score_threshold = 0.10  # 10% better score needed
        self.min_holding_days = 3
        self.max_trades_per_rotation = 3
        
        # Market breadth
        self.market_breadth_threshold = 0.4  # 40% stocks must be positive
        
        # Strategies
        self.realistic_strategy = IndicatorBacktest()
        self.hierarchical_strategy = HierarchicalBacktest('4h')
        
        # State
        self.positions = {}
        self.trailing_stops = {}
        
        logger.info("Improved Rotation Strategy initialized")
    
    def calculate_improved_scores(self, symbols: List[str]) -> List[ImprovedStockScore]:
        """Calculate scores with key improvements"""
        scores = []
        
        # First calculate market average for relative strength
        market_returns = []
        
        for symbol in symbols:
            try:
                signals = self.realistic_strategy.generate_signals(symbol)
                if len(signals) >= 20:
                    ret_20d = (signals['close'].iloc[-1] / signals['close'].iloc[-20] - 1)
                    market_returns.append(ret_20d)
            except:
                continue
        
        avg_market_return = np.mean(market_returns) if market_returns else 0
        
        # Calculate individual scores
        for symbol in symbols:
            try:
                # Get signals
                realistic_signals = self.realistic_strategy.generate_signals(symbol)
                
                if realistic_signals.empty or len(realistic_signals) < 30:
                    continue
                
                # Basic data
                latest = realistic_signals.iloc[-1]
                current_price = latest.get('close', 0)
                
                # 1. Momentum Score (Multi-timeframe)
                momentum_score = self._calculate_momentum(realistic_signals)
                
                # 2. Trend Score (Simple but effective)
                trend_score = self._calculate_trend(realistic_signals, latest)
                
                # 3. Relative Strength
                relative_strength = self._calculate_relative_strength(
                    realistic_signals, avg_market_return
                )
                
                # 4. Volatility (for adjustment)
                volatility = self._calculate_volatility(realistic_signals)
                
                # 5. Volume Ratio
                volume_ratio = self._calculate_volume_ratio(realistic_signals)
                
                # Calculate total score
                base_score = (
                    self.weights['momentum'] * momentum_score +
                    self.weights['trend'] * trend_score +
                    self.weights['relative_strength'] * relative_strength
                )
                
                # Volatility adjustment (lower volatility = higher score)
                volatility_factor = 1 - (volatility / 0.04)  # Assume 4% is high volatility
                volatility_factor = max(0.5, min(1.5, volatility_factor))
                
                # Volume boost (higher recent volume = better)
                volume_factor = 1 + (max(0, min(0.2, (volume_ratio - 1) * 0.1)))
                
                # Final score
                total_score = base_score * volatility_factor * volume_factor
                
                scores.append(ImprovedStockScore(
                    symbol=symbol,
                    total_score=total_score,
                    momentum_score=momentum_score,
                    trend_score=trend_score,
                    volatility_adjusted_score=base_score * volatility_factor,
                    relative_strength=relative_strength,
                    current_price=current_price,
                    volatility=volatility,
                    volume_ratio=volume_ratio
                ))
                
            except Exception as e:
                logger.error(f"Error calculating score for {symbol}: {e}")
                continue
        
        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _calculate_momentum(self, signals_df: pd.DataFrame) -> float:
        """Simple but effective momentum calculation"""
        if len(signals_df) < 20:
            return 0.5
        
        prices = signals_df['close'].values
        
        # Multi-timeframe returns
        ret_5d = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
        ret_10d = (prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        ret_20d = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
        
        # Weight recent returns more
        momentum = ret_5d * 0.5 + ret_10d * 0.3 + ret_20d * 0.2
        
        # Normalize to 0-1
        return np.clip((momentum + 0.10) / 0.20, 0, 1)
    
    def _calculate_trend(self, signals_df: pd.DataFrame, latest: pd.Series) -> float:
        """Trend strength using available indicators"""
        score = 0.5  # Default neutral
        
        # 1. Price above moving averages
        if len(signals_df) >= 20:
            ma_20 = signals_df['close'].rolling(20).mean().iloc[-1]
            if signals_df['close'].iloc[-1] > ma_20:
                score += 0.2
        
        # 2. Supertrend
        if 'st_trend' in latest and latest['st_trend'] == 1:
            score += 0.2
        
        # 3. MACD
        if 'macd_hist' in latest and latest['macd_hist'] > 0:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_relative_strength(self, signals_df: pd.DataFrame, market_return: float) -> float:
        """Relative strength vs market"""
        if len(signals_df) < 20:
            return 0.5
        
        stock_return = (signals_df['close'].iloc[-1] / signals_df['close'].iloc[-20] - 1)
        
        # Outperformance
        if market_return != 0:
            rs = stock_return - market_return
            # Normalize: +10% outperformance = 1.0 score
            return np.clip((rs + 0.05) / 0.10, 0, 1)
        else:
            return 0.5 + stock_return
    
    def _calculate_volatility(self, signals_df: pd.DataFrame) -> float:
        """Calculate 20-day volatility"""
        if len(signals_df) < 20:
            return 0.02  # Default 2%
        
        returns = signals_df['close'].pct_change().dropna()
        return returns.tail(20).std()
    
    def _calculate_volume_ratio(self, signals_df: pd.DataFrame) -> float:
        """Recent volume vs average"""
        if 'volume' not in signals_df.columns or len(signals_df) < 20:
            return 1.0
        
        recent_vol = signals_df['volume'].tail(5).mean()
        avg_vol = signals_df['volume'].tail(20).mean()
        
        if avg_vol > 0:
            return recent_vol / avg_vol
        return 1.0
    
    def check_market_breadth(self, scores: List[ImprovedStockScore]) -> bool:
        """Check if market conditions are favorable"""
        if not scores:
            return False
        
        # Count positive momentum stocks
        positive_count = sum(1 for s in scores if s.momentum_score > 0.5)
        breadth = positive_count / len(scores)
        
        return breadth >= self.market_breadth_threshold
    
    def calculate_position_size(self, score: ImprovedStockScore, available_capital: float) -> float:
        """Volatility-adjusted position sizing"""
        # Base size from score
        score_factor = min(1.0, score.total_score)
        base_size = self.base_position_size + (self.max_position_size - self.base_position_size) * score_factor
        
        # Volatility adjustment
        vol_adjustment = 1 - (score.volatility / 0.03)  # 3% = neutral
        vol_adjustment = max(0.7, min(1.3, vol_adjustment))
        
        # Final size
        position_size = base_size * vol_adjustment * available_capital
        
        # Apply limits
        position_size = max(self.min_position_size * available_capital, position_size)
        position_size = min(self.max_position_size * available_capital, position_size)
        
        return position_size
    
    def check_exit_conditions(self, position: Dict, current_score: ImprovedStockScore) -> Tuple[bool, str]:
        """Improved exit logic with trailing stops"""
        symbol = position['symbol']
        entry_price = position['entry_price']
        current_price = current_score.current_price
        return_pct = ((current_price - entry_price) / entry_price)
        holding_days = (datetime.now() - position.get('entry_date', datetime.now())).days
        
        # 1. Stop loss (ATR-based)
        atr = current_score.volatility * np.sqrt(20) * current_price  # Approximate daily ATR
        stop_loss = entry_price - (atr * self.stop_loss_atr_multiplier)
        
        if current_price <= stop_loss:
            return True, 'stop_loss'
        
        # 2. Trailing stop
        if symbol in self.trailing_stops:
            if current_price <= self.trailing_stops[symbol]:
                return True, 'trailing_stop'
        
        # Update/activate trailing stop
        if return_pct >= self.trailing_stop_activation:
            trail_price = current_price * (1 - self.trailing_stop_distance)
            if symbol not in self.trailing_stops or trail_price > self.trailing_stops[symbol]:
                self.trailing_stops[symbol] = trail_price
        
        # 3. Time-based exit
        if holding_days >= self.time_exit_days and return_pct < self.time_exit_min_return:
            return True, 'time_exit'
        
        # 4. Score deterioration
        if current_score.total_score < 0.3:  # Very low score
            return True, 'low_score'
        
        return False, ''
    
    def identify_rotation_trades(self, scores: List[ImprovedStockScore]) -> Tuple[List[str], List[str]]:
        """Identify rotation opportunities"""
        sell_candidates = []
        buy_candidates = []
        
        # Get top N stocks
        top_stocks = [s.symbol for s in scores[:self.max_positions]]
        
        # Check each position
        for symbol, position in self.positions.items():
            # Find current score
            stock_score = next((s for s in scores if s.symbol == symbol), None)
            if not stock_score:
                continue
            
            # Check exit conditions first
            should_exit, reason = self.check_exit_conditions(position, stock_score)
            if should_exit:
                sell_candidates.append((symbol, reason))
                continue
            
            # Check if still in top N
            if symbol not in top_stocks:
                # Find replacement with significantly better score
                for new_stock in scores[:self.max_positions + 5]:
                    if new_stock.symbol not in self.positions:
                        score_diff = (new_stock.total_score - stock_score.total_score) / stock_score.total_score
                        if score_diff > self.rotation_score_threshold:
                            sell_candidates.append((symbol, 'rotation'))
                            buy_candidates.append(new_stock.symbol)
                            break
        
        # Add new positions if space available
        current_positions = len(self.positions) - len(sell_candidates)
        for score in scores[:self.max_positions + 5]:
            if current_positions >= self.max_positions:
                break
            
            if score.symbol not in self.positions and score.symbol not in [b for b in buy_candidates]:
                # Only buy if momentum is positive
                if score.momentum_score > 0.5 and score.trend_score > 0.5:
                    buy_candidates.append(score.symbol)
                    current_positions += 1
        
        # Limit trades per rotation
        sell_candidates = sell_candidates[:self.max_trades_per_rotation]
        buy_candidates = buy_candidates[:self.max_trades_per_rotation]
        
        return [s[0] for s in sell_candidates], buy_candidates
    
    def generate_signals(self, symbols: List[str] = None) -> Dict:
        """Generate rotation signals"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:20]
        
        # Calculate scores
        scores = self.calculate_improved_scores(symbols)
        
        if not scores:
            return {'buy': [], 'sell': [], 'hold': list(self.positions.keys())}
        
        # Check market breadth
        if not self.check_market_breadth(scores):
            logger.warning("Poor market breadth - reducing positions")
            # In poor market, only keep top 5
            keep_symbols = [s.symbol for s in scores[:5] if s.symbol in self.positions]
            sell_list = [s for s in self.positions.keys() if s not in keep_symbols]
            return {'buy': [], 'sell': sell_list, 'hold': keep_symbols}
        
        # Identify rotations
        sell_list, buy_list = self.identify_rotation_trades(scores)
        
        # Log top scores
        logger.info("Top 5 stocks by score:")
        for i, score in enumerate(scores[:5]):
            logger.info(f"{i+1}. {score.symbol}: {score.total_score:.3f} "
                       f"(M:{score.momentum_score:.2f}, RS:{score.relative_strength:.2f})")
        
        return {
            'buy': buy_list,
            'sell': sell_list,
            'hold': [s for s in self.positions.keys() if s not in sell_list],
            'scores': {s.symbol: s for s in scores}
        }