"""
Enhanced Dynamic Portfolio Rotation Strategy
Improved version with better timing, scoring, and risk management
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from loguru import logger
from dataclasses import dataclass
# import talib  # Removed due to installation issues

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS
from backtest.realistic_backtest import IndicatorBacktest
from backtest.backtest_sirali import HierarchicalBacktest
from indicators.vixfix import calculate_vixfix


@dataclass
class EnhancedStockScore:
    """Enhanced stock scoring with more metrics"""
    symbol: str
    total_score: float
    momentum_score: float
    trend_score: float
    volatility_score: float
    relative_strength: float
    indicator_score: float
    ml_score: float
    volume_score: float
    current_price: float
    atr: float
    signals: Dict
    kelly_fraction: float


class EnhancedRotationStrategy:
    """Enhanced portfolio rotation with improved features"""
    
    def __init__(self, initial_capital: float = 100000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        
        # Enhanced position sizing
        self.base_position_size = 0.08   # 8% base
        self.max_position_size = 0.15    # 15% max
        self.kelly_multiplier = 0.25     # Conservative Kelly
        
        # Improved scoring weights
        self.weights = {
            'momentum': 0.25,         # Recent price momentum
            'trend': 0.20,           # Trend strength
            'volatility': 0.15,      # Volatility (inverse)
            'relative_strength': 0.20, # vs market/sector
            'indicators': 0.15,      # Technical indicators
            'volume': 0.05          # Volume profile
        }
        
        # Dynamic rotation parameters
        self.base_rotation_days = 3
        self.min_rotation_days = 1
        self.max_rotation_days = 5
        
        # Enhanced exit parameters
        self.dynamic_stop_loss_atr = 2.0    # 2x ATR
        self.profit_target_atr = 4.0        # 4x ATR
        self.partial_exit_threshold = 0.10   # 10% profit
        self.partial_exit_percent = 0.5      # Exit 50%
        self.time_exit_days = 30            # Exit after 30 days if underperforming
        
        # Market regime detection
        self.market_regime = 'neutral'  # bull, bear, neutral
        self.regime_position_adjustment = {
            'bull': 1.0,
            'neutral': 0.8,
            'bear': 0.6
        }
        
        # Strategies
        self.realistic_strategy = IndicatorBacktest()
        self.hierarchical_strategy = HierarchicalBacktest('4h')
        
        # Current state
        self.positions = {}
        self.partial_exits = {}  # Track partial exits
        
        logger.info("Enhanced Rotation Strategy initialized")
    
    def calculate_enhanced_scores(self, symbols: List[str], market_data: Dict = None) -> List[EnhancedStockScore]:
        """Calculate comprehensive scores with enhanced metrics"""
        scores = []
        
        # First, calculate market regime
        self._update_market_regime(symbols, market_data)
        
        # Calculate market average for relative strength
        market_returns = self._calculate_market_returns(symbols, market_data)
        
        for symbol in symbols:
            try:
                # Get signals from strategies
                realistic_signals = self.realistic_strategy.generate_signals(symbol)
                hier_signals = self.hierarchical_strategy.generate_signals_sequential(symbol, '4h')
                
                if realistic_signals.empty or hier_signals.empty:
                    continue
                
                # Get latest data
                latest_realistic = realistic_signals.iloc[-1]
                latest_hier = hier_signals.iloc[-1]
                current_price = latest_realistic.get('close', 0)
                
                # Calculate ATR for risk management
                atr = self._calculate_atr(realistic_signals)
                
                # Enhanced scoring components
                momentum_score = self._calculate_enhanced_momentum(realistic_signals)
                trend_score = self._calculate_enhanced_trend(realistic_signals, latest_realistic)
                volatility_score = self._calculate_enhanced_volatility(realistic_signals, atr)
                relative_strength = self._calculate_relative_strength(realistic_signals, market_returns)
                indicator_score = self._calculate_enhanced_indicators(latest_realistic, latest_hier)
                volume_score = self._calculate_volume_score(realistic_signals)
                ml_score = self._calculate_ml_score(symbol, realistic_signals)
                
                # Kelly Criterion calculation
                kelly_fraction = self._calculate_kelly_fraction(realistic_signals)
                
                # Calculate total score
                total_score = (
                    self.weights['momentum'] * momentum_score +
                    self.weights['trend'] * trend_score +
                    self.weights['volatility'] * volatility_score +
                    self.weights['relative_strength'] * relative_strength +
                    self.weights['indicators'] * indicator_score +
                    self.weights['volume'] * volume_score
                )
                
                # Add ML bonus if available
                if ml_score > 0:
                    total_score = total_score * 0.9 + ml_score * 0.1
                
                scores.append(EnhancedStockScore(
                    symbol=symbol,
                    total_score=total_score,
                    momentum_score=momentum_score,
                    trend_score=trend_score,
                    volatility_score=volatility_score,
                    relative_strength=relative_strength,
                    indicator_score=indicator_score,
                    ml_score=ml_score,
                    volume_score=volume_score,
                    current_price=current_price,
                    atr=atr,
                    signals={
                        'realistic': latest_realistic.get('signal', 0),
                        'hierarchical': latest_hier.get('signal', 0)
                    },
                    kelly_fraction=kelly_fraction
                ))
                
            except Exception as e:
                logger.error(f"Error calculating enhanced score for {symbol}: {e}")
                continue
        
        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _calculate_enhanced_momentum(self, signals_df: pd.DataFrame) -> float:
        """Enhanced momentum calculation with multiple timeframes"""
        if len(signals_df) < 20:
            return 0.5
        
        try:
            prices = signals_df['close'].values
            
            # Multi-timeframe momentum
            momentum_scores = []
            
            # Short term (5 days)
            if len(prices) >= 5:
                ret_5d = (prices[-1] / prices[-5] - 1)
                momentum_scores.append(ret_5d * 3)  # Higher weight
            
            # Medium term (10 days)
            if len(prices) >= 10:
                ret_10d = (prices[-1] / prices[-10] - 1)
                momentum_scores.append(ret_10d * 2)
            
            # Longer term (20 days)
            if len(prices) >= 20:
                ret_20d = (prices[-1] / prices[-20] - 1)
                momentum_scores.append(ret_20d * 1)
            
            # Rate of change
            if len(prices) >= 10:
                roc = ((prices[-1] - prices[-11]) / prices[-11]) * 100  # 10-day ROC
                momentum_scores.append(roc / 10)  # Normalize
            
            # Weighted average
            if momentum_scores:
                weighted_momentum = sum(momentum_scores) / len(momentum_scores)
                return np.clip((weighted_momentum + 0.1) / 0.2, 0, 1)
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_enhanced_trend(self, signals_df: pd.DataFrame, latest_data: pd.Series) -> float:
        """Enhanced trend calculation with multiple indicators"""
        try:
            prices = signals_df['close'].values
            
            if len(prices) < 20:
                return 0.5
            
            trend_scores = []
            
            # ADX trend strength
            adx = latest_data.get('adx', 0)
            plus_di = latest_data.get('plus_di', 0)
            minus_di = latest_data.get('minus_di', 0)
            
            if adx > 25 and plus_di > minus_di:
                adx_score = min(adx / 50, 1.0)
                di_ratio = plus_di / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0.5
                trend_scores.append(adx_score * di_ratio)
            
            # Moving average alignment
            if len(prices) >= 50:
                ma_10 = np.mean(prices[-10:])
                ma_20 = np.mean(prices[-20:])
                ma_50 = np.mean(prices[-50:])
                
                if ma_10 > ma_20 > ma_50:
                    trend_scores.append(0.9)
                elif ma_10 > ma_20:
                    trend_scores.append(0.7)
                else:
                    trend_scores.append(0.3)
            
            # MACD trend
            if 'macd_hist' in latest_data:
                macd_hist = latest_data['macd_hist']
                if macd_hist > 0:
                    trend_scores.append(0.7)
                else:
                    trend_scores.append(0.3)
            
            # Average all trend scores
            if trend_scores:
                return sum(trend_scores) / len(trend_scores)
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_enhanced_volatility(self, signals_df: pd.DataFrame, atr: float) -> float:
        """Enhanced volatility score with ATR consideration"""
        if len(signals_df) < 20:
            return 0.5
        
        try:
            returns = signals_df['close'].pct_change().dropna()
            
            # Historical volatility
            hist_vol = returns.tail(20).std()
            
            # ATR-based volatility
            if 'close' in signals_df.columns:
                close_price = signals_df['close'].iloc[-1]
                atr_vol = atr / close_price if close_price > 0 else 0.02
            else:
                atr_vol = 0.02
            
            # Combine both measures
            combined_vol = (hist_vol + atr_vol) / 2
            
            # Lower volatility = higher score
            # Assuming 2% daily volatility is average
            vol_score = 1 - min(combined_vol / 0.03, 1.0)
            
            return vol_score
            
        except:
            return 0.5
    
    def _calculate_relative_strength(self, signals_df: pd.DataFrame, market_returns: float) -> float:
        """Calculate relative strength vs market"""
        if len(signals_df) < 20:
            return 0.5
        
        try:
            # Stock returns
            stock_return = (signals_df['close'].iloc[-1] / signals_df['close'].iloc[-20] - 1)
            
            # Relative strength
            if market_returns != 0:
                rs = stock_return / market_returns
                # Normalize: RS > 1.2 = strong, RS < 0.8 = weak
                return np.clip((rs - 0.8) / 0.4, 0, 1)
            else:
                return 0.5 + stock_return  # Fallback
                
        except:
            return 0.5
    
    def _calculate_enhanced_indicators(self, realistic_data: pd.Series, hier_data: pd.Series) -> float:
        """Enhanced indicator scoring with more signals"""
        score = 0
        weight = 0
        
        # Strategy signals
        if realistic_data.get('signal', 0) == 1:
            score += 0.3
            weight += 0.3
        
        if hier_data.get('signal', 0) == 1:
            score += 0.3
            weight += 0.3
        
        # Individual indicators
        indicators = {
            'st_trend': (1, 0.15),           # Supertrend
            'sq_on': (False, 0.1),           # Squeeze off
            'wt_cross_up': (True, 0.1),      # WaveTrend cross
            'macd_hist': (lambda x: x > 0, 0.05),  # MACD positive
        }
        
        for ind, (target, w) in indicators.items():
            if ind in realistic_data:
                value = realistic_data[ind]
                if callable(target):
                    if target(value):
                        score += w
                elif value == target:
                    score += w
                weight += w
        
        # Normalize
        return score / weight if weight > 0 else 0.5
    
    def _calculate_volume_score(self, signals_df: pd.DataFrame) -> float:
        """Calculate volume-based score"""
        if 'volume' not in signals_df.columns or len(signals_df) < 20:
            return 0.5
        
        try:
            volumes = signals_df['volume'].values
            
            # Recent vs average volume
            recent_vol = volumes[-5:].mean()
            avg_vol = volumes[-20:].mean()
            
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                # Higher recent volume is better
                return np.clip(vol_ratio / 2, 0, 1)
            
            return 0.5
            
        except:
            return 0.5
    
    def _calculate_ml_score(self, symbol: str, signals_df: pd.DataFrame) -> float:
        """Placeholder for ML model integration"""
        # TODO: Integrate with trained XGBoost/LSTM models
        return 0
    
    def _calculate_atr(self, signals_df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(signals_df) < period:
            return signals_df['close'].iloc[-1] * 0.02  # Default 2%
        
        try:
            # Simple ATR calculation without talib
            close = signals_df['close'].values
            
            # Use price volatility as ATR proxy
            if 'high' in signals_df and 'low' in signals_df:
                high = signals_df['high'].values
                low = signals_df['low'].values
                
                # True Range calculation
                tr_list = []
                for i in range(1, len(close)):
                    hl = high[i] - low[i]
                    hc = abs(high[i] - close[i-1])
                    lc = abs(low[i] - close[i-1])
                    tr = max(hl, hc, lc)
                    tr_list.append(tr)
                
                # ATR is average of last 'period' true ranges
                if len(tr_list) >= period:
                    atr = np.mean(tr_list[-period:])
                    return atr
            
            # Fallback: use standard deviation of returns
            returns = signals_df['close'].pct_change().dropna()
            if len(returns) >= period:
                return signals_df['close'].iloc[-1] * returns.tail(period).std() * np.sqrt(period)
            
            return signals_df['close'].iloc[-1] * 0.02
            
        except:
            return signals_df['close'].iloc[-1] * 0.02
    
    def _calculate_kelly_fraction(self, signals_df: pd.DataFrame) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if len(signals_df) < 30:
            return 0.1  # Default conservative
        
        try:
            # Calculate win rate and avg win/loss from price movements
            returns = signals_df['close'].pct_change().dropna()
            
            # Define wins as days with >1% gain
            wins = returns[returns > 0.01]
            losses = returns[returns < -0.01]
            
            if len(wins) > 0 and len(losses) > 0:
                win_rate = len(wins) / (len(wins) + len(losses))
                avg_win = wins.mean()
                avg_loss = abs(losses.mean())
                
                # Kelly formula: f = (p*b - q) / b
                # where p = win rate, q = loss rate, b = win/loss ratio
                if avg_loss > 0:
                    b = avg_win / avg_loss
                    q = 1 - win_rate
                    kelly = (win_rate * b - q) / b
                    
                    # Apply Kelly multiplier for safety
                    return max(0, min(kelly * self.kelly_multiplier, 0.25))
            
            return 0.1
            
        except:
            return 0.1
    
    def _calculate_market_returns(self, symbols: List[str], market_data: Dict = None) -> float:
        """Calculate average market returns"""
        if market_data:
            return market_data.get('market_return', 0)
        
        # Simple average of top symbols
        total_return = 0
        count = 0
        
        for symbol in symbols[:10]:  # Top 10 as proxy
            try:
                signals = self.realistic_strategy.generate_signals(symbol)
                if len(signals) >= 20:
                    ret = (signals['close'].iloc[-1] / signals['close'].iloc[-20] - 1)
                    total_return += ret
                    count += 1
            except:
                continue
        
        return total_return / count if count > 0 else 0
    
    def _update_market_regime(self, symbols: List[str], market_data: Dict = None):
        """Detect current market regime"""
        if market_data and 'regime' in market_data:
            self.market_regime = market_data['regime']
            return
        
        # Simple regime detection based on market breadth
        bullish_count = 0
        bearish_count = 0
        
        for symbol in symbols[:10]:
            try:
                signals = self.realistic_strategy.generate_signals(symbol)
                if len(signals) >= 20:
                    ma_20 = signals['close'].rolling(20).mean().iloc[-1]
                    current = signals['close'].iloc[-1]
                    
                    if current > ma_20:
                        bullish_count += 1
                    else:
                        bearish_count += 1
            except:
                continue
        
        total = bullish_count + bearish_count
        if total > 0:
            bullish_pct = bullish_count / total
            if bullish_pct > 0.7:
                self.market_regime = 'bull'
            elif bullish_pct < 0.3:
                self.market_regime = 'bear'
            else:
                self.market_regime = 'neutral'
    
    def calculate_dynamic_rotation_interval(self, market_volatility: float) -> int:
        """Calculate rotation interval based on market conditions"""
        # High volatility = more frequent rotation
        if market_volatility > 0.03:  # 3% daily vol
            return self.min_rotation_days
        elif market_volatility < 0.01:  # 1% daily vol
            return self.max_rotation_days
        else:
            # Linear interpolation
            return int(self.base_rotation_days)
    
    def enhanced_position_sizing(self, score: EnhancedStockScore, available_capital: float) -> float:
        """Enhanced position sizing with multiple factors"""
        # Base size from score
        score_factor = score.total_score
        base_size = self.base_position_size + (self.max_position_size - self.base_position_size) * score_factor
        
        # Kelly adjustment
        kelly_size = available_capital * score.kelly_fraction
        
        # Take minimum of score-based and Kelly
        position_size = min(base_size * available_capital, kelly_size)
        
        # Market regime adjustment
        regime_factor = self.regime_position_adjustment.get(self.market_regime, 0.8)
        position_size *= regime_factor
        
        # Ensure minimum position size
        min_size = available_capital * 0.05  # 5% minimum
        position_size = max(position_size, min_size)
        
        # Don't exceed available capital
        return min(position_size, available_capital * 0.9)  # Keep 10% buffer
    
    def check_enhanced_exit_conditions(self, position: Dict, current_score: EnhancedStockScore) -> Tuple[bool, str, float]:
        """Enhanced exit logic with partial exits"""
        symbol = position['symbol']
        entry_price = position['entry_price']
        current_price = current_score.current_price
        return_pct = ((current_price - entry_price) / entry_price) * 100
        holding_days = (datetime.now() - position['entry_date']).days
        
        # Dynamic stop loss based on ATR
        stop_loss = entry_price - (current_score.atr * self.dynamic_stop_loss_atr)
        if current_price <= stop_loss:
            return True, 'dynamic_stop_loss', 1.0  # Exit 100%
        
        # Partial profit taking
        if return_pct >= self.partial_exit_threshold * 100 and symbol not in self.partial_exits:
            self.partial_exits[symbol] = True
            return True, 'partial_profit', self.partial_exit_percent  # Exit 50%
        
        # Full profit target based on ATR
        take_profit = entry_price + (current_score.atr * self.profit_target_atr)
        if current_price >= take_profit:
            return True, 'profit_target', 1.0  # Exit 100%
        
        # Time-based exit for underperformers
        if holding_days > self.time_exit_days and return_pct < 5:
            return True, 'time_exit', 1.0  # Exit 100%
        
        # Trend reversal exit
        if current_score.trend_score < 0.3 and return_pct > 0:
            return True, 'trend_reversal', 1.0  # Exit 100%
        
        # Score deterioration exit
        if current_score.total_score < 0.4:
            return True, 'low_score', 1.0  # Exit 100%
        
        return False, '', 0


def create_enhanced_config() -> Dict:
    """Create configuration for enhanced strategy"""
    from typing import Dict
    config = {
        "strategy_name": "Enhanced Dynamic Rotation",
        "version": "2.0",
        "max_positions": 10,
        "position_sizing": {
            "method": "kelly_score_hybrid",
            "base_size": 0.08,
            "max_size": 0.15,
            "kelly_multiplier": 0.25
        },
        "scoring_weights": {
            "momentum": 0.25,
            "trend": 0.20,
            "volatility": 0.15,
            "relative_strength": 0.20,
            "indicators": 0.15,
            "volume": 0.05
        },
        "exit_rules": {
            "dynamic_stop_loss_atr": 2.0,
            "profit_target_atr": 4.0,
            "partial_exit_threshold": 0.10,
            "partial_exit_percent": 0.5,
            "time_exit_days": 30
        },
        "rotation_rules": {
            "base_interval_days": 3,
            "min_interval_days": 1,
            "max_interval_days": 5,
            "dynamic_adjustment": True
        },
        "market_regime": {
            "detection": True,
            "position_adjustment": {
                "bull": 1.0,
                "neutral": 0.8,
                "bear": 0.6
            }
        }
    }
    
    import json
    config_path = Path(__file__).parent / 'enhanced_rotation_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Enhanced config saved to {config_path}")
    return config


if __name__ == "__main__":
    # Test enhanced strategy
    strategy = EnhancedRotationStrategy()
    
    # Test scoring
    scores = strategy.calculate_enhanced_scores(SACRED_SYMBOLS[:5])
    
    print("\n" + "="*60)
    print("ENHANCED ROTATION STRATEGY TEST")
    print("="*60)
    
    for i, score in enumerate(scores):
        print(f"\n{i+1}. {score.symbol}:")
        print(f"   Total Score: {score.total_score:.3f}")
        print(f"   Components: M:{score.momentum_score:.2f} T:{score.trend_score:.2f} "
              f"V:{score.volatility_score:.2f} RS:{score.relative_strength:.2f}")
        print(f"   Kelly Fraction: {score.kelly_fraction:.3f}")
        print(f"   ATR: ${score.atr:.2f}")
    
    print(f"\nMarket Regime: {strategy.market_regime}")
    
    # Create config
    create_enhanced_config()