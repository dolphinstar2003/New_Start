"""
Trading Signal Generation System
Combines ML ensemble with technical analysis for signal generation
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

from ml_models.ensemble_model import EnsembleTradingModel
from ml_models.feature_engineer import FeatureEngineer
from config.settings import SACRED_SYMBOLS, DATA_DIR, SIGNAL_STRENGTH


class TradingSignalGenerator:
    """Generate trading signals using ML ensemble and technical analysis"""
    
    def __init__(self, data_dir: Path, model_dir: Path):
        """
        Initialize signal generator
        
        Args:
            data_dir: Path to data directory
            model_dir: Path to model directory
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(data_dir)
        self.ensemble_model = EnsembleTradingModel(model_dir)
        
        # Signal thresholds (God Mode configuration)
        self.signal_thresholds = {
            'strong_buy': 0.7,      # High confidence buy
            'weak_buy': 0.55,       # Low confidence buy
            'hold': 0.45,           # Hold/neutral zone
            'weak_sell': 0.45,      # Low confidence sell
            'strong_sell': 0.3      # High confidence sell
        }
        
        # Signal strength mapping from God Mode
        self.signal_mapping = {
            'STRONG_BUY': 'ML sinyali + Teknik onay',
            'STRONG_SELL': 'ML sinyali + Teknik onay', 
            'WEAK_BUY': 'Sadece ML sinyali',
            'WEAK_SELL': 'Sadece ML sinyali',
            'HOLD': 'Sadece teknik sinyal'
        }
        
        logger.info("Trading Signal Generator initialized")
    
    def _calculate_technical_confirmation(self, features: pd.DataFrame) -> pd.Series:
        """
        Calculate technical confirmation score
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Technical confirmation scores (-1 to 1)
        """
        confirmation = pd.Series(0.0, index=features.index)
        
        try:
            # Supertrend confirmation
            if 'supertrend_signal' in features.columns:
                st_signal = features['supertrend_signal'].fillna(0)
                confirmation += 0.3 * st_signal  # 30% weight
            
            # ADX trend strength confirmation
            if 'adx' in features.columns and 'di_diff' in features.columns:
                adx_strong = features['adx'].fillna(0) > 25
                di_direction = np.sign(features['di_diff'].fillna(0))
                adx_confirmation = np.where(adx_strong, di_direction, 0) * 0.25
                confirmation += adx_confirmation  # 25% weight
            
            # MACD confirmation
            if 'macd_above_zero' in features.columns:
                macd_signal = features['macd_above_zero'].fillna(0).astype(int) * 2 - 1  # -1 or 1
                confirmation += 0.2 * macd_signal  # 20% weight
            
            # WaveTrend confirmation
            if 'wt1' in features.columns:
                wt_bullish = (features['wt1'].fillna(0) > -50).astype(int) * 2 - 1
                confirmation += 0.15 * wt_bullish  # 15% weight
            
            # Squeeze momentum confirmation
            if 'momentum_positive' in features.columns:
                squeeze_signal = features['momentum_positive'].fillna(0).astype(int) * 2 - 1
                confirmation += 0.1 * squeeze_signal  # 10% weight
            
            # Normalize to -1 to 1 range
            confirmation = np.clip(confirmation, -1, 1)
            
        except Exception as e:
            logger.error(f"Error calculating technical confirmation: {e}")
            confirmation = pd.Series(0.0, index=features.index)
        
        return confirmation
    
    def _classify_signal_strength(self, ml_probabilities: np.ndarray, 
                                technical_confirmation: pd.Series) -> List[str]:
        """
        Classify signal strength based on ML probabilities and technical confirmation
        
        Args:
            ml_probabilities: ML model probabilities [sell, hold, buy]
            technical_confirmation: Technical confirmation scores
            
        Returns:
            List of signal strength classifications
        """
        signals = []
        
        for i, (proba, tech_conf) in enumerate(zip(ml_probabilities, technical_confirmation)):
            sell_prob, hold_prob, buy_prob = proba
            
            # Determine base ML signal
            if buy_prob > self.signal_thresholds['strong_buy']:
                ml_signal = 'BUY'
                ml_strength = 'STRONG'
            elif buy_prob > self.signal_thresholds['weak_buy']:
                ml_signal = 'BUY'
                ml_strength = 'WEAK'
            elif sell_prob > self.signal_thresholds['strong_sell']:
                ml_signal = 'SELL'
                ml_strength = 'STRONG'
            elif sell_prob > self.signal_thresholds['weak_sell']:
                ml_signal = 'SELL'
                ml_strength = 'WEAK'
            else:
                ml_signal = 'HOLD'
                ml_strength = 'NEUTRAL'
            
            # Check technical confirmation
            tech_agrees = False
            if ml_signal == 'BUY' and tech_conf > 0.3:
                tech_agrees = True
            elif ml_signal == 'SELL' and tech_conf < -0.3:
                tech_agrees = True
            elif ml_signal == 'HOLD':
                tech_agrees = True  # Technical doesn't contradict hold
            
            # Final signal classification
            if ml_signal == 'BUY':
                if ml_strength == 'STRONG' and tech_agrees:
                    final_signal = 'STRONG_BUY'
                else:
                    final_signal = 'WEAK_BUY'
            elif ml_signal == 'SELL':
                if ml_strength == 'STRONG' and tech_agrees:
                    final_signal = 'STRONG_SELL'
                else:
                    final_signal = 'WEAK_SELL'
            else:
                final_signal = 'HOLD'
            
            signals.append(final_signal)
        
        return signals
    
    def generate_signals_for_symbol(self, symbol: str, 
                                  lookback_days: int = 50) -> pd.DataFrame:
        """
        Generate trading signals for a single symbol
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of recent days to analyze
            
        Returns:
            DataFrame with signals and metadata
        """
        logger.info(f"Generating signals for {symbol}")
        
        try:
            # Build feature matrix
            features, targets = self.feature_engineer.build_feature_matrix(symbol)
            
            if features.empty:
                logger.warning(f"No features available for {symbol}")
                return pd.DataFrame()
            
            # Use recent data only
            if lookback_days > 0:
                features = features.tail(lookback_days)
            
            # Get ML predictions
            if not self.ensemble_model.is_trained:
                logger.warning("Ensemble model not trained, using technical signals only")
                ml_probabilities = np.array([[0.0, 1.0, 0.0]] * len(features))  # All hold
            else:
                ml_probabilities = self.ensemble_model.predict_proba(features)
            
            # Calculate technical confirmation
            technical_confirmation = self._calculate_technical_confirmation(features)
            
            # Get prediction confidence
            if self.ensemble_model.is_trained:
                confidence_scores = self.ensemble_model.get_prediction_confidence(features)
            else:
                confidence_scores = np.ones(len(features)) * 0.5
            
            # Classify signal strength
            signal_strengths = self._classify_signal_strength(ml_probabilities, technical_confirmation)
            
            # Create results DataFrame
            results = pd.DataFrame(index=features.index)
            results['symbol'] = symbol
            results['signal'] = signal_strengths
            results['ml_buy_prob'] = ml_probabilities[:, 2]
            results['ml_sell_prob'] = ml_probabilities[:, 0]
            results['ml_hold_prob'] = ml_probabilities[:, 1]
            results['technical_confirmation'] = technical_confirmation
            results['confidence'] = confidence_scores
            
            # Add signal metadata
            results['signal_description'] = results['signal'].map(self.signal_mapping)
            
            # Add risk metrics
            results['signal_risk'] = 'MEDIUM'
            results.loc[results['confidence'] > 0.8, 'signal_risk'] = 'LOW'
            results.loc[results['confidence'] < 0.6, 'signal_risk'] = 'HIGH'
            
            # Add price context if available
            price_data = self.feature_engineer.load_symbol_data(symbol)
            if 'price' in price_data and not price_data['price'].empty:
                price_df = price_data['price'].reindex(results.index, method='ffill')
                results['current_price'] = price_df['close']
                results['volume'] = price_df['volume']
                
                # Calculate price changes
                results['price_change_1d'] = price_df['close'].pct_change()
                results['price_change_5d'] = price_df['close'].pct_change(5)
            
            logger.info(f"Generated {len(results)} signals for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return pd.DataFrame()
    
    def generate_signals_for_all_symbols(self, symbols: List[str] = None,
                                       lookback_days: int = 50) -> Dict[str, pd.DataFrame]:
        """
        Generate signals for multiple symbols
        
        Args:
            symbols: List of symbols (uses SACRED_SYMBOLS if None)
            lookback_days: Number of recent days to analyze
            
        Returns:
            Dictionary of symbol -> signals DataFrame
        """
        if symbols is None:
            symbols = SACRED_SYMBOLS
        
        logger.info(f"Generating signals for {len(symbols)} symbols")
        
        all_signals = {}
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
            
            signals = self.generate_signals_for_symbol(symbol, lookback_days)
            if not signals.empty:
                all_signals[symbol] = signals
        
        logger.info(f"Signal generation completed for {len(all_signals)} symbols")
        return all_signals
    
    def get_latest_signals(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        Get latest signals for all symbols
        
        Args:
            symbols: List of symbols
            
        Returns:
            DataFrame with latest signals
        """
        all_signals = self.generate_signals_for_all_symbols(symbols, lookback_days=1)
        
        latest_signals = []
        
        for symbol, signals_df in all_signals.items():
            if not signals_df.empty:
                latest = signals_df.iloc[-1].copy()
                latest['timestamp'] = signals_df.index[-1]
                latest['symbol'] = symbol
                latest_signals.append(latest)
        
        if latest_signals:
            result_df = pd.DataFrame(latest_signals)
            result_df = result_df.sort_values('confidence', ascending=False)
            return result_df
        else:
            return pd.DataFrame()
    
    def filter_actionable_signals(self, signals_df: pd.DataFrame,
                                min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Filter signals for actionable trading opportunities
        
        Args:
            signals_df: Signals DataFrame
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered DataFrame with actionable signals
        """
        if signals_df.empty:
            return signals_df
        
        # Filter by confidence
        actionable = signals_df[signals_df['confidence'] >= min_confidence].copy()
        
        # Filter out HOLD signals for active trading
        actionable = actionable[actionable['signal'] != 'HOLD']
        
        # Sort by confidence and signal strength
        signal_priority = {
            'STRONG_BUY': 5,
            'STRONG_SELL': 4,
            'WEAK_BUY': 3,
            'WEAK_SELL': 2,
            'HOLD': 1
        }
        
        actionable['signal_priority'] = actionable['signal'].map(signal_priority)
        actionable = actionable.sort_values(['signal_priority', 'confidence'], ascending=[False, False])
        
        return actionable
    
    def save_signals(self, signals_dict: Dict[str, pd.DataFrame], 
                    filename: str = None) -> Path:
        """
        Save signals to CSV files
        
        Args:
            signals_dict: Dictionary of symbol -> signals
            filename: Optional filename prefix
            
        Returns:
            Path to saved directory
        """
        if filename is None:
            filename = f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        signals_dir = self.data_dir / 'signals'
        signals_dir.mkdir(exist_ok=True)
        
        # Save individual symbol signals
        for symbol, signals_df in signals_dict.items():
            if not signals_df.empty:
                filepath = signals_dir / f"{filename}_{symbol}.csv"
                signals_df.to_csv(filepath)
        
        # Save combined latest signals
        latest_signals = self.get_latest_signals(list(signals_dict.keys()))
        if not latest_signals.empty:
            latest_path = signals_dir / f"{filename}_latest.csv"
            latest_signals.to_csv(latest_path, index=False)
        
        logger.info(f"Signals saved to {signals_dir}")
        return signals_dir
    
    def get_signal_summary(self, signals_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary statistics of generated signals"""
        summary = {
            'total_symbols': len(signals_dict),
            'signal_distribution': {},
            'confidence_stats': {},
            'latest_signals': {}
        }
        
        # Collect all signals
        all_signals = []
        for symbol, signals_df in signals_dict.items():
            if not signals_df.empty:
                latest_signal = signals_df.iloc[-1]
                all_signals.append({
                    'symbol': symbol,
                    'signal': latest_signal['signal'],
                    'confidence': latest_signal['confidence']
                })
        
        if all_signals:
            signals_df = pd.DataFrame(all_signals)
            
            # Signal distribution
            summary['signal_distribution'] = signals_df['signal'].value_counts().to_dict()
            
            # Confidence statistics
            summary['confidence_stats'] = {
                'mean': float(signals_df['confidence'].mean()),
                'median': float(signals_df['confidence'].median()),
                'min': float(signals_df['confidence'].min()),
                'max': float(signals_df['confidence'].max())
            }
            
            # Latest strong signals
            strong_signals = signals_df[signals_df['signal'].isin(['STRONG_BUY', 'STRONG_SELL'])]
            summary['latest_signals'] = {
                'strong_buy': strong_signals[strong_signals['signal'] == 'STRONG_BUY']['symbol'].tolist(),
                'strong_sell': strong_signals[strong_signals['signal'] == 'STRONG_SELL']['symbol'].tolist(),
                'high_confidence': signals_df[signals_df['confidence'] > 0.8]['symbol'].tolist()
            }
        
        return summary