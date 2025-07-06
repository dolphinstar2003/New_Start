"""
Ensemble Model combining XGBoost, LightGBM, and LSTM
Implements God Mode ensemble weights: XGBoost=0.4, LSTM=0.3, LightGBM=0.2, Technical=0.1
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import joblib

from .xgboost_model import XGBoostTradingModel
from .lightgbm_model import LightGBMTradingModel
from .lstm_model import LSTMTradingModel


class EnsembleTradingModel:
    """Ensemble model combining multiple ML models for trading signals"""
    
    def __init__(self, model_dir: Path):
        """
        Initialize ensemble model
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize individual models
        self.xgb_model = XGBoostTradingModel(model_dir)
        self.lgb_model = LightGBMTradingModel(model_dir)
        self.lstm_model = LSTMTradingModel(model_dir)
        
        # God Mode ensemble weights
        self.ensemble_weights = {
            'xgboost': 0.4,  # Main fast signal model
            'lightgbm': 0.2,  # Secondary fast signal model  
            'lstm': 0.3,      # Deep analysis model
            'technical': 0.1  # Pure technical signals
        }
        
        self.is_trained = False
        self.training_stats = {}
        
        logger.info("Ensemble model initialized with God Mode weights")
    
    def train_all_models(self, features: pd.DataFrame, target: pd.Series, 
                        validation_split: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        Train all ensemble models
        
        Args:
            features: Training features
            target: Training target
            validation_split: Validation split ratio
            **kwargs: Additional parameters for individual models
            
        Returns:
            Combined training statistics
        """
        logger.info("Training ensemble models...")
        
        training_results = {}
        
        # Train XGBoost
        try:
            logger.info("Training XGBoost model...")
            xgb_stats = self.xgb_model.train(features, target, validation_split, **kwargs.get('xgboost', {}))
            training_results['xgboost'] = xgb_stats
            logger.info("✅ XGBoost training completed")
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
            training_results['xgboost'] = {'error': str(e)}
        
        # Train LightGBM
        try:
            logger.info("Training LightGBM model...")
            lgb_stats = self.lgb_model.train(features, target, validation_split, **kwargs.get('lightgbm', {}))
            training_results['lightgbm'] = lgb_stats
            logger.info("✅ LightGBM training completed")
        except Exception as e:
            logger.error(f"❌ LightGBM training failed: {e}")
            training_results['lightgbm'] = {'error': str(e)}
        
        # Train LSTM
        try:
            logger.info("Training LSTM model...")
            lstm_stats = self.lstm_model.train(features, target, validation_split, **kwargs.get('lstm', {}))
            training_results['lstm'] = lstm_stats
            logger.info("✅ LSTM training completed")
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
            training_results['lstm'] = {'error': str(e)}
        
        # Check which models trained successfully
        successful_models = [model for model, stats in training_results.items() if 'error' not in stats]
        failed_models = [model for model, stats in training_results.items() if 'error' in stats]
        
        if successful_models:
            self.is_trained = True
            logger.info(f"✅ Ensemble training completed. Successful: {successful_models}")
            if failed_models:
                logger.warning(f"⚠️ Failed models: {failed_models}")
        else:
            logger.error("❌ All models failed to train")
        
        self.training_stats = training_results
        return training_results
    
    def _get_technical_signals(self, features: pd.DataFrame) -> np.ndarray:
        """
        Generate pure technical signals (simple rule-based)
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Technical signal probabilities [sell_prob, hold_prob, buy_prob]
        """
        try:
            signals = np.zeros((len(features), 3))  # [sell, hold, buy] probabilities
            
            # Simple technical rules based on available features
            for i, (idx, row) in enumerate(features.iterrows()):
                buy_signals = 0
                sell_signals = 0
                total_signals = 0
                
                # Supertrend signals
                if 'supertrend_signal' in row and not pd.isna(row['supertrend_signal']):
                    total_signals += 1
                    if row['supertrend_signal'] > 0:
                        buy_signals += 1
                    elif row['supertrend_signal'] < 0:
                        sell_signals += 1
                
                # ADX trend strength
                if 'adx' in row and 'di_diff' in row:
                    if not pd.isna(row['adx']) and not pd.isna(row['di_diff']):
                        if row['adx'] > 25:  # Strong trend
                            total_signals += 1
                            if row['di_diff'] > 0:
                                buy_signals += 1
                            else:
                                sell_signals += 1
                
                # WaveTrend signals
                if 'wt_buy_signal' in row and 'wt_sell_signal' in row:
                    if not pd.isna(row['wt_buy_signal']) and row['wt_buy_signal']:
                        buy_signals += 1
                        total_signals += 1
                    elif not pd.isna(row['wt_sell_signal']) and row['wt_sell_signal']:
                        sell_signals += 1
                        total_signals += 1
                
                # MACD signals
                if 'macd_cross_up' in row and 'macd_cross_down' in row:
                    if not pd.isna(row['macd_cross_up']) and row['macd_cross_up']:
                        buy_signals += 1
                        total_signals += 1
                    elif not pd.isna(row['macd_cross_down']) and row['macd_cross_down']:
                        sell_signals += 1
                        total_signals += 1
                
                # Calculate probabilities
                if total_signals > 0:
                    buy_prob = buy_signals / total_signals
                    sell_prob = sell_signals / total_signals
                    hold_prob = 1 - buy_prob - sell_prob
                    
                    signals[i] = [sell_prob, hold_prob, buy_prob]
                else:
                    # Default to hold if no signals
                    signals[i] = [0.0, 1.0, 0.0]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")
            # Return default hold signals
            return np.array([[0.0, 1.0, 0.0]] * len(features))
    
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble prediction probabilities
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Ensemble probabilities [sell_prob, hold_prob, buy_prob]
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before prediction")
        
        ensemble_proba = np.zeros((len(features), 3))
        total_weight = 0.0
        
        # Get XGBoost predictions
        if self.xgb_model.is_trained:
            try:
                xgb_proba = self.xgb_model.predict_proba(features)
                if len(xgb_proba) == len(features):
                    ensemble_proba += self.ensemble_weights['xgboost'] * xgb_proba
                    total_weight += self.ensemble_weights['xgboost']
                    logger.debug("✅ XGBoost predictions added to ensemble")
            except Exception as e:
                logger.warning(f"⚠️ XGBoost prediction failed: {e}")
        
        # Get LightGBM predictions
        if self.lgb_model.is_trained:
            try:
                lgb_proba = self.lgb_model.predict_proba(features)
                if len(lgb_proba) == len(features):
                    ensemble_proba += self.ensemble_weights['lightgbm'] * lgb_proba
                    total_weight += self.ensemble_weights['lightgbm']
                    logger.debug("✅ LightGBM predictions added to ensemble")
            except Exception as e:
                logger.warning(f"⚠️ LightGBM prediction failed: {e}")
        
        # Get LSTM predictions
        if self.lstm_model.is_trained:
            try:
                lstm_proba = self.lstm_model.predict_proba(features)
                if len(lstm_proba) == len(features):
                    ensemble_proba += self.ensemble_weights['lstm'] * lstm_proba
                    total_weight += self.ensemble_weights['lstm']
                    logger.debug("✅ LSTM predictions added to ensemble")
            except Exception as e:
                logger.warning(f"⚠️ LSTM prediction failed: {e}")
        
        # Get technical signals
        try:
            tech_proba = self._get_technical_signals(features)
            ensemble_proba += self.ensemble_weights['technical'] * tech_proba
            total_weight += self.ensemble_weights['technical']
            logger.debug("✅ Technical signals added to ensemble")
        except Exception as e:
            logger.warning(f"⚠️ Technical signals failed: {e}")
        
        # Normalize if total weight is different from 1.0
        if total_weight > 0:
            ensemble_proba /= total_weight
        else:
            # Fallback to hold signal
            ensemble_proba = np.array([[0.0, 1.0, 0.0]] * len(features))
            logger.warning("⚠️ No predictions available, defaulting to HOLD")
        
        return ensemble_proba
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble predictions
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predicted labels
        """
        probabilities = self.predict_proba(features)
        return np.argmax(probabilities, axis=1)
    
    def get_prediction_confidence(self, features: pd.DataFrame) -> np.ndarray:
        """Get prediction confidence scores"""
        probabilities = self.predict_proba(features)
        return np.max(probabilities, axis=1)
    
    def get_model_consensus(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed consensus information from all models
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Dictionary with individual model predictions and consensus
        """
        consensus = {
            'individual_predictions': {},
            'individual_probabilities': {},
            'ensemble_probabilities': None,
            'ensemble_prediction': None,
            'consensus_strength': 0.0
        }
        
        # Get individual model predictions
        if self.xgb_model.is_trained:
            try:
                consensus['individual_predictions']['xgboost'] = self.xgb_model.predict(features)
                consensus['individual_probabilities']['xgboost'] = self.xgb_model.predict_proba(features)
            except Exception as e:
                logger.warning(f"XGBoost consensus failed: {e}")
        
        if self.lgb_model.is_trained:
            try:
                consensus['individual_predictions']['lightgbm'] = self.lgb_model.predict(features)
                consensus['individual_probabilities']['lightgbm'] = self.lgb_model.predict_proba(features)
            except Exception as e:
                logger.warning(f"LightGBM consensus failed: {e}")
        
        if self.lstm_model.is_trained:
            try:
                consensus['individual_predictions']['lstm'] = self.lstm_model.predict(features)
                consensus['individual_probabilities']['lstm'] = self.lstm_model.predict_proba(features)
            except Exception as e:
                logger.warning(f"LSTM consensus failed: {e}")
        
        # Get ensemble predictions
        try:
            consensus['ensemble_probabilities'] = self.predict_proba(features)
            consensus['ensemble_prediction'] = self.predict(features)
            
            # Calculate consensus strength (agreement between models)
            predictions = list(consensus['individual_predictions'].values())
            if len(predictions) >= 2:
                # Count agreements
                agreements = 0
                total_comparisons = 0
                
                for i in range(len(predictions)):
                    for j in range(i + 1, len(predictions)):
                        if len(predictions[i]) == len(predictions[j]):
                            agreements += np.sum(predictions[i] == predictions[j])
                            total_comparisons += len(predictions[i])
                
                if total_comparisons > 0:
                    consensus['consensus_strength'] = agreements / total_comparisons
                    
        except Exception as e:
            logger.error(f"Ensemble consensus failed: {e}")
        
        return consensus
    
    def save_ensemble(self, filename: str = "ensemble_model") -> Path:
        """Save ensemble model"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before saving")
        
        # Save individual models
        model_paths = {}
        
        if self.xgb_model.is_trained:
            model_paths['xgboost'] = self.xgb_model.save_model(f"{filename}_xgboost.joblib")
        
        if self.lgb_model.is_trained:
            model_paths['lightgbm'] = self.lgb_model.save_model(f"{filename}_lightgbm.joblib")
        
        if self.lstm_model.is_trained:
            model_paths['lstm'] = self.lstm_model.save_model(f"{filename}_lstm")
        
        # Save ensemble metadata
        ensemble_metadata = {
            'ensemble_weights': self.ensemble_weights,
            'training_stats': self.training_stats,
            'model_paths': model_paths,
            'is_trained': self.is_trained
        }
        
        metadata_path = self.model_dir / f"{filename}_ensemble_metadata.joblib"
        joblib.dump(ensemble_metadata, metadata_path)
        
        logger.info(f"Ensemble model saved. Metadata: {metadata_path}")
        return metadata_path
    
    def load_ensemble(self, metadata_path: Path) -> None:
        """Load ensemble model"""
        if not metadata_path.exists():
            raise FileNotFoundError(f"Ensemble metadata not found: {metadata_path}")
        
        # Load metadata
        metadata = joblib.load(metadata_path)
        self.ensemble_weights = metadata['ensemble_weights']
        self.training_stats = metadata['training_stats']
        
        # Load individual models
        model_paths = metadata['model_paths']
        
        if 'xgboost' in model_paths and Path(model_paths['xgboost']).exists():
            self.xgb_model.load_model(Path(model_paths['xgboost']))
        
        if 'lightgbm' in model_paths and Path(model_paths['lightgbm']).exists():
            self.lgb_model.load_model(Path(model_paths['lightgbm']))
        
        if 'lstm' in model_paths:
            lstm_path = Path(model_paths['lstm'])
            if lstm_path.exists():
                self.lstm_model.load_model(lstm_path)
        
        self.is_trained = metadata['is_trained']
        logger.info(f"Ensemble model loaded from {metadata_path}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        summary = {
            'ensemble_weights': self.ensemble_weights,
            'is_trained': self.is_trained,
            'models_status': {
                'xgboost': self.xgb_model.is_trained,
                'lightgbm': self.lgb_model.is_trained,
                'lstm': self.lstm_model.is_trained
            },
            'training_stats': self.training_stats
        }
        
        # Add individual model summaries
        if self.xgb_model.is_trained:
            summary['xgboost_summary'] = self.xgb_model.get_model_summary()
        
        if self.lgb_model.is_trained:
            summary['lightgbm_summary'] = self.lgb_model.get_model_summary()
        
        if self.lstm_model.is_trained:
            summary['lstm_summary'] = self.lstm_model.get_model_summary()
        
        return summary