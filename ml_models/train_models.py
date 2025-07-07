"""
Train ML Models for BIST100 Trading System
Standalone training module with timezone fixes
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR, ML_CONFIG
from ml_models.feature_engineer import FeatureEngineer
from ml_models.ensemble_model import EnsembleTradingModel


class ModelTrainer:
    """Train and save ML models for trading"""
    
    def __init__(self, data_dir: Path = None, model_dir: Path = None):
        """
        Initialize model trainer
        
        Args:
            data_dir: Data directory path
            model_dir: Model save directory path
        """
        self.data_dir = data_dir or DATA_DIR
        self.model_dir = model_dir or (DATA_DIR.parent / 'ml_models' / 'saved_models')
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(self.data_dir)
        self.ensemble_model = EnsembleTradingModel(self.model_dir)
        
        logger.info(f"Model Trainer initialized")
        logger.info(f"Data dir: {self.data_dir}")
        logger.info(f"Model dir: {self.model_dir}")
    
    def prepare_training_data(self, 
                            start_date: str = None,
                            end_date: str = None,
                            symbols: list = None) -> tuple:
        """
        Prepare training data from all symbols
        
        Args:
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (YYYY-MM-DD)
            symbols: List of symbols to use (default: SACRED_SYMBOLS)
            
        Returns:
            Tuple of (features, targets)
        """
        if symbols is None:
            symbols = SACRED_SYMBOLS
        
        logger.info(f"Preparing training data for {len(symbols)} symbols")
        
        all_features = []
        all_targets = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                # Build feature matrix
                features, targets = self.feature_engineer.build_feature_matrix(symbol)
                
                if features.empty or targets.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # Fix timezone issues - convert to timezone-naive
                if features.index.tz is not None:
                    features.index = features.index.tz_localize(None)
                if targets.index.tz is not None:
                    targets.index = targets.index.tz_localize(None)
                
                # Filter by date range if specified
                if start_date:
                    start_dt = pd.to_datetime(start_date)
                    mask = features.index >= start_dt
                    features = features[mask]
                    targets = targets[mask]
                
                if end_date:
                    end_dt = pd.to_datetime(end_date)
                    mask = features.index <= end_dt
                    features = features[mask]
                    targets = targets[mask]
                
                # Ensure we have enough data
                if len(features) < 100:
                    logger.warning(f"Insufficient data for {symbol}: {len(features)} samples")
                    continue
                
                # Select target column - use 3-day signal for better stability
                if 'signal_3d' in targets.columns:
                    target_col = 'signal_3d'
                elif 'signal_1d' in targets.columns:
                    target_col = 'signal_1d'
                elif 'up_3d' in targets.columns:
                    target_col = 'up_3d'
                else:
                    target_col = targets.columns[0]
                
                # Add to combined dataset
                all_features.append(features)
                all_targets.append(targets[target_col])
                
                logger.info(f"  Added {len(features)} samples from {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No training data available")
        
        # Combine all data
        logger.info("Combining training data...")
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        # Ensure same length
        min_len = min(len(combined_features), len(combined_targets))
        combined_features = combined_features.iloc[:min_len]
        combined_targets = combined_targets.iloc[:min_len]
        
        logger.info(f"Before NaN removal: {len(combined_features)} samples")
        
        # Remove any remaining NaN values
        mask = ~(combined_features.isna().any(axis=1) | combined_targets.isna())
        logger.info(f"NaN mask sum: {mask.sum()} valid samples")
        
        combined_features = combined_features[mask]
        combined_targets = combined_targets[mask]
        
        # Fill any remaining NaN values as backup
        if len(combined_features) == 0:
            logger.warning("All samples removed by NaN filter, using fillna as backup")
            # Reset without mask
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)
            min_len = min(len(combined_features), len(combined_targets))
            combined_features = combined_features.iloc[:min_len]
            combined_targets = combined_targets.iloc[:min_len]
            
            # Fill NaN values
            combined_features = combined_features.fillna(method='ffill').fillna(0)
            combined_targets = combined_targets.fillna(0)
        
        # Shuffle data (important for training)
        if len(combined_features) > 0:
            shuffle_idx = np.random.permutation(len(combined_features))
            combined_features = combined_features.iloc[shuffle_idx]
            combined_targets = combined_targets.iloc[shuffle_idx]
        
        logger.info(f"Training data prepared: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        # Log class distribution
        if combined_targets.dtype in ['int', 'int64']:
            value_counts = combined_targets.value_counts()
            logger.info("Target distribution:")
            for value, count in value_counts.items():
                logger.info(f"  Class {value}: {count} ({count/len(combined_targets)*100:.1f}%)")
        
        return combined_features, combined_targets
    
    def train_ensemble(self, 
                      features: pd.DataFrame, 
                      targets: pd.Series,
                      validation_split: float = 0.2,
                      optimize_hyperparameters: bool = False) -> dict:
        """
        Train ensemble model
        
        Args:
            features: Training features
            targets: Training targets
            validation_split: Validation data split
            optimize_hyperparameters: Whether to run hyperparameter optimization
            
        Returns:
            Training results dictionary
        """
        logger.info("Training ensemble models...")
        
        # Model-specific parameters
        model_params = {
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            'lightgbm': {
                'n_estimators': 300,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            'lstm': {
                'lstm_units': [64, 32],
                'dropout_rate': 0.3,
                'batch_size': 64,
                'epochs': 30,
                'early_stopping_patience': 10
            }
        }
        
        # Train ensemble
        training_results = self.ensemble_model.train_all_models(
            features, 
            targets, 
            validation_split=validation_split,
            **model_params
        )
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("TRAINING RESULTS")
        logger.info("="*60)
        
        for model_name, stats in training_results.items():
            if 'error' not in stats:
                logger.info(f"\n{model_name.upper()}:")
                if 'validation_metrics' in stats:
                    metrics = stats['validation_metrics']
                    logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
                    logger.info(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
            else:
                logger.error(f"{model_name}: {stats['error']}")
        
        return training_results
    
    def evaluate_models(self, test_features: pd.DataFrame, test_targets: pd.Series) -> dict:
        """
        Evaluate trained models on test data
        
        Args:
            test_features: Test features
            test_targets: Test targets
            
        Returns:
            Evaluation metrics
        """
        if not self.ensemble_model.is_trained:
            raise ValueError("Models must be trained before evaluation")
        
        logger.info("Evaluating models on test data...")
        
        # Get predictions
        predictions = self.ensemble_model.predict(test_features)
        probabilities = self.ensemble_model.predict_proba(test_features)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(test_targets, predictions)
        
        logger.info(f"\nTest Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(test_targets, predictions))
        
        # Get consensus info
        consensus = self.ensemble_model.get_model_consensus(test_features)
        
        evaluation_results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'consensus': consensus,
            'classification_report': classification_report(test_targets, predictions, output_dict=True)
        }
        
        return evaluation_results
    
    def save_trained_models(self, model_name: str = None) -> Path:
        """
        Save trained models
        
        Args:
            model_name: Name for saved model files
            
        Returns:
            Path to saved model
        """
        if model_name is None:
            model_name = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Saving models as '{model_name}'...")
        
        model_path = self.ensemble_model.save_ensemble(model_name)
        
        # Save feature names and metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'symbols_used': SACRED_SYMBOLS,
            'ensemble_weights': self.ensemble_model.ensemble_weights,
            'model_summary': self.ensemble_model.get_ensemble_summary()
        }
        
        import json
        metadata_path = self.model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Models saved to {model_path}")
        
        return model_path
    
    def run_full_training_pipeline(self,
                                 train_start: str = "2022-07-07",  # Start from when all indicators are available
                                 train_end: str = "2024-06-30",
                                 test_split: float = 0.2) -> dict:
        """
        Run complete training pipeline
        
        Args:
            train_start: Training data start date
            train_end: Training data end date
            test_split: Test data split ratio
            
        Returns:
            Pipeline results
        """
        logger.info("="*80)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Training period: {train_start} to {train_end}")
        logger.info(f"Test split: {test_split*100:.0f}%")
        
        try:
            # Step 1: Prepare data
            logger.info("\n📊 STEP 1: Preparing training data...")
            features, targets = self.prepare_training_data(
                start_date=train_start,
                end_date=train_end
            )
            
            # Step 2: Split data
            logger.info("\n📈 STEP 2: Splitting data...")
            split_idx = int(len(features) * (1 - test_split))
            train_features = features.iloc[:split_idx]
            train_targets = targets.iloc[:split_idx]
            test_features = features.iloc[split_idx:]
            test_targets = targets.iloc[split_idx:]
            
            logger.info(f"Training samples: {len(train_features)}")
            logger.info(f"Test samples: {len(test_features)}")
            
            # Step 3: Train models
            logger.info("\n🤖 STEP 3: Training ensemble models...")
            training_results = self.train_ensemble(
                train_features,
                train_targets,
                validation_split=0.2
            )
            
            # Step 4: Evaluate
            logger.info("\n📊 STEP 4: Evaluating models...")
            evaluation_results = self.evaluate_models(test_features, test_targets)
            
            # Step 5: Save models
            logger.info("\n💾 STEP 5: Saving trained models...")
            model_path = self.save_trained_models()
            
            # Final summary
            logger.info("\n" + "="*80)
            logger.info("TRAINING PIPELINE COMPLETED")
            logger.info("="*80)
            logger.info(f"✅ Models trained and saved successfully")
            logger.info(f"📁 Model location: {model_path}")
            logger.info(f"🎯 Test accuracy: {evaluation_results['accuracy']:.4f}")
            
            pipeline_results = {
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'model_path': model_path,
                'data_stats': {
                    'total_samples': len(features),
                    'train_samples': len(train_features),
                    'test_samples': len(test_features),
                    'feature_count': len(features.columns)
                }
            }
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def train_xgboost_model(symbols: list = None):
    """Train XGBoost model specifically for backtesting"""
    trainer = ModelTrainer()
    
    # Prepare data
    features, targets = trainer.prepare_training_data(
        start_date="2023-01-01",
        end_date="2024-12-31",
        symbols=symbols or SACRED_SYMBOLS[:10]
    )
    
    # Train XGBoost
    from ml_models.xgboost_model import XGBoostTradingModel
    xgb_model = XGBoostTradingModel(trainer.model_dir)
    
    # Split data
    split_idx = int(len(features) * 0.8)
    X_train = features.iloc[:split_idx]
    y_train = targets.iloc[:split_idx]
    X_val = features.iloc[split_idx:]
    y_val = targets.iloc[split_idx:]
    
    # Convert targets to 0-based classes for XGBoost
    # -1 -> 0 (sell), 0 -> 1 (hold), 1 -> 2 (buy), 2 -> 3 (strong buy)
    y_train_adj = y_train + 1
    y_val_adj = y_val + 1
    
    # Update model params for correct number of classes
    model = xgb_model._create_model(num_class=4)
    model.fit(X_train, y_train_adj, eval_set=[(X_val, y_val_adj)], verbose=False)
    
    # Save
    import joblib
    model_path = trainer.model_dir / 'xgboost_model.pkl'
    joblib.dump(model, model_path)
    
    return model


def train_lstm_model(symbols: list = None):
    """Train LSTM model specifically for backtesting"""
    trainer = ModelTrainer()
    
    # Prepare sequential data for LSTM
    from ml_models.lstm_model import LSTMModel
    lstm_model = LSTMModel(
        sequence_length=20,
        n_features=50,  # Will be adjusted based on actual features
        lstm_units=[64, 32],
        dropout_rate=0.3
    )
    
    # Prepare data with sequences
    all_sequences = []
    all_targets = []
    
    for symbol in (symbols or SACRED_SYMBOLS[:5]):
        try:
            features, targets = trainer.feature_engineer.build_feature_matrix(symbol)
            if features.empty:
                continue
            
            # Create sequences
            sequences, seq_targets = lstm_model.create_sequences(features.values, targets.values)
            all_sequences.append(sequences)
            all_targets.append(seq_targets)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    if all_sequences:
        # Combine all sequences
        X = np.vstack(all_sequences)
        y = np.hstack(all_targets)
        
        # Update model with correct feature count
        lstm_model.n_features = X.shape[2]
        lstm_model.build_model()
        
        # Train
        split_idx = int(len(X) * 0.8)
        history = lstm_model.train(
            X[:split_idx], y[:split_idx],
            X[split_idx:], y[split_idx:],
            epochs=30,
            batch_size=32
        )
        
        # Save
        model_path = trainer.model_dir / 'lstm_model.h5'
        lstm_model.model.save(model_path)
        
        return lstm_model.model
    
    return None


def main():
    """Run model training"""
    trainer = ModelTrainer()
    
    # Run full training pipeline
    results = trainer.run_full_training_pipeline(
        train_start="2022-07-07",  # Start from when all indicators are available
        train_end="2024-06-30",    # Recent data
        test_split=0.2
    )
    
    return results


if __name__ == "__main__":
    main()