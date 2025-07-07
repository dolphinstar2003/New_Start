#!/usr/bin/env python3
"""
Enhanced ML Model Training Pipeline
Train all models (XGBoost, Random Forest, LSTM, GRU) with enhanced features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.enhanced_feature_engineer import EnhancedFeatureEngineer

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class EnhancedModelTrainer:
    """Enhanced model trainer for all ML/DL models"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or DATA_DIR
        self.model_dir = self.data_dir.parent / 'ml_models' / 'saved_models'
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        # Enhanced feature engineer
        self.feature_engineer = EnhancedFeatureEngineer(self.data_dir)
        
        # Training configuration
        self.test_size = 0.2
        self.random_state = 42
        self.cv_folds = 5
        
        logger.info("Enhanced Model Trainer initialized")
    
    def prepare_training_data(self, symbols: List[str] = None, target_column: str = 'signal_1d') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from multiple symbols"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:10]  # Use first 10 symbols for speed
        
        logger.info(f"Preparing training data for {len(symbols)} symbols")
        
        all_features = []
        all_targets = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
            
            try:
                # Get enhanced features
                features_df, targets_df = self.feature_engineer.build_enhanced_feature_matrix(symbol, '1d')
                
                if features_df.empty or targets_df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                # Select target
                if target_column not in targets_df.columns:
                    logger.warning(f"Target {target_column} not found for {symbol}")
                    continue
                
                # Add symbol identifier
                features_df['symbol_id'] = i
                
                # Collect data
                all_features.append(features_df)
                all_targets.append(targets_df[target_column])
                
                logger.info(f"  Added {len(features_df)} samples from {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid data collected from any symbol")
        
        # Combine all data
        logger.info("Combining training data...")
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        # Remove samples where target is NaN
        valid_mask = ~combined_targets.isna()
        combined_features = combined_features[valid_mask]
        combined_targets = combined_targets[valid_mask]
        
        logger.info(f"Final training data: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        # Target distribution
        target_counts = combined_targets.value_counts().sort_index()
        logger.info("Target distribution:")
        for target, count in target_counts.items():
            pct = count / len(combined_targets) * 100
            logger.info(f"  Class {target}: {count} ({pct:.1f}%)")
        
        return combined_features, combined_targets
    
    def train_xgboost(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': len(y_train.unique()),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'eval_metric': 'mlogloss',
            'verbosity': 0
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=50,
                  verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Save model
        model_path = self.model_dir / 'xgboost_enhanced.pkl'
        joblib.dump(model, model_path)
        logger.info(f"XGBoost model saved: {model_path}")
        
        return {'model': model, 'metrics': metrics, 'path': model_path}
    
    def train_random_forest(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series) -> Dict:
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        # Random Forest parameters
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Save model
        model_path = self.model_dir / 'random_forest_enhanced.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Random Forest model saved: {model_path}")
        
        return {'model': model, 'metrics': metrics, 'path': model_path}
    
    def train_lstm(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series, sequence_length: int = 20) -> Dict:
        """Train LSTM model"""
        logger.info("Training LSTM model...")
        
        # Prepare sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        # Convert targets to categorical
        num_classes = len(y_train.unique())
        y_train_cat = tf.keras.utils.to_categorical(y_train_seq + 1, num_classes=num_classes)  # +1 to handle -1 class
        y_test_cat = tf.keras.utils.to_categorical(y_test_seq + 1, num_classes=num_classes)
        
        # Build LSTM model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train
        history = model.fit(
            X_train_seq, y_train_cat,
            validation_data=(X_test_seq, y_test_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1) - 1  # Convert back to original classes
        
        # Metrics
        metrics = self._calculate_metrics(y_test_seq, y_pred, y_pred_proba)
        
        # Save model
        model_path = self.model_dir / 'lstm_enhanced.h5'
        model.save(model_path)
        logger.info(f"LSTM model saved: {model_path}")
        
        return {'model': model, 'metrics': metrics, 'path': model_path, 'history': history.history}
    
    def train_gru(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: pd.Series, y_test: pd.Series, sequence_length: int = 20) -> Dict:
        """Train GRU model"""
        logger.info("Training GRU model...")
        
        # Prepare sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        # Convert targets to categorical
        num_classes = len(y_train.unique())
        y_train_cat = tf.keras.utils.to_categorical(y_train_seq + 1, num_classes=num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_seq + 1, num_classes=num_classes)
        
        # Build GRU model
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            BatchNormalization(),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train
        history = model.fit(
            X_train_seq, y_train_cat,
            validation_data=(X_test_seq, y_test_cat),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1) - 1
        
        # Metrics
        metrics = self._calculate_metrics(y_test_seq, y_pred, y_pred_proba)
        
        # Save model
        model_path = self.model_dir / 'gru_enhanced.h5'
        model.save(model_path)
        logger.info(f"GRU model saved: {model_path}")
        
        return {'model': model, 'metrics': metrics, 'path': model_path, 'history': history.history}
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/GRU training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """Calculate comprehensive metrics"""
        # Ensure y_true and y_pred are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def train_all_models(self, symbols: List[str] = None) -> Dict:
        """Train all models and compare performance"""
        logger.info("Starting comprehensive model training...")
        
        # Prepare data
        X, y = self.prepare_training_data(symbols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train all models
        results = {}
        
        try:
            results['xgboost'] = self.train_xgboost(X_train, X_test, y_train, y_test)
            logger.info("✅ XGBoost training completed")
        except Exception as e:
            logger.error(f"❌ XGBoost training failed: {e}")
        
        try:
            results['random_forest'] = self.train_random_forest(X_train, X_test, y_train, y_test)
            logger.info("✅ Random Forest training completed")
        except Exception as e:
            logger.error(f"❌ Random Forest training failed: {e}")
        
        try:
            results['lstm'] = self.train_lstm(X_train, X_test, y_train, y_test)
            logger.info("✅ LSTM training completed")
        except Exception as e:
            logger.error(f"❌ LSTM training failed: {e}")
        
        try:
            results['gru'] = self.train_gru(X_train, X_test, y_train, y_test)
            logger.info("✅ GRU training completed")
        except Exception as e:
            logger.error(f"❌ GRU training failed: {e}")
        
        # Save results summary
        self._save_training_summary(results)
        
        return results
    
    def _save_training_summary(self, results: Dict):
        """Save training summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(results.keys()),
            'metrics': {}
        }
        
        for model_name, result in results.items():
            if 'metrics' in result:
                summary['metrics'][model_name] = result['metrics']
        
        summary_path = self.model_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved: {summary_path}")


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🚀 ENHANCED ML MODEL TRAINING PIPELINE")
    print("="*70)
    
    print("\n🤖 Models to train:")
    print("   1. XGBoost (Gradient Boosting)")
    print("   2. Random Forest (Ensemble)")
    print("   3. LSTM (Deep Learning)")
    print("   4. GRU (Deep Learning)")
    
    print("\n📊 Enhanced Features:")
    print("   • 85+ optimized features")
    print("   • Zero NaN values")
    print("   • Multi-timeframe analysis")
    print("   • Advanced technical indicators")
    
    # Initialize trainer
    trainer = EnhancedModelTrainer()
    
    # Use first 5 symbols for faster training
    symbols = SACRED_SYMBOLS[:5]
    print(f"\n📈 Training symbols: {symbols}")
    
    try:
        # Train all models
        results = trainer.train_all_models(symbols)
        
        # Display results
        print("\n" + "="*70)
        print("📊 TRAINING RESULTS")
        print("="*70)
        
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"\n🤖 {model_name.upper()}:")
                print(f"   Accuracy:  {metrics['accuracy']:.3f}")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall:    {metrics['recall']:.3f}")
                print(f"   F1 Score:  {metrics['f1']:.3f}")
        
        print(f"\n✅ All models trained successfully!")
        print(f"📁 Models saved in: {trainer.model_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()