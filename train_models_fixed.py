#!/usr/bin/env python3
"""
Fixed ML Model Training
Address the issues found in previous training
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class FixedModelTrainer:
    """Fixed model trainer addressing all issues"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_dir = self.data_dir.parent / 'ml_models' / 'saved_models'
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.feature_engineer = EnhancedFeatureEngineer(self.data_dir)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        logger.info("Fixed Model Trainer initialized")
    
    def prepare_clean_data(self, symbols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare clean numerical data for training"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:3]  # Smaller set for testing
        
        logger.info(f"Preparing clean data for {len(symbols)} symbols")
        
        all_features = []
        all_targets = []
        
        for symbol in symbols:
            try:
                features_df, targets_df = self.feature_engineer.build_enhanced_feature_matrix(symbol, '1d')
                
                if features_df.empty or targets_df.empty:
                    continue
                
                # Select target column
                target_col = 'signal_1d'
                if target_col not in targets_df.columns:
                    continue
                
                # Clean data
                target_data = targets_df[target_col].dropna()
                feature_data = features_df.loc[target_data.index]
                
                # Ensure numerical features only
                numeric_cols = feature_data.select_dtypes(include=[np.number]).columns
                feature_data = feature_data[numeric_cols]
                
                # Remove any remaining NaN
                mask = feature_data.isna().any(axis=1) | target_data.isna()
                feature_data = feature_data[~mask]
                target_data = target_data[~mask]
                
                all_features.append(feature_data.values)
                all_targets.append(target_data.values)
                
                logger.info(f"Added {len(feature_data)} clean samples from {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid data collected")
        
        # Combine data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        
        # Encode labels to 0, 1, 2, 3
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Final clean data: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        logger.info(f"Label distribution: {np.bincount(y_encoded)}")
        
        return X_scaled, y_encoded
    
    def train_fixed_xgboost(self, X_train: np.ndarray, X_test: np.ndarray, 
                           y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train XGBoost with fixed parameters"""
        logger.info("Training XGBoost (fixed)...")
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'random_state': 42,
            'eval_metric': 'mlogloss',
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds=30,
                  verbose=False)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Save model
        model_path = self.model_dir / 'xgboost_fixed.pkl'
        model_data = {
            'model': model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }
        joblib.dump(model_data, model_path)
        
        logger.info(f"Fixed XGBoost saved: {model_path}")
        
        return {'metrics': metrics, 'path': model_path}
    
    def train_fixed_lstm(self, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train LSTM with fixed data types"""
        logger.info("Training LSTM (fixed)...")
        
        # Prepare sequences
        sequence_length = 10  # Shorter sequence for small dataset
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
        
        # Convert to categorical
        num_classes = len(np.unique(y_train))
        y_train_cat = tf.keras.utils.to_categorical(y_train_seq, num_classes=num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_seq, num_classes=num_classes)
        
        # Build simpler LSTM
        model = Sequential([
            LSTM(32, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train_seq, y_train_cat,
            validation_data=(X_test_seq, y_test_cat),
            epochs=50,
            batch_size=16,
            callbacks=[EarlyStopping(patience=10)],
            verbose=0
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test_seq, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test_seq, y_pred),
            'precision': precision_score(y_test_seq, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test_seq, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test_seq, y_pred, average='weighted', zero_division=0)
        }
        
        # Save model
        model_path = self.model_dir / 'lstm_fixed.h5'
        model.save(model_path)
        
        # Save preprocessing objects
        preprocessing_path = self.model_dir / 'lstm_preprocessing.pkl'
        preprocessing_data = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'sequence_length': sequence_length
        }
        joblib.dump(preprocessing_data, preprocessing_path)
        
        logger.info(f"Fixed LSTM saved: {model_path}")
        
        return {'metrics': metrics, 'path': model_path}
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_all_fixed(self) -> Dict:
        """Train all models with fixes"""
        logger.info("Starting fixed model training...")
        
        # Prepare clean data
        X, y = self.prepare_clean_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training: {len(X_train)}, Test: {len(X_test)}")
        
        results = {}
        
        # Train XGBoost
        try:
            results['xgboost'] = self.train_fixed_xgboost(X_train, X_test, y_train, y_test)
            logger.info("✅ XGBoost fixed training completed")
        except Exception as e:
            logger.error(f"❌ XGBoost failed: {e}")
        
        # Train LSTM
        try:
            results['lstm'] = self.train_fixed_lstm(X_train, X_test, y_train, y_test)
            logger.info("✅ LSTM fixed training completed")
        except Exception as e:
            logger.error(f"❌ LSTM failed: {e}")
        
        return results


def main():
    """Test fixed training"""
    print("\n" + "="*60)
    print("🔧 FIXED ML MODEL TRAINING")
    print("="*60)
    
    trainer = FixedModelTrainer()
    
    try:
        results = trainer.train_all_fixed()
        
        print("\n📊 Training Results:")
        for model_name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                print(f"\n🤖 {model_name.upper()}:")
                print(f"   Accuracy:  {metrics['accuracy']:.3f}")
                print(f"   Precision: {metrics['precision']:.3f}")
                print(f"   Recall:    {metrics['recall']:.3f}")
                print(f"   F1 Score:  {metrics['f1']:.3f}")
        
        print(f"\n✅ Fixed training completed!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()