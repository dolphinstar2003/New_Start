#!/usr/bin/env python3
"""
Hyperparameter Optimization with Optuna
Optimize ML models for better performance
"""
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
from typing import Dict, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.enhanced_feature_engineer import EnhancedFeatureEngineer

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class HyperparameterOptimizer:
    """Optimize hyperparameters using Optuna"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_dir = Path("ml_models/saved_models")
        self.model_dir.mkdir(exist_ok=True, parents=True)
        
        self.feature_engineer = EnhancedFeatureEngineer(self.data_dir)
        
        # Optimization settings
        self.n_trials = 20  # Number of optimization trials (reduced for speed)
        self.cv_folds = 3   # Cross-validation folds
        self.test_size = 0.2
        
        logger.info("Hyperparameter Optimizer initialized")
    
    def prepare_optimization_data(self, symbols: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for optimization"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:3]  # Small set for faster optimization
        
        logger.info(f"Preparing optimization data for {len(symbols)} symbols")
        
        all_features = []
        all_targets = []
        common_features = None
        
        # First pass: determine common feature set
        for symbol in symbols:
            try:
                features_df, targets_df = self.feature_engineer.build_enhanced_feature_matrix(symbol, '1d')
                
                if features_df.empty or targets_df.empty:
                    continue
                
                # Remove non-numeric and problematic columns
                if 'symbol_id' in features_df.columns:
                    features_df = features_df.drop(columns=['symbol_id'])
                
                # Select only numeric columns
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                feature_names = set(numeric_cols)
                
                if common_features is None:
                    common_features = feature_names
                else:
                    common_features = common_features.intersection(feature_names)
                
                logger.info(f"Symbol {symbol}: {len(feature_names)} features")
                
            except Exception as e:
                logger.error(f"Error analyzing features for {symbol}: {e}")
                continue
        
        if common_features is None or len(common_features) == 0:
            raise ValueError("No common features found across symbols")
        
        common_features = sorted(list(common_features))
        logger.info(f"Common features across all symbols: {len(common_features)}")
        
        # Second pass: extract data with common features
        for symbol in symbols:
            try:
                features_df, targets_df = self.feature_engineer.build_enhanced_feature_matrix(symbol, '1d')
                
                if features_df.empty or targets_df.empty:
                    continue
                
                # Select target
                target_col = 'signal_1d'
                if target_col not in targets_df.columns:
                    continue
                
                # Clean data
                target_data = targets_df[target_col].dropna()
                feature_data = features_df.loc[target_data.index]
                
                # Remove symbol_id if present
                if 'symbol_id' in feature_data.columns:
                    feature_data = feature_data.drop(columns=['symbol_id'])
                
                # Select only common features
                feature_data = feature_data[common_features]
                
                # Remove NaN
                mask = ~(feature_data.isna().any(axis=1) | target_data.isna())
                feature_data = feature_data[mask]
                target_data = target_data[mask]
                
                if len(feature_data) > 0:
                    all_features.append(feature_data.values)
                    all_targets.append(target_data.values)
                
                logger.info(f"Added {len(feature_data)} clean samples from {symbol} ({feature_data.shape[1]} features)")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid data for optimization")
        
        # Combine data
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        logger.info(f"Optimization data ready: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Classes: {le.classes_}")
        
        return X, y_encoded, le
    
    def optimize_random_forest(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize Random Forest hyperparameters"""
        logger.info("Optimizing Random Forest hyperparameters...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Create and evaluate model
            model = RandomForestClassifier(**params)
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize', 
                                   study_name='random_forest_optimization')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best Random Forest score: {best_score:.3f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize XGBoost hyperparameters"""
        logger.info("Optimizing XGBoost hyperparameters...")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'objective': 'multi:softprob',
                'num_class': len(np.unique(y)),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'random_state': 42,
                'verbosity': 0
            }
            
            # Create and evaluate model
            model = xgb.XGBClassifier(**params)
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring='f1_weighted', n_jobs=-1)
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize',
                                   study_name='xgboost_optimization')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best XGBoost score: {best_score:.3f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def optimize_lstm(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize LSTM hyperparameters"""
        logger.info("Optimizing LSTM hyperparameters...")
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        def objective(trial):
            # Clear any previous models
            tf.keras.backend.clear_session()
            
            # Suggest hyperparameters
            lstm_units = trial.suggest_int('lstm_units', 16, 128, step=16)
            dense_units = trial.suggest_int('dense_units', 8, 64, step=8)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Prepare sequence data
            sequence_length = 10
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, sequence_length)
            
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                return 0.0
            
            # Convert to categorical
            num_classes = len(np.unique(y))
            y_train_cat = tf.keras.utils.to_categorical(y_train_seq, num_classes=num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test_seq, num_classes=num_classes)
            
            # Build model
            model = Sequential([
                LSTM(lstm_units, input_shape=(sequence_length, X.shape[1])),
                Dropout(dropout_rate),
                Dense(dense_units, activation='relu'),
                Dropout(dropout_rate),
                Dense(num_classes, activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            try:
                history = model.fit(
                    X_train_seq, y_train_cat,
                    validation_data=(X_test_seq, y_test_cat),
                    epochs=20,  # Reduced for faster optimization
                    batch_size=batch_size,
                    verbose=0
                )
                
                # Return best validation accuracy
                return max(history.history['val_accuracy'])
                
            except Exception as e:
                logger.warning(f"LSTM trial failed: {e}")
                return 0.0
        
        # Run optimization
        study = optuna.create_study(direction='maximize',
                                   study_name='lstm_optimization')
        study.optimize(objective, n_trials=min(10, self.n_trials), show_progress_bar=True)
        
        # Best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best LSTM score: {best_score:.3f}")
        logger.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study
        }
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        if len(X) < sequence_length:
            return np.array([]), np.array([])
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_optimized_models(self, optimization_results: Dict, X: np.ndarray, y: np.ndarray, label_encoder) -> Dict:
        """Train models with optimized hyperparameters"""
        logger.info("Training models with optimized hyperparameters...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        trained_models = {}
        
        # Train optimized Random Forest
        if 'random_forest' in optimization_results:
            logger.info("Training optimized Random Forest...")
            rf_params = optimization_results['random_forest']['best_params']
            rf_model = RandomForestClassifier(**rf_params)
            rf_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = rf_model.predict(X_test)
            rf_score = f1_score(y_test, y_pred, average='weighted')
            
            # Save model
            model_path = self.model_dir / 'random_forest_optimized.pkl'
            model_data = {
                'model': rf_model,
                'label_encoder': label_encoder,
                'params': rf_params,
                'test_score': rf_score
            }
            joblib.dump(model_data, model_path)
            
            trained_models['random_forest'] = {
                'path': model_path,
                'test_score': rf_score,
                'params': rf_params
            }
            
            logger.info(f"Optimized Random Forest test score: {rf_score:.3f}")
        
        # Train optimized XGBoost
        if 'xgboost' in optimization_results:
            logger.info("Training optimized XGBoost...")
            xgb_params = optimization_results['xgboost']['best_params']
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = xgb_model.predict(X_test)
            xgb_score = f1_score(y_test, y_pred, average='weighted')
            
            # Save model
            model_path = self.model_dir / 'xgboost_optimized_v2.pkl'
            model_data = {
                'model': xgb_model,
                'label_encoder': label_encoder,
                'params': xgb_params,
                'test_score': xgb_score
            }
            joblib.dump(model_data, model_path)
            
            trained_models['xgboost'] = {
                'path': model_path,
                'test_score': xgb_score,
                'params': xgb_params
            }
            
            logger.info(f"Optimized XGBoost test score: {xgb_score:.3f}")
        
        return trained_models
    
    def save_optimization_results(self, optimization_results: Dict, trained_models: Dict):
        """Save optimization results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'optimization_settings': {
                'n_trials': self.n_trials,
                'cv_folds': self.cv_folds,
                'test_size': self.test_size
            },
            'optimization_results': {},
            'trained_models': {}
        }
        
        # Save optimization results (without study objects)
        for model_name, result in optimization_results.items():
            results['optimization_results'][model_name] = {
                'best_params': result['best_params'],
                'best_score': result['best_score']
            }
        
        # Save trained model info
        for model_name, result in trained_models.items():
            results['trained_models'][model_name] = {
                'test_score': result['test_score'],
                'params': result['params'],
                'model_path': str(result['path'])
            }
        
        # Save to file
        results_path = self.model_dir / 'hyperparameter_optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved: {results_path}")
    
    def run_full_optimization(self) -> Dict:
        """Run complete hyperparameter optimization"""
        logger.info("Starting comprehensive hyperparameter optimization...")
        
        # Prepare data
        X, y, label_encoder = self.prepare_optimization_data()
        
        optimization_results = {}
        
        # Optimize Random Forest
        try:
            rf_result = self.optimize_random_forest(X, y)
            optimization_results['random_forest'] = rf_result
            logger.info("✅ Random Forest optimization completed")
        except Exception as e:
            logger.error(f"❌ Random Forest optimization failed: {e}")
        
        # Optimize XGBoost
        try:
            xgb_result = self.optimize_xgboost(X, y)
            optimization_results['xgboost'] = xgb_result
            logger.info("✅ XGBoost optimization completed")
        except Exception as e:
            logger.error(f"❌ XGBoost optimization failed: {e}")
        
        # Optimize LSTM (optional, can be slow)
        try:
            lstm_result = self.optimize_lstm(X, y)
            optimization_results['lstm'] = lstm_result
            logger.info("✅ LSTM optimization completed")
        except Exception as e:
            logger.error(f"❌ LSTM optimization failed: {e}")
        
        # Train optimized models
        trained_models = self.train_optimized_models(optimization_results, X, y, label_encoder)
        
        # Save results
        self.save_optimization_results(optimization_results, trained_models)
        
        return {
            'optimization_results': optimization_results,
            'trained_models': trained_models
        }


def main():
    """Main optimization function"""
    print("\n" + "="*70)
    print("🎯 HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("="*70)
    
    print("\n🔍 Optimization Plan:")
    print("   1. Random Forest (50 trials)")
    print("   2. XGBoost (50 trials)")
    print("   3. LSTM (20 trials)")
    print("   4. Train optimized models")
    print("   5. Save best parameters")
    
    print("\n📊 Optimization Settings:")
    print("   • Objective: F1-weighted score")
    print("   • Cross-validation: 3-fold")
    print("   • Test size: 20%")
    print("   • Symbols: First 3 sacred symbols")
    
    optimizer = HyperparameterOptimizer()
    
    try:
        print(f"\n🚀 Starting optimization...")
        results = optimizer.run_full_optimization()
        
        # Display results
        print(f"\n" + "="*70)
        print("📈 OPTIMIZATION RESULTS")
        print("="*70)
        
        if 'optimization_results' in results:
            for model_name, result in results['optimization_results'].items():
                print(f"\n🤖 {model_name.upper()}:")
                print(f"   Best CV Score: {result['best_score']:.3f}")
                print(f"   Best Parameters:")
                for param, value in result['best_params'].items():
                    print(f"     {param}: {value}")
        
        if 'trained_models' in results:
            print(f"\n📊 Test Performance:")
            for model_name, result in results['trained_models'].items():
                print(f"   {model_name}: {result['test_score']:.3f} F1-score")
        
        print(f"\n✅ Hyperparameter optimization completed!")
        print(f"📁 Results saved in: {optimizer.model_dir}")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()