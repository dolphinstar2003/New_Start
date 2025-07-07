"""
Train ML/DL Models for Different Backtest Strategies
Each model is trained to mimic specific backtest strategy signals
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.feature_engineer import FeatureEngineer
from backtest.realistic_backtest import IndicatorBacktest
from backtest.backtest_sirali import HierarchicalBacktest
from backtest.backtest_oncelikli import PriorityBacktest


class BacktestStrategyTrainer:
    """Train ML models to learn from different backtest strategies"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.model_dir = DATA_DIR.parent / 'ml_models' / 'saved_models'
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.feature_engineer = FeatureEngineer(self.data_dir)
        
        logger.info("BacktestStrategyTrainer initialized")
    
    def collect_strategy_signals(self, strategy_name: str, symbols: list = None, days: int = 365) -> pd.DataFrame:
        """Collect signals from a specific backtest strategy"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:10]
        
        all_signals = []
        
        logger.info(f"Collecting signals from {strategy_name} strategy...")
        
        for symbol in symbols:
            try:
                if strategy_name == 'realistic':
                    # Use indicator-based backtest
                    backtest = IndicatorBacktest()
                    signals = backtest.generate_signals(symbol)
                    
                elif strategy_name == 'hierarchical_1d':
                    # Use 1D hierarchical backtest
                    backtest = HierarchicalBacktest('1d')
                    signals = backtest.generate_signals_sequential(symbol, '1d')
                    
                elif strategy_name == 'hierarchical_4h':
                    # Use 4H hierarchical backtest
                    backtest = HierarchicalBacktest('4h')
                    signals = backtest.generate_signals_sequential(symbol, '4h')
                    
                elif strategy_name == 'priority':
                    # Use priority-based backtest
                    backtest = PriorityBacktest()
                    signals = backtest.generate_priority_signals(symbol)
                    
                else:
                    logger.warning(f"Unknown strategy: {strategy_name}")
                    continue
                
                if not signals.empty and isinstance(signals, pd.DataFrame):
                    # Create a copy to avoid modifying original
                    signals_copy = signals.copy()
                    signals_copy['symbol'] = symbol
                    all_signals.append(signals_copy)
                    logger.info(f"  {symbol}: {len(signals)} signals collected")
                
            except Exception as e:
                logger.error(f"Error collecting signals for {symbol}: {e}")
                continue
        
        if all_signals:
            combined_signals = pd.concat(all_signals, ignore_index=True)
            logger.info(f"Total signals collected: {len(combined_signals)}")
            return combined_signals
        else:
            return pd.DataFrame()
    
    def prepare_training_data(self, signals_df: pd.DataFrame, symbols: list) -> tuple:
        """Prepare features and labels from signals"""
        all_features = []
        all_labels = []
        
        logger.info("Preparing training data from signals...")
        
        for symbol in symbols:
            # Get signals for this symbol
            symbol_signals = signals_df[signals_df['symbol'] == symbol]
            
            if symbol_signals.empty:
                continue
            
            # Build features
            features, _ = self.feature_engineer.build_feature_matrix(symbol)
            
            if features.empty:
                continue
            
            # Align with signals
            common_dates = features.index.intersection(symbol_signals.index)
            
            if len(common_dates) == 0:
                continue
            
            features_aligned = features.loc[common_dates]
            signals_aligned = symbol_signals.loc[common_dates]
            
            # Create labels from signals
            # Convert signals: -1 (sell), 0 (hold), 1 (buy) to 0, 1, 2 for classification
            labels = signals_aligned['signal'] + 1
            
            all_features.append(features_aligned)
            all_labels.append(labels)
        
        if all_features:
            X = pd.concat(all_features)
            y = pd.concat(all_labels)
            
            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Training data prepared: {len(X)} samples, {X.shape[1]} features")
            
            # Log class distribution
            logger.info("Label distribution:")
            for label, count in y.value_counts().items():
                logger.info(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")
            
            return X, y
        else:
            return pd.DataFrame(), pd.Series()
    
    def train_xgboost_for_strategy(self, strategy_name: str, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Train XGBoost model for specific strategy"""
        logger.info(f"Training XGBoost for {strategy_name} strategy...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Model parameters optimized for each strategy
        if strategy_name == 'realistic':
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8
            }
        elif 'hierarchical' in strategy_name:
            params = {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.03,
                'subsample': 0.7
            }
        else:  # priority
            params = {
                'n_estimators': 250,
                'max_depth': 7,
                'learning_rate': 0.04,
                'subsample': 0.75
            }
        
        # Create and train model
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            **params
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = self.model_dir / f'xgboost_{strategy_name}.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
    
    def train_lstm_for_strategy(self, strategy_name: str, X: pd.DataFrame, y: pd.Series) -> keras.Model:
        """Train LSTM model for specific strategy"""
        logger.info(f"Training LSTM for {strategy_name} strategy...")
        
        # Prepare sequences
        sequence_length = 20
        n_features = X.shape[1]
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Convert labels to categorical
        y_seq = keras.utils.to_categorical(y_seq, num_classes=3)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3)
            ]
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        model_path = self.model_dir / f'lstm_{strategy_name}.h5'
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model
    
    def train_all_strategies(self, strategies: list = None):
        """Train models for all strategies"""
        if strategies is None:
            strategies = ['realistic', 'hierarchical_1d', 'hierarchical_4h', 'priority']
        
        logger.info("="*80)
        logger.info("TRAINING ML MODELS FOR BACKTEST STRATEGIES")
        logger.info("="*80)
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {strategy} strategy")
            logger.info(f"{'='*60}")
            
            try:
                # Collect signals
                signals = self.collect_strategy_signals(strategy, SACRED_SYMBOLS[:10])
                
                if signals.empty:
                    logger.warning(f"No signals collected for {strategy}")
                    continue
                
                # Prepare training data
                X, y = self.prepare_training_data(signals, SACRED_SYMBOLS[:10])
                
                if X.empty:
                    logger.warning(f"No training data for {strategy}")
                    continue
                
                # Train XGBoost
                xgb_model = self.train_xgboost_for_strategy(strategy, X, y)
                
                # Train LSTM (only for 1D strategies due to data requirements)
                if '4h' not in strategy:
                    lstm_model = self.train_lstm_for_strategy(strategy, X, y)
                    results[strategy] = {'xgboost': xgb_model, 'lstm': lstm_model}
                else:
                    results[strategy] = {'xgboost': xgb_model}
                
                logger.info(f"✅ Successfully trained models for {strategy}")
                
            except Exception as e:
                logger.error(f"Error training {strategy}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED")
        logger.info("="*80)
        logger.info(f"Successfully trained models for {len(results)} strategies")
        
        return results


def main():
    """Train all models"""
    trainer = BacktestStrategyTrainer()
    
    # Train for specific strategies that work well
    strategies = ['realistic', 'hierarchical_1d', 'priority']
    
    results = trainer.train_all_strategies(strategies)
    
    # Also train a general XGBoost model for ml_xgboost backtest
    logger.info("\nTraining general XGBoost model...")
    from ml_models.train_models import train_xgboost_model
    train_xgboost_model(SACRED_SYMBOLS[:10])
    
    return results


if __name__ == "__main__":
    main()