"""
Train Optimized XGBoost Model for Trading
Uses actual trading signals from successful strategies
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR


def prepare_simple_features(symbol: str, timeframe: str = '1d') -> pd.DataFrame:
    """Prepare simple features from indicators"""
    features = pd.DataFrame()
    
    # Load indicators
    indicators_dir = DATA_DIR / 'indicators' / timeframe
    
    indicator_files = {
        'supertrend': ['trend', 'supertrend'],
        'adx_di': ['adx', 'plus_di', 'minus_di'],
        'squeeze_momentum': ['squeeze_on', 'momentum'],
        'wavetrend': ['wt1', 'wt2'],
        'macd_custom': ['macd', 'signal', 'histogram'],
        'vixfix': ['vixfix', 'position_factor']
    }
    
    for indicator, cols in indicator_files.items():
        filepath = indicators_dir / f"{symbol}_{timeframe}_{indicator}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            for col in cols:
                if col in df.columns:
                    features[f"{indicator}_{col}"] = df[col]
    
    # Load price data for additional features
    price_path = DATA_DIR / 'raw' / timeframe / f"{symbol}_{timeframe}_raw.csv"
    if price_path.exists():
        price_df = pd.read_csv(price_path)
        price_df['datetime'] = pd.to_datetime(price_df['datetime'])
        price_df.set_index('datetime', inplace=True)
        
        # Add price-based features
        features['returns'] = price_df['close'].pct_change()
        features['volume_ratio'] = price_df['volume'] / price_df['volume'].rolling(20).mean()
        features['price_position'] = (price_df['close'] - price_df['low']) / (price_df['high'] - price_df['low'])
        features['volatility'] = features['returns'].rolling(20).std()
        features['close'] = price_df['close']
    
    return features


def create_trading_labels(features_df: pd.DataFrame, lookahead: int = 3) -> pd.Series:
    """Create labels based on future returns"""
    if 'close' not in features_df.columns:
        return pd.Series()
    
    # Calculate future returns
    future_returns = features_df['close'].pct_change(lookahead).shift(-lookahead)
    
    # Create labels: 0=sell, 1=hold, 2=buy
    labels = pd.Series(1, index=features_df.index)  # Default to hold
    
    # Buy signal: future return > 2%
    labels[future_returns > 0.02] = 2
    
    # Sell signal: future return < -2%
    labels[future_returns < -0.02] = 0
    
    return labels


def train_optimized_model():
    """Train optimized XGBoost model"""
    logger.info("="*80)
    logger.info("TRAINING OPTIMIZED XGBOOST MODEL")
    logger.info("="*80)
    
    all_features = []
    all_labels = []
    
    # Collect data from multiple symbols
    symbols_to_use = SACRED_SYMBOLS[:10]
    
    for symbol in symbols_to_use:
        logger.info(f"Processing {symbol}...")
        
        # Get features
        features = prepare_simple_features(symbol)
        
        if features.empty:
            continue
        
        # Create labels
        labels = create_trading_labels(features)
        
        if labels.empty:
            continue
        
        # Remove NaN values
        valid_idx = features.notna().all(axis=1) & labels.notna()
        features_clean = features[valid_idx]
        labels_clean = labels[valid_idx]
        
        # Remove close price from features
        if 'close' in features_clean.columns:
            features_clean = features_clean.drop('close', axis=1)
        
        all_features.append(features_clean)
        all_labels.append(labels_clean)
        
        logger.info(f"  Added {len(features_clean)} samples")
    
    # Combine all data
    X = pd.concat(all_features)
    y = pd.concat(all_labels)
    
    logger.info(f"\nTotal samples: {len(X)}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info("\nLabel distribution:")
    for label, count in y.value_counts().items():
        logger.info(f"  Class {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost with optimized parameters
    logger.info("\nTraining XGBoost model...")
    
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        reg_alpha=0.1,
        reg_lambda=1.0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"\nTest Accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 10 Important Features:")
    for _, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_dir = Path(__file__).parent / 'saved_models'
    model_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = model_dir / 'xgboost_optimized.pkl'
    joblib.dump(model, model_path)
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # Also save as the default model for backtest
    default_path = model_dir / 'xgboost_model.pkl'
    joblib.dump(model, default_path)
    logger.info(f"✅ Also saved as default model: {default_path}")
    
    return model


if __name__ == "__main__":
    train_optimized_model()