"""
Complete Multi-Timeframe Pipeline
1. Fetch MTF data
2. Calculate indicators for all timeframes  
3. Train ML models with MTF features
4. Run backtest
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from loguru import logger
from core.mtf_data_fetcher import MultiTimeframeDataFetcher
from indicators.calculator import IndicatorCalculator
from config.settings import SACRED_SYMBOLS, DATA_DIR
import time
import pandas as pd
import numpy as np

logger.info("="*80)
logger.info("STARTING COMPLETE MTF PIPELINE")
logger.info("="*80)

# Step 1: Fetch MTF Data
logger.info("\n" + "="*60)
logger.info("STEP 1: FETCHING MULTI-TIMEFRAME DATA")
logger.info("="*60)

mtf_fetcher = MultiTimeframeDataFetcher()
mtf_fetcher.fetch_all_symbols_mtf()

# Step 2: Calculate indicators for each timeframe
logger.info("\n" + "="*60)
logger.info("STEP 2: CALCULATING INDICATORS FOR ALL TIMEFRAMES")
logger.info("="*60)

for timeframe in ['1h', '4h', '1d', '1wk']:
    logger.info(f"\n--- Processing {timeframe} timeframe ---")
    
    # Create indicator directory for timeframe
    tf_indicator_dir = DATA_DIR / 'indicators' / timeframe
    tf_indicator_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize calculator for this timeframe
    calc = IndicatorCalculator(DATA_DIR)
    
    # Process each symbol
    for symbol in SACRED_SYMBOLS:
        try:
            # Check if raw data exists
            raw_file = DATA_DIR / 'raw' / timeframe / f"{symbol}_{timeframe}_raw.csv"
            if not raw_file.exists():
                logger.warning(f"No {timeframe} data for {symbol}")
                continue
                
            # Load data
            df = pd.read_csv(raw_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Calculate each indicator
            for indicator_name, indicator_func in calc.core_indicators.items():
                try:
                    result = indicator_func(df.copy())
                    result.reset_index(inplace=True)
                    
                    # Save to timeframe-specific directory
                    output_file = tf_indicator_dir / f"{symbol}_{timeframe}_{indicator_name}.csv"
                    result.to_csv(output_file, index=False)
                    
                except Exception as e:
                    logger.error(f"Error calculating {indicator_name} for {symbol} {timeframe}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing {symbol} {timeframe}: {e}")
    
    logger.info(f"Completed {timeframe} indicators")
    time.sleep(1)

logger.info("\n✅ MTF indicator calculation completed!")

# Step 3: Train ML models with MTF features
logger.info("\n" + "="*60)
logger.info("STEP 3: TRAINING ML MODELS WITH MTF FEATURES")
logger.info("="*60)

# Import after indicator calculation
from ml_models.feature_engineer import FeatureEngineer
from ml_models.ensemble_model import EnsembleTradingModel

# Create MTF feature engineer
class MTFFeatureEngineer(FeatureEngineer):
    """Feature engineer with multi-timeframe support"""
    
    def build_mtf_features(self, symbol: str) -> pd.DataFrame:
        """Build features from multiple timeframes"""
        all_features = []
        
        # Load features for each timeframe
        for tf in ['1h', '4h', '1d']:
            tf_dir = self.indicators_dir / tf
            if not tf_dir.exists():
                tf_dir = self.indicators_dir  # Fallback to flat structure
            
            # Load supertrend for this timeframe
            st_file = tf_dir / f"{symbol}_{tf}_supertrend.csv"
            if st_file.exists():
                df = pd.read_csv(st_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                # Create features
                features = pd.DataFrame(index=df.index)
                features[f'{tf}_trend'] = df['trend']
                features[f'{tf}_buy_signal'] = df['buy_signal'].astype(int)
                features[f'{tf}_sell_signal'] = df['sell_signal'].astype(int)
                
                # Add to list
                all_features.append(features)
        
        # Combine all timeframe features
        if all_features:
            # Align to daily timeframe
            daily_features = all_features[-1]  # 1d features
            
            # Resample other timeframes to daily
            for i, tf_features in enumerate(all_features[:-1]):
                tf_features_daily = tf_features.resample('D').last()
                # Forward fill to handle missing data
                tf_features_daily = tf_features_daily.fillna(method='ffill')
                
                # Align with daily index
                for col in tf_features_daily.columns:
                    daily_features[col] = tf_features_daily[col].reindex(daily_features.index)
            
            return daily_features.fillna(0)
        else:
            return pd.DataFrame()

# Prepare training data with MTF features
logger.info("Preparing MTF training data...")

mtf_engineer = MTFFeatureEngineer(DATA_DIR)
training_features = []
training_targets = []

for symbol in SACRED_SYMBOLS[:5]:  # Use first 5 symbols for faster training
    try:
        # Get regular features
        features, targets = mtf_engineer.build_feature_matrix(symbol)
        
        # Add MTF features
        mtf_features = mtf_engineer.build_mtf_features(symbol)
        
        if not features.empty and not mtf_features.empty:
            # Align indices
            common_idx = features.index.intersection(mtf_features.index)
            if len(common_idx) > 100:
                features = features.loc[common_idx]
                mtf_features = mtf_features.loc[common_idx]
                targets = targets.loc[common_idx]
                
                # Combine features
                combined = pd.concat([features, mtf_features], axis=1)
                
                training_features.append(combined)
                training_targets.append(targets['signal_1d'])
                
                logger.info(f"Added {len(combined)} samples from {symbol}")
                
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")

# Combine all data
if training_features:
    X = pd.concat(training_features, ignore_index=True)
    y = pd.concat(training_targets, ignore_index=True)
    
    # Remove NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Total training samples: {len(X)}, features: {X.shape[1]}")
    
    # Train ensemble
    ensemble = EnsembleTradingModel(DATA_DIR.parent / 'ml_models')
    
    model_params = {
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
        },
        'lightgbm': {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
        },
        'lstm': {
            'lstm_units': [32, 16],
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 10,
        }
    }
    
    training_results = ensemble.train_all_models(X, y, validation_split=0.2, **model_params)
    
    logger.info("\nTraining Results:")
    for model, results in training_results.items():
        if 'error' not in results:
            logger.info(f"{model}: Accuracy={results.get('accuracy', 0):.3f}")
    
    # Save models
    ensemble.save_ensemble("mtf_ensemble")
    logger.info("✅ MTF models trained and saved!")
else:
    logger.error("No training data available!")

# Step 4: Run backtest with MTF strategy
logger.info("\n" + "="*60)
logger.info("STEP 4: RUNNING BACKTEST WITH MTF STRATEGY")
logger.info("="*60)

from backtest.backtest_engine import BacktestEngine

# Run backtest
backtest = BacktestEngine(
    start_date="2023-01-01",
    end_date="2024-12-31",
    initial_capital=100000.0,
    commission_rate=0.001,
    slippage_rate=0.0005
)

try:
    results = backtest.run_backtest(train_split=0.0)  # Models already trained
    
    # Save results
    backtest.save_backtest_results("mtf_backtest")
    
    # Generate report
    report = backtest.generate_backtest_report()
    print("\n" + "="*80)
    print("FINAL BACKTEST RESULTS")
    print("="*80)
    print(report)
    
    # Save report
    report_path = DATA_DIR.parent / 'backtest_results' / 'mtf_backtest_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
except Exception as e:
    logger.error(f"Backtest failed: {e}")

logger.info("\n" + "="*80)
logger.info("MTF PIPELINE COMPLETED!")
logger.info("="*80)