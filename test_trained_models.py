#!/usr/bin/env python3
"""
Test Trained Models
Test the models we have successfully trained
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.enhanced_feature_engineer import EnhancedFeatureEngineer

def test_available_models():
    """Test all available trained models"""
    print("\n" + "="*60)
    print("🧪 TESTING TRAINED MODELS")
    print("="*60)
    
    model_dir = Path("ml_models/saved_models")
    
    # Check available models
    available_models = {
        'Random Forest (Enhanced)': model_dir / 'random_forest_enhanced.pkl',
        'XGBoost (Original)': model_dir / 'xgboost_model.pkl',
        'XGBoost (Optimized)': model_dir / 'xgboost_optimized.pkl'
    }
    
    print("\n📦 Available Models:")
    working_models = {}
    
    for name, path in available_models.items():
        if path.exists():
            try:
                model = joblib.load(path)
                print(f"✅ {name}: {path.name}")
                working_models[name] = {'path': path, 'model': model}
            except Exception as e:
                print(f"❌ {name}: Failed to load - {e}")
        else:
            print(f"❌ {name}: File not found")
    
    if not working_models:
        print("❌ No working models found!")
        return
    
    # Test with sample data
    print(f"\n🧪 Testing with sample data...")
    
    fe = EnhancedFeatureEngineer(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]  # GARAN
    
    try:
        features_df, targets_df = fe.build_enhanced_feature_matrix(symbol, '1d')
        print(f"✅ Sample data loaded: {features_df.shape}")
        
        # Test each working model
        for name, model_info in working_models.items():
            print(f"\n🤖 Testing {name}:")
            
            try:
                model = model_info['model']
                
                # Get last 5 rows for prediction
                test_features = features_df.tail(5)
                
                if 'Random Forest (Enhanced)' in name:
                    # This model expects the enhanced feature set
                    predictions = model.predict(test_features)
                    probabilities = model.predict_proba(test_features)
                    
                    print(f"   ✅ Predictions: {predictions}")
                    print(f"   ✅ Max probability: {probabilities.max(axis=1)}")
                    print(f"   ✅ Feature importance available: {hasattr(model, 'feature_importances_')}")
                    
                elif 'XGBoost' in name:
                    # These models expect the original 18-feature set
                    # We need to adapt our features or retrain
                    expected_features = getattr(model, 'n_features_in_', 'unknown')
                    print(f"   ⚠️  Expected features: {expected_features}")
                    print(f"   ⚠️  Our features: {test_features.shape[1]}")
                    
                    if expected_features == test_features.shape[1]:
                        predictions = model.predict(test_features)
                        print(f"   ✅ Predictions: {predictions}")
                    else:
                        print(f"   ❌ Feature mismatch - need retraining")
                
            except Exception as e:
                print(f"   ❌ Testing failed: {e}")
    
    except Exception as e:
        print(f"❌ Sample data preparation failed: {e}")
    
    # Model summary
    print(f"\n" + "="*60)
    print("📊 MODEL SUMMARY")
    print("="*60)
    
    print(f"\n✅ Working Models: {len(working_models)}")
    for name in working_models.keys():
        print(f"   • {name}")
    
    print(f"\n🎯 Status:")
    if 'Random Forest (Enhanced)' in working_models:
        print(f"   ✅ Enhanced Random Forest is ready for backtesting")
    
    old_xgb_count = sum(1 for name in working_models if 'XGBoost' in name and 'Enhanced' not in name)
    if old_xgb_count > 0:
        print(f"   ⚠️  {old_xgb_count} old XGBoost models need feature compatibility fix")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Test Random Forest in backtest")
    print(f"   2. Retrain XGBoost with enhanced features")
    print(f"   3. Train LSTM/GRU models")


if __name__ == "__main__":
    test_available_models()