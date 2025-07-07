#!/usr/bin/env python3
"""
Test Enhanced Feature Engineering
Compare old vs new feature engineering
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ml_models.feature_engineer import FeatureEngineer
from ml_models.enhanced_feature_engineer import EnhancedFeatureEngineer
from config.settings import SACRED_SYMBOLS, DATA_DIR

def compare_feature_engineers():
    print("\n" + "="*70)
    print("🔬 FEATURE ENGINEERING COMPARISON")
    print("="*70)
    
    # Test symbol
    symbol = SACRED_SYMBOLS[0]  # GARAN
    print(f"\n📊 Testing with symbol: {symbol}")
    
    # Initialize both feature engineers
    fe_old = FeatureEngineer(DATA_DIR)
    fe_new = EnhancedFeatureEngineer(DATA_DIR)
    
    print(f"\n1️⃣ Original Feature Engineer:")
    try:
        features_old, targets_old = fe_old.build_feature_matrix(symbol, '1d')
        print(f"   ✅ Features: {features_old.shape}")
        print(f"   ✅ Targets: {targets_old.shape}")
        print(f"   ✅ NaN count: {features_old.isna().sum().sum()}")
        print(f"   ✅ Feature names (first 10): {list(features_old.columns[:10])}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        features_old, targets_old = None, None
    
    print(f"\n2️⃣ Enhanced Feature Engineer:")
    try:
        features_new, targets_new = fe_new.build_enhanced_feature_matrix(symbol, '1d')
        print(f"   ✅ Features: {features_new.shape}")
        print(f"   ✅ Targets: {targets_new.shape}")
        print(f"   ✅ NaN count: {features_new.isna().sum().sum()}")
        print(f"   ✅ Feature names (first 10): {list(features_new.columns[:10])}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        features_new, targets_new = None, None
    
    # Comparison
    print(f"\n" + "="*70)
    print("📈 COMPARISON RESULTS")
    print("="*70)
    
    if features_old is not None and features_new is not None:
        print(f"\n📊 Feature Count:")
        print(f"   Original: {features_old.shape[1]} features")
        print(f"   Enhanced: {features_new.shape[1]} features")
        print(f"   Difference: {features_new.shape[1] - features_old.shape[1]:+d} features")
        
        print(f"\n🧹 Data Quality:")
        old_nan_ratio = features_old.isna().sum().sum() / (features_old.shape[0] * features_old.shape[1])
        new_nan_ratio = features_new.isna().sum().sum() / (features_new.shape[0] * features_new.shape[1])
        print(f"   Original NaN ratio: {old_nan_ratio:.2%}")
        print(f"   Enhanced NaN ratio: {new_nan_ratio:.2%}")
        print(f"   Quality improvement: {old_nan_ratio - new_nan_ratio:+.2%}")
        
        print(f"\n📅 Data Coverage:")
        print(f"   Original: {features_old.shape[0]} samples")
        print(f"   Enhanced: {features_new.shape[0]} samples")
        
        # Feature categories
        print(f"\n🏷️ New Feature Categories:")
        new_feature_names = features_new.columns.tolist()
        
        categories = {
            'Returns': [f for f in new_feature_names if 'returns' in f or 'roc' in f],
            'Price Ratios': [f for f in new_feature_names if 'ratio' in f],
            'Volatility': [f for f in new_feature_names if 'volatility' in f or 'atr' in f],
            'Technical': [f for f in new_feature_names if any(x in f for x in ['macd', 'wt', 'adx', 'st'])],
            'Volume': [f for f in new_feature_names if 'volume' in f],
            'Momentum': [f for f in new_feature_names if 'momentum' in f],
            'Advanced': [f for f in new_feature_names if any(x in f for x in ['regime', 'consensus', 'acceleration'])]
        }
        
        for category, features in categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
        
    else:
        if features_old is None:
            print("❌ Original feature engineer failed")
        if features_new is None:
            print("❌ Enhanced feature engineer failed")
    
    print(f"\n💡 Recommendations:")
    if features_new is not None:
        if features_new.shape[1] >= 80:
            print("   ✅ Feature count is in target range (80-100)")
        else:
            print(f"   ⚠️  Need {80 - features_new.shape[1]} more features")
        
        if features_new.isna().sum().sum() == 0:
            print("   ✅ No NaN values - excellent data quality")
        else:
            print("   ⚠️  Some NaN values remain")
        
        print("   🚀 Ready for model training!")
    
    return features_new, targets_new

if __name__ == "__main__":
    features, targets = compare_feature_engineers()