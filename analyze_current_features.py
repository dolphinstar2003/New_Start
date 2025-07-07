#!/usr/bin/env python3
"""
Analyze Current Feature Engineering
Check current feature structure and count
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ml_models.feature_engineer import FeatureEngineer
from config.settings import SACRED_SYMBOLS, DATA_DIR

def analyze_features():
    print("\n" + "="*60)
    print("🔍 CURRENT FEATURE ANALYSIS")
    print("="*60)
    
    # Initialize feature engineer
    fe = FeatureEngineer(DATA_DIR)
    
    # Test on first symbol
    symbol = SACRED_SYMBOLS[0]
    print(f"\n📊 Analyzing features for: {symbol}")
    
    try:
        # Build feature matrix
        features_df, targets_df = fe.build_feature_matrix(symbol, '1d')
        
        print(f"\n📈 Feature Matrix Info:")
        print(f"   Shape: {features_df.shape}")
        print(f"   Features: {features_df.shape[1]} columns")
        print(f"   Samples: {features_df.shape[0]} rows")
        print(f"   Targets: {targets_df.shape}")
        
        print(f"\n📋 Feature Categories:")
        
        # Categorize features
        feature_names = features_df.columns.tolist()
        
        categories = {
            'Price Features': [f for f in feature_names if any(x in f for x in ['returns', 'ratio', 'gap', 'volatility', 'position'])],
            'Technical Indicators': [f for f in feature_names if any(x in f for x in ['supertrend', 'adx', 'macd', 'wt', 'squeeze', 'momentum'])],
            'Time Features': [f for f in feature_names if any(x in f for x in ['day', 'week', 'month', 'quarter'])],
            'SMA Features': [f for f in feature_names if 'sma' in f],
            'Other Features': []
        }
        
        # Find uncategorized features
        categorized = []
        for cat_features in categories.values():
            categorized.extend(cat_features)
        
        categories['Other Features'] = [f for f in feature_names if f not in categorized]
        
        # Display categories
        total_count = 0
        for category, features in categories.items():
            if features:
                print(f"\n   {category}: {len(features)} features")
                total_count += len(features)
                for feature in features[:5]:  # Show first 5
                    print(f"     • {feature}")
                if len(features) > 5:
                    print(f"     ... and {len(features) - 5} more")
        
        print(f"\n📊 Summary:")
        print(f"   Total Features: {total_count}")
        print(f"   Data Quality: {features_df.isna().sum().sum()} NaN values")
        
        # Check feature compatibility with existing models
        print(f"\n🤖 Model Compatibility:")
        
        # Check XGBoost model expectations
        try:
            import joblib
            model_path = Path("ml_models/saved_models/xgboost_model.pkl")
            if model_path.exists():
                model = joblib.load(model_path)
                print(f"   Current XGBoost expects: {model.n_features_in_} features")
                print(f"   New feature set has: {features_df.shape[1]} features")
                
                if model.n_features_in_ != features_df.shape[1]:
                    print(f"   ⚠️  Feature count mismatch - retraining needed")
                else:
                    print(f"   ✅ Feature count matches")
            else:
                print(f"   📝 No existing XGBoost model found")
        except Exception as e:
            print(f"   ❌ Error checking model: {e}")
        
        return features_df, targets_df
        
    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def recommend_improvements(features_df):
    """Recommend feature engineering improvements"""
    print(f"\n" + "="*60)
    print("💡 FEATURE ENGINEERING RECOMMENDATIONS")
    print("="*60)
    
    if features_df is None:
        print("❌ Cannot provide recommendations - feature analysis failed")
        return
    
    print(f"\n🎯 Current Status: {features_df.shape[1]} features")
    
    print(f"\n📈 Recommended Additions:")
    print(f"   1. RSI features (multiple periods)")
    print(f"   2. Bollinger Bands features") 
    print(f"   3. Fibonacci retracement levels")
    print(f"   4. Volume indicators (OBV, Volume Profile)")
    print(f"   5. Market breadth features")
    print(f"   6. Cross-asset correlations")
    print(f"   7. Regime detection features")
    print(f"   8. Higher timeframe features")
    
    print(f"\n🔧 Improvements Needed:")
    print(f"   • Feature scaling/normalization")
    print(f"   • Feature selection (remove redundant)")
    print(f"   • Lag features for time series")
    print(f"   • Rolling statistics improvements")
    
    print(f"\n🎯 Target: 80-100 high-quality features")
    print(f"   Current: {features_df.shape[1]} features")
    print(f"   Gap: {max(0, 80 - features_df.shape[1])} features to add")

if __name__ == "__main__":
    features_df, targets_df = analyze_features()
    recommend_improvements(features_df)