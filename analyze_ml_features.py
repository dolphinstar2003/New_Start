#!/usr/bin/env python3
"""
ML Feature Importance Analysis
Analyze which features are most important for predictions
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.enhanced_feature_engineer import EnhancedFeatureEngineer


def analyze_random_forest_features():
    """Analyze Random Forest feature importance"""
    print("\n" + "="*60)
    print("🔍 RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Load model
    model_path = Path("ml_models/saved_models/random_forest_enhanced.pkl")
    if not model_path.exists():
        print("❌ Random Forest model not found!")
        return
    
    model = joblib.load(model_path)
    
    # Get sample features to understand feature names
    fe = EnhancedFeatureEngineer(DATA_DIR)
    symbol = SACRED_SYMBOLS[0]
    
    try:
        features_df, _ = fe.build_enhanced_feature_matrix(symbol, '1d')
        
        # Remove symbol_id if present
        if 'symbol_id' in features_df.columns:
            features_df = features_df.drop(columns=['symbol_id'])
        
        feature_names = features_df.columns.tolist()
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 20 Most Important Features:")
        print("-" * 50)
        
        for i, row in importance_df.head(20).iterrows():
            print(f"{row['feature']:<30} {row['importance']:.4f}")
        
        # Feature categories
        print(f"\n🏷️ Feature Importance by Category:")
        print("-" * 50)
        
        categories = {
            'Price Features': [f for f in feature_names if any(x in f for x in ['returns', 'ratio', 'price', 'close'])],
            'Technical Indicators': [f for f in feature_names if any(x in f for x in ['macd', 'wt', 'adx', 'st'])],
            'Volatility Features': [f for f in feature_names if any(x in f for x in ['volatility', 'atr'])],
            'Volume Features': [f for f in feature_names if 'volume' in f],
            'Time Features': [f for f in feature_names if any(x in f for x in ['day', 'week', 'month'])],
            'Advanced Features': [f for f in feature_names if any(x in f for x in ['regime', 'consensus', 'acceleration'])]
        }
        
        category_importance = {}
        for category, features in categories.items():
            category_features = [f for f in features if f in feature_names]
            if category_features:
                category_indices = [feature_names.index(f) for f in category_features]
                category_importance[category] = importances[category_indices].sum()
            else:
                category_importance[category] = 0
        
        for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{category:<20} {importance:.4f}")
        
        # Model performance info
        print(f"\n🤖 Model Information:")
        print(f"   Model Type: {type(model).__name__}")
        print(f"   Number of Features: {len(feature_names)}")
        print(f"   Number of Trees: {getattr(model, 'n_estimators', 'N/A')}")
        
        if hasattr(model, 'oob_score_'):
            print(f"   OOB Score: {model.oob_score_:.3f}")
        
        # Save importance plot data
        importance_df.to_csv('ml_feature_importance.csv', index=False)
        print(f"\n📁 Feature importance saved to: ml_feature_importance.csv")
        
        return importance_df
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_feature_importance_plot(importance_df):
    """Create feature importance visualization"""
    if importance_df is None:
        return
    
    try:
        # Set style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top 15 features
        top_features = importance_df.head(15)
        
        ax1.barh(range(len(top_features)), top_features['importance'])
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'])
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('Top 15 Most Important Features')
        ax1.grid(axis='x', alpha=0.3)
        
        # Feature importance distribution
        ax2.hist(importance_df['importance'], bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Importance Score')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Distribution of Feature Importance Scores')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_feature_importance_plot.png', dpi=300, bbox_inches='tight')
        print(f"📊 Feature importance plot saved: ml_feature_importance_plot.png")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            print("📊 Plot saved but cannot display (no GUI)")
        
    except Exception as e:
        print(f"❌ Plot creation failed: {e}")


def main():
    """Main analysis function"""
    print("🚀 Starting ML Feature Analysis...")
    
    try:
        importance_df = analyze_random_forest_features()
        create_feature_importance_plot(importance_df)
        
        print(f"\n✅ Feature analysis completed!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


if __name__ == "__main__":
    main()