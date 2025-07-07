#!/usr/bin/env python3
"""
Check ML Model Status
Shows which models are trained and ready for use
"""
from pathlib import Path
import pandas as pd
from datetime import datetime

def check_model_status():
    """Check which ML models are available"""
    print("\n" + "="*60)
    print("🤖 ML/DL MODEL STATUS CHECK")
    print("="*60)
    
    model_dir = Path("ml_models/saved_models")
    
    print(f"\n📁 Model Directory: {model_dir.absolute()}")
    
    if not model_dir.exists():
        print("❌ Model directory not found!")
        return
    
    # Check for model files
    models = {
        "XGBoost Model": "xgboost_model.pkl",
        "XGBoost Optimized": "xgboost_optimized.pkl", 
        "Random Forest": "rf_model.pkl",
        "LSTM Model": "lstm_model.h5",
        "GRU Model": "gru_model.h5",
        "Ensemble Model": "ensemble_model.pkl"
    }
    
    print(f"\n📊 Model Availability:")
    print("-" * 50)
    
    trained_models = []
    missing_models = []
    
    for model_name, filename in models.items():
        model_path = model_dir / filename
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(model_path.stat().st_mtime)
            print(f"✅ {model_name:<20} | {size_mb:>6.1f} MB | {mod_time.strftime('%Y-%m-%d %H:%M')}")
            trained_models.append(model_name)
        else:
            print(f"❌ {model_name:<20} | Not Found")
            missing_models.append(model_name)
    
    # Summary
    print(f"\n📈 Summary:")
    print(f"  Trained Models: {len(trained_models)}/{len(models)}")
    print(f"  Ready for Use: {', '.join(trained_models) if trained_models else 'None'}")
    
    if missing_models:
        print(f"  Need Training: {', '.join(missing_models)}")
    
    # Test XGBoost model if available
    if "XGBoost Model" in trained_models:
        print(f"\n🧪 Testing XGBoost Model...")
        try:
            import joblib
            import numpy as np
            
            model = joblib.load(model_dir / "xgboost_model.pkl")
            
            # Create dummy features for testing
            dummy_features = np.random.randn(1, 80)  # 80 features
            prediction = model.predict(dummy_features)
            
            print(f"   ✅ Model loaded successfully")
            print(f"   🎯 Test prediction: {prediction[0]}")
            print(f"   📊 Model type: {type(model).__name__}")
            
        except Exception as e:
            print(f"   ❌ Model test failed: {e}")
    
    # Explain the process
    print(f"\n" + "="*60)
    print("🔄 MODEL TRAINING PROCESS")
    print("="*60)
    
    print(f"\n1. **Feature Engineering**")
    print(f"   • Technical indicators (RSI, MACD, Bollinger)")
    print(f"   • Price patterns (returns, volatility)")
    print(f"   • Volume analysis")
    print(f"   • Market breadth")
    
    print(f"\n2. **Model Training**")
    print(f"   • XGBoost: Gradient boosting for classification")
    print(f"   • LSTM: Sequence modeling for time series")
    print(f"   • Random Forest: Ensemble of decision trees")
    print(f"   • Target: Predict next-day price direction")
    
    print(f"\n3. **Backtest Usage**")
    print(f"   • Load trained models")
    print(f"   • Generate predictions for each day")
    print(f"   • Execute trades based on ML signals")
    print(f"   • Track performance vs traditional strategies")
    
    print(f"\n💡 To train missing models:")
    print(f"   python train_all_models.py")
    
    print(f"\n💡 To run ML backtests:")
    print(f"   python -m backtest.backtest_ml_xgboost")


if __name__ == "__main__":
    check_model_status()