#!/usr/bin/env python3
"""
Verify ML Setup
Quick test of all ML components
"""
import sys
import os

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Critical Imports...")
    
    tests = [
        ("numpy", "np"),
        ("pandas", "pd"), 
        ("sklearn", "from sklearn.ensemble import RandomForestClassifier"),
        ("xgboost", "import xgboost as xgb"),
        ("lightgbm", "import lightgbm as lgb"),
        ("joblib", "import joblib"),
        ("tensorflow", "import tensorflow as tf"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("optuna", "import optuna")
    ]
    
    passed = 0
    failed = 0
    
    for name, import_cmd in tests:
        try:
            if "from" in import_cmd or "as" in import_cmd:
                exec(import_cmd)
            else:
                exec(f"import {import_cmd}")
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    return passed, failed

def test_tensorflow():
    """Test TensorFlow specifically"""
    print("\n🧠 Testing TensorFlow...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test basic operation
        x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        y = tf.reduce_sum(x)
        print(f"✅ Basic tensor operations work")
        
        # Check devices
        devices = tf.config.list_physical_devices()
        print(f"✅ Available devices: {len(devices)}")
        for device in devices:
            print(f"   📱 {device}")
            
        return True
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        return False

def test_model_files():
    """Test model file structure"""
    print("\n📁 Testing File Structure...")
    
    required_files = [
        "ml_models/feature_engineer.py",
        "ml_models/xgboost_model.py", 
        "ml_models/lstm_model.py",
        "ml_models/ensemble_model.py",
        "ml_models/train_models.py"
    ]
    
    missing = []
    existing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            existing.append(file_path)
        else:
            print(f"❌ {file_path}")
            missing.append(file_path)
    
    return len(existing), len(missing)

def test_data_access():
    """Test data file access"""
    print("\n📊 Testing Data Access...")
    
    try:
        sys.path.append('.')
        from config.settings import SACRED_SYMBOLS, DATA_DIR
        
        print(f"✅ Config loaded")
        print(f"✅ Sacred symbols: {len(SACRED_SYMBOLS)} symbols")
        print(f"✅ Data directory: {DATA_DIR}")
        
        # Test data file access
        data_files = list(DATA_DIR.glob("raw/1d/*_1d_raw.csv"))
        print(f"✅ Found {len(data_files)} data files")
        
        if len(data_files) > 0:
            print(f"   📄 Sample: {data_files[0].name}")
        
        return len(data_files) > 0
        
    except Exception as e:
        print(f"❌ Data access test failed: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("🔍 ML SETUP VERIFICATION")
    print("="*60)
    
    print(f"\n🐍 Environment Info:")
    print(f"   Python: {sys.executable}")
    print(f"   Virtual env: {os.environ.get('VIRTUAL_ENV', 'None')}")
    
    # Run tests
    imports_passed, imports_failed = test_imports()
    tf_ok = test_tensorflow()
    files_existing, files_missing = test_model_files()
    data_ok = test_data_access()
    
    # Summary
    print(f"\n" + "="*60)
    print("📋 VERIFICATION SUMMARY")
    print("="*60)
    
    print(f"\n📦 Package Imports: {imports_passed}/{imports_passed + imports_failed}")
    print(f"🧠 TensorFlow: {'✅ OK' if tf_ok else '❌ Failed'}")
    print(f"📁 Model Files: {files_existing}/{files_existing + files_missing}")
    print(f"📊 Data Access: {'✅ OK' if data_ok else '❌ Failed'}")
    
    # Overall status
    total_score = (imports_passed + int(tf_ok) + files_existing + int(data_ok))
    max_score = (imports_passed + imports_failed + 1 + files_existing + files_missing + 1)
    
    readiness = (total_score / max_score) * 100
    
    print(f"\n🎯 ML Training Readiness: {readiness:.0f}%")
    
    if readiness >= 80:
        print("🚀 Ready for ML model training!")
        print("\n📝 Next Steps:")
        print("   1. python prepare_training_features.py")
        print("   2. python train_all_models.py")
        print("   3. python run_ml_backtests.py")
    else:
        print("⚠️  Setup needs improvement before training")
        
        if imports_failed > 0:
            print("   🔧 Fix package imports")
        if not tf_ok:
            print("   🔧 Fix TensorFlow installation")
        if files_missing > 0:
            print("   🔧 Create missing model files")
        if not data_ok:
            print("   🔧 Fix data access")

if __name__ == "__main__":
    main()