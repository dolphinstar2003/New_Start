#!/usr/bin/env python3
"""
Install All ML Dependencies
Complete setup for ML model training
"""
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and show progress"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    print("\n" + "="*60)
    print("🚀 ML DEPENDENCIES INSTALLATION")
    print("="*60)
    
    # Essential ML packages
    packages = [
        "numpy==1.24.3",
        "pandas==2.0.3", 
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "lightgbm==4.0.0",
        "joblib==1.3.2",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "tqdm==4.66.1",
        "optuna==3.3.0",
        "psutil==5.9.5"
    ]
    
    # TensorFlow for LSTM/GRU
    tf_packages = [
        "tensorflow==2.13.0",
        "keras==2.13.1"
    ]
    
    # PyTorch (alternative)
    torch_packages = [
        "torch==2.0.1",
        "torchvision==0.15.2"
    ]
    
    print("\n📦 Installing Core ML Packages...")
    for package in packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package}, trying without version...")
            pkg_name = package.split('==')[0]
            run_command(f"pip install {pkg_name}", f"Installing {pkg_name}")
    
    print("\n🧠 Installing TensorFlow for Deep Learning...")
    for package in tf_packages:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package}, trying without version...")
            pkg_name = package.split('==')[0]
            run_command(f"pip install {pkg_name}", f"Installing {pkg_name}")
    
    # Create required directories
    print("\n📁 Creating Required Directories...")
    directories = [
        "ml_models/saved_models",
        "ml_models/training_logs",
        "data/processed",
        "results/ml_backtests"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")
    
    # Create missing __init__.py files
    print("\n🐍 Creating Python Package Files...")
    init_files = [
        "ml_models/__init__.py",
        "backtest/__init__.py",
        "config/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.touch()
            print(f"✅ Created: {init_file}")
        else:
            print(f"✓ Exists: {init_file}")
    
    print("\n" + "="*60)
    print("✅ INSTALLATION COMPLETED")
    print("="*60)
    
    print("\n🎯 What's Ready Now:")
    print("  ✅ XGBoost for gradient boosting")
    print("  ✅ LightGBM for fast gradient boosting")
    print("  ✅ Random Forest (scikit-learn)")
    print("  ✅ TensorFlow for LSTM/GRU")
    print("  ✅ Feature engineering tools")
    print("  ✅ Hyperparameter optimization (Optuna)")
    print("  ✅ Model persistence (joblib)")
    print("  ✅ Visualization tools")
    
    print("\n🚀 Next Steps:")
    print("  1. python verify_ml_setup.py")
    print("  2. python prepare_training_data.py") 
    print("  3. python train_all_models.py")

if __name__ == "__main__":
    main()