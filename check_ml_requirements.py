#!/usr/bin/env python3
"""
ML Model Training Requirements Check
Verify all necessary modules and dependencies
"""
import sys
import subprocess
from pathlib import Path
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'Unknown')
            return True, version
        else:
            return False, None
    except ImportError:
        return False, None

def check_file_structure():
    """Check if necessary files and directories exist"""
    required_files = [
        "ml_models/__init__.py",
        "ml_models/feature_engineer.py", 
        "ml_models/base_model.py",
        "ml_models/xgboost_model.py",
        "ml_models/lstm_model.py",
        "ml_models/ensemble_model.py",
        "ml_models/train_models.py",
        "config/settings.py",
        "data/"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files

def main():
    print("\n" + "="*70)
    print("🔍 ML MODEL TRAINING REQUIREMENTS CHECK")
    print("="*70)
    
    # 1. Python packages check
    print("\n📦 Checking Python Packages:")
    print("-" * 50)
    
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"), 
        ("scikit-learn", "sklearn"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("tensorflow", "tensorflow"),
        ("keras", "keras"),
        ("torch", "torch"),
        ("joblib", "joblib"),
        ("optuna", "optuna"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("loguru", "loguru"),
        ("tqdm", "tqdm"),
        ("psutil", "psutil")
    ]
    
    installed_packages = []
    missing_packages = []
    
    for package_name, import_name in required_packages:
        is_installed, version = check_package(package_name, import_name)
        if is_installed:
            print(f"✅ {package_name:<15} | v{version}")
            installed_packages.append(package_name)
        else:
            print(f"❌ {package_name:<15} | Not Found")
            missing_packages.append(package_name)
    
    # 2. File structure check
    print(f"\n📁 Checking File Structure:")
    print("-" * 50)
    
    existing_files, missing_files = check_file_structure()
    
    for file_path in existing_files:
        print(f"✅ {file_path}")
    
    for file_path in missing_files:
        print(f"❌ {file_path}")
    
    # 3. Data availability check
    print(f"\n📊 Checking Data Availability:")
    print("-" * 50)
    
    data_dir = Path("data")
    if data_dir.exists():
        raw_dir = data_dir / "raw" / "1d"
        if raw_dir.exists():
            csv_files = list(raw_dir.glob("*_1d_raw.csv"))
            print(f"✅ Data directory: {data_dir}")
            print(f"✅ Raw data files: {len(csv_files)} CSV files found")
            
            # Show sample files
            for i, file in enumerate(csv_files[:5]):
                print(f"   📄 {file.name}")
            if len(csv_files) > 5:
                print(f"   ... and {len(csv_files) - 5} more files")
        else:
            print(f"❌ Raw data directory not found: {raw_dir}")
    else:
        print(f"❌ Data directory not found: {data_dir}")
    
    # 4. System resources check
    print(f"\n💻 Checking System Resources:")
    print("-" * 50)
    
    try:
        import psutil
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        print(f"🧠 RAM: {memory_gb:.1f} GB total, {memory_available_gb:.1f} GB available")
        
        if memory_gb >= 8:
            print("✅ Sufficient RAM for ML training")
        else:
            print("⚠️  Limited RAM - may affect large model training")
        
        # CPU
        cpu_count = psutil.cpu_count()
        print(f"⚡ CPU: {cpu_count} cores")
        
        # GPU check
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"🎮 GPU: {len(gpus)} GPU(s) detected")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
            else:
                print("🔧 GPU: CPU-only mode (no GPU detected)")
        except:
            print("🔧 GPU: Unable to check GPU status")
            
    except ImportError:
        print("❌ Cannot check system resources (psutil not available)")
    
    # 5. Summary and recommendations
    print(f"\n" + "="*70)
    print("📋 SUMMARY & RECOMMENDATIONS")
    print("="*70)
    
    print(f"\n✅ Ready Components:")
    print(f"   • Installed packages: {len(installed_packages)}/{len(required_packages)}")
    print(f"   • Existing files: {len(existing_files)}/{len(existing_files + missing_files)}")
    
    if missing_packages:
        print(f"\n❌ Missing Packages:")
        for pkg in missing_packages:
            print(f"   • {pkg}")
        
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    if missing_files:
        print(f"\n❌ Missing Files:")
        for file in missing_files:
            print(f"   • {file}")
    
    # Readiness assessment
    readiness_score = (len(installed_packages) / len(required_packages)) * 100
    
    print(f"\n🎯 Training Readiness: {readiness_score:.0f}%")
    
    if readiness_score >= 90:
        print("🚀 System ready for ML model training!")
        print("\n📝 Next steps:")
        print("   1. python check_feature_compatibility.py")
        print("   2. python train_all_models.py")
        print("   3. python run_ml_backtests.py")
    elif readiness_score >= 70:
        print("⚠️  System mostly ready, install missing packages first")
    else:
        print("❌ System needs significant setup before training")
    
    return readiness_score >= 70

if __name__ == "__main__":
    main()