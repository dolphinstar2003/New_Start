#!/usr/bin/env python3
"""
ML Commands Interface
Easy commands for ML model operations
"""
import asyncio
import sys
from pathlib import Path

def show_commands():
    """Show available ML commands"""
    print("\n" + "="*60)
    print("🤖 ML MODEL COMMANDS")
    print("="*60)
    
    commands = [
        ("1", "python run_ml_performance_comparison.py", "ML vs Traditional Strategy Comparison"),
        ("2", "python analyze_ml_features.py", "Feature Importance Analysis"),
        ("3", "python test_trained_models.py", "Test Available Models"),
        ("4", "python train_models_fixed.py", "Retrain Models (Fixed)"),
        ("5", "python verify_ml_setup.py", "Verify ML Setup"),
        ("", "", ""),
        ("Quick", "python -c \"import asyncio; from run_ml_performance_comparison import main; asyncio.run(main())\"", "Quick Performance Test"),
        ("Features", "python -c \"from analyze_ml_features import main; main()\"", "Quick Feature Analysis")
    ]
    
    print(f"\n📋 Available Commands:")
    print("-" * 60)
    
    for num, command, description in commands:
        if num and command:
            print(f"{num:<6} {description}")
            print(f"       {command}")
            print()
        elif description:  # Empty line
            print()
    
    print("🎯 Usage Examples:")
    print("   # Run comprehensive comparison")
    print("   python run_ml_performance_comparison.py")
    print()
    print("   # Check feature importance")
    print("   python analyze_ml_features.py")
    print()
    print("   # Quick test")
    print("   python ml_commands.py test")

def run_quick_test():
    """Run quick ML test"""
    print("🚀 Running Quick ML Test...")
    
    try:
        # Import and run
        from analyze_ml_features import analyze_random_forest_features
        importance_df = analyze_random_forest_features()
        
        if importance_df is not None:
            print("\n✅ Quick test successful!")
            print(f"📊 Top 5 features:")
            for i, row in importance_df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        else:
            print("❌ Quick test failed")
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")

def run_quick_comparison():
    """Run quick comparison"""
    print("🚀 Running Quick Comparison...")
    
    try:
        from run_ml_performance_comparison import MLPerformanceComparison
        
        comparison = MLPerformanceComparison()
        
        # Test with one symbol
        symbol = "GARAN"
        result = comparison.run_ml_backtest(symbol, days=10)
        
        print(f"\n✅ Quick ML backtest for {symbol}:")
        print(f"   Total trades: {result.get('total_trades', 0)}")
        print(f"   High confidence: {result.get('high_confidence_trades', 0)}")
        print(f"   Avg confidence: {result.get('avg_confidence', 0):.3f}")
        print(f"   Simulated return: {result.get('simulated_return', 0):.2f}%")
        
    except Exception as e:
        print(f"❌ Quick comparison error: {e}")

def main():
    """Main interface"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            run_quick_test()
        elif command == "compare":
            run_quick_comparison()
        elif command == "full":
            print("🚀 Running Full Comparison...")
            import subprocess
            subprocess.run([sys.executable, "run_ml_performance_comparison.py"])
        elif command == "features":
            print("🚀 Running Feature Analysis...")
            import subprocess
            subprocess.run([sys.executable, "analyze_ml_features.py"])
        else:
            print(f"❌ Unknown command: {command}")
            show_commands()
    else:
        show_commands()

if __name__ == "__main__":
    main()