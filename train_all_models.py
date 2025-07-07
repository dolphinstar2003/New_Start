#!/usr/bin/env python3
"""
Train All ML/DL Models
Comprehensive model training for BIST trading system
"""
import asyncio
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS
from ml_models.train_models import ModelTrainer


async def main():
    """Train all models sequentially"""
    print("\n" + "="*60)
    print("🤖 ML/DL MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train on first 10 symbols for speed
    symbols = SACRED_SYMBOLS[:10]
    
    print(f"\n📊 Training on {len(symbols)} symbols:")
    for i, symbol in enumerate(symbols, 1):
        print(f"  {i:2d}. {symbol}")
    
    print(f"\n🔄 Starting training process...")
    
    try:
        # Prepare training data
        print("1. Preparing training data...")
        X_train, X_test, y_train, y_test = trainer.prepare_training_data(
            symbols=symbols,
            start_date="2023-01-01",
            end_date="2024-12-01"
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        
        # Train ensemble model
        print("\n2. Training ensemble model...")
        results = trainer.train_ensemble_model(X_train, X_test, y_train, y_test)
        
        print(f"   Training completed!")
        print(f"   Accuracy: {results.get('accuracy', 0):.3f}")
        print(f"   Precision: {results.get('precision', 0):.3f}")
        print(f"   Recall: {results.get('recall', 0):.3f}")
        
        # Save models
        print("\n3. Saving models...")
        trainer.save_all_models()
        print("   Models saved successfully!")
        
        print("\n" + "="*60)
        print("✅ MODEL TRAINING COMPLETED")
        print("="*60)
        
        print("\nTrained models are now ready for:")
        print("  • ML-based backtesting")
        print("  • Real-time trading signals")
        print("  • Performance comparison")
        
        print(f"\nModel files saved in: {trainer.model_dir}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Model training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())