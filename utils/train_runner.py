"""
ML Model Training Runner
Provides interface for training ML models from telegram bot
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, ML_MODELS_DIR


def run_training(symbols: Optional[List[str]] = None, force_retrain: bool = False) -> Dict:
    """
    Run ML model training
    
    Args:
        symbols: List of symbols to train (defaults to SACRED_SYMBOLS)
        force_retrain: Force retraining even if models exist
        
    Returns:
        Dict with training results
    """
    try:
        # Use sacred symbols if none specified
        if not symbols:
            symbols = SACRED_SYMBOLS
            
        logger.info(f"Starting ML model training for {len(symbols)} symbols")
        
        # Record start time
        start_time = datetime.now()
        
        # Import and run training
        from ml_models.train_models import ModelTrainer
        
        trainer = ModelTrainer()
        results = trainer.run_full_training_pipeline(
            train_start="2022-07-07",  # Start from when all indicators are available
            train_end="2024-06-30",    # Recent data
            test_split=0.2,
            symbols=symbols
        )
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Process results
        if results and 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            
            training_results = {
                'success': True,
                'models_trained': 3,  # XGBoost, LightGBM, LSTM
                'symbols_processed': len(symbols),
                'training_time_seconds': round(training_time, 2),
                'training_time_minutes': round(training_time / 60, 2),
                'avg_accuracy': round(eval_results.get('accuracy', 0) * 100, 2),
                'avg_f1_score': round(eval_results.get('f1_score', 0), 3),
                'total_samples': results['data_stats']['total_samples'],
                'model_types': ['XGBoost', 'LightGBM', 'LSTM', 'Ensemble'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Save training results
            save_training_results(training_results, results)
            
            return training_results
        else:
            return {
                'success': False,
                'error': 'Training failed to produce models'
            }
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def get_model_status() -> Dict:
    """Get current ML model status"""
    try:
        model_dir = Path(ML_MODELS_DIR) / 'saved_models'
        
        if not model_dir.exists():
            return {
                'models_exist': False,
                'model_count': 0,
                'last_training': None
            }
        
        # Count model files
        model_files = list(model_dir.glob('*.pkl')) + list(model_dir.glob('*.h5'))
        
        # Get latest modification time
        last_training = None
        if model_files:
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            last_training = datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat()
        
        # Check for model info file
        model_info = {}
        info_file = model_dir / 'model_info.json'
        if info_file.exists():
            with open(info_file, 'r') as f:
                model_info = json.load(f)
        
        return {
            'models_exist': len(model_files) > 0,
            'model_count': len(model_files),
            'last_training': last_training,
            'model_types': model_info.get('model_types', []),
            'avg_accuracy': model_info.get('avg_accuracy', 0),
            'symbols_trained': model_info.get('symbols_trained', 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        return {
            'models_exist': False,
            'error': str(e)
        }


def save_training_results(summary: Dict, detailed_results: Dict):
    """Save training results to file"""
    try:
        # Create analysis directory
        analysis_dir = Path("data/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = analysis_dir / f"training_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save model info for future reference
        model_dir = Path(ML_MODELS_DIR) / 'saved_models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        info_file = model_dir / 'model_info.json'
        with open(info_file, 'w') as f:
            json.dump({
                'last_training': summary['timestamp'],
                'model_types': summary['model_types'],
                'avg_accuracy': summary['avg_accuracy'],
                'symbols_trained': summary['symbols_processed'],
                'training_time_minutes': summary['training_time_minutes']
            }, f, indent=2)
            
        logger.info(f"Training results saved")
        
    except Exception as e:
        logger.error(f"Failed to save training results: {e}")


def get_training_parameters() -> Dict:
    """Get current training parameters"""
    return {
        'train_split': 0.8,  # 80% training data
        'val_split': 0.1,   # 10% validation data
        'test_split': 0.1,  # 10% test data
        'lookback_periods': 20,  # Days of historical data
        'prediction_horizon': 5,  # Days to predict ahead
        'min_samples': 100,  # Minimum samples for training
        'models': ['XGBoost', 'LightGBM', 'LSTM', 'Ensemble'],
        'epochs': 50,  # For neural networks
        'batch_size': 32,  # For neural networks
        'early_stopping': True,
        'feature_selection': True
    }