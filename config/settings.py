"""
BIST100 Trading System Configuration
According to README_GOD_MODE.md sacred rules
"""
import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
ML_MODELS_DIR = BASE_DIR / "ml_models"
PORTFOLIO_DIR = BASE_DIR / "portfolio"
TRADING_DIR = BASE_DIR / "trading"
UTILS_DIR = BASE_DIR / "utils"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
INDICATORS_DIR = DATA_DIR / "indicators"
ANALYSIS_DIR = DATA_DIR / "analysis"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Ensure directories exist
for dir_path in [
    DATA_DIR, LOG_DIR, ML_MODELS_DIR, PORTFOLIO_DIR, TRADING_DIR, UTILS_DIR,
    RAW_DATA_DIR, INDICATORS_DIR, ANALYSIS_DIR, PREDICTIONS_DIR
]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Sacred 20 Symbols from README_GOD_MODE.md
SACRED_SYMBOLS = [
    # Bankalar (4)
    'GARAN', 'AKBNK', 'ISCTR', 'YKBNK',
    # Holdingler (3)
    'SAHOL', 'KCHOL', 'SISE',
    # Sanayi (3)
    'EREGL', 'KRDMD', 'TUPRS',
    # Teknoloji/Savunma (3)
    'ASELS', 'THYAO', 'TCELL',
    # Tüketim (3)
    'BIMAS', 'MGROS', 'ULKER',
    # Enerji/Altyapı (2)
    'AKSEN', 'ENKAI',
    # Fırsat (2)
    'PETKM', 'KOZAL'
]

# Timeframe Hierarchy
TIMEFRAMES = {
    '1d': {'description': 'Ana trend', 'history_years': 3},
    '1h': {'description': 'Gün içi momentum', 'history_months': 6},
    '15m': {'description': 'Hassas timing', 'history_months': 2}
}

# Core 5 Indicators (TradingView)
CORE_INDICATORS = [
    'supertrend',     # Ana trend filtresi
    'adx_di',         # Trend gücü
    'squeeze_momentum',  # Volatilite patlaması
    'wavetrend',      # Momentum osilatörü
    'macd_custom'     # Multi-timeframe
]

# Additional Indicators
ADDITIONAL_INDICATORS = [
    'volume_profile',
    'market_profile',
    'order_flow_imbalance',
    'vwap_anchored',
    'auto_fibonacci',
    'dynamic_sr',
    'pivot_points',
    'market_breadth',
    'correlation_matrix',
    'custom_volatility_bands'
]

# All indicators
ALL_INDICATORS = CORE_INDICATORS + ADDITIONAL_INDICATORS

# ML Model Settings
ML_CONFIG = {
    'models': {
        'xgboost': {'weight': 0.4, 'type': 'fast_signal'},
        'lightgbm': {'weight': 0.2, 'type': 'fast_signal'},
        'random_forest': {'weight': 0.1, 'type': 'ensemble'},
        'lstm': {'weight': 0.3, 'type': 'deep_analysis'}
    },
    'ensemble_weights': {
        'ml_signal': 0.4,
        'lstm_signal': 0.3,
        'technical_signal': 0.2,
        'sentiment_signal': 0.1
    }
}

# System Performance Limits
PERFORMANCE_LIMITS = {
    'max_ram_gb': 12,
    'cpu_limit_percent': 80,
    'process_pool_workers': 4,
    'thread_pool_workers': 20,
    'batch_size': 100,
    'memory_cache_gb': 1
}

# Risk Management Rules
RISK_MANAGEMENT = {
    'max_position_percent': 10,  # Max pozisyon: Sermayenin %10'u
    'max_daily_loss_percent': 5,  # Max kayıp/gün: %5
    'stop_loss_mandatory': True,
    'max_open_positions': 5,
    'min_trades_per_day': 3,
    'target_win_rate': 0.55,
    'target_profit_factor': 1.5,
    'monthly_target_percent': 8,  # Aylık hedef: %8-10
    'max_drawdown_percent': 15,
    'target_sharpe_ratio': 1.5
}

# Trading Time Zones
TRADING_HOURS = {
    '09:30-10:00': 'gap_trading',
    '10:00-12:00': 'trend_following',
    '12:00-14:00': 'range_trading',
    '14:00-16:30': 'momentum',
    '16:30-18:00': 'position_closing'
}

# Signal Hierarchy
SIGNAL_STRENGTH = {
    'STRONG_BUY': 'ML sinyali + Teknik onay',
    'STRONG_SELL': 'ML sinyali + Teknik onay',
    'WEAK_BUY': 'Sadece ML sinyali',
    'WEAK_SELL': 'Sadece ML sinyali',
    'HOLD': 'Sadece teknik sinyal'
}

# AlgoLab API Configuration
ALGOLAB_CONFIG = {
    'api_key': os.getenv('ALGOLAB_API_KEY'),
    'username': os.getenv('ALGOLAB_USERNAME'),
    'password': os.getenv('ALGOLAB_PASSWORD'),
    'hash': os.getenv('ALGOLAB_HASH'),
    'base_url': 'https://algolab.com.tr'
}

# Database Configuration
DATABASE_CONFIG = {
    'redis': {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'db': int(os.getenv('REDIS_DB', 0))
    }
}

# Logging Configuration
LOG_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}',
    'rotation': '100 MB',
    'retention': '1 week',
    'compression': 'zip'
}

# CSV File Naming Convention
CSV_NAMING = {
    'raw': '{symbol}_{timeframe}_raw.csv',
    'indicator': '{symbol}_{timeframe}_{indicator}.csv',
    'analysis': '{symbol}_risk_metrics.csv',
    'prediction': '{symbol}_ml_signals.csv'
}