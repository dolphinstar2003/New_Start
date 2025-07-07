"""
ML-based Backtest using XGBoost
Predict price movements using trained XGBoost model
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import joblib
import asyncio

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.feature_engineer import FeatureEngineer
# XGBoost model is loaded from pickle file


class MLXGBoostBacktest:
    """Machine Learning backtest using XGBoost predictions"""
    
    def __init__(self):
        self.initial_capital = 100000
        self.position_size_pct = 0.1
        self.stop_loss_pct = 0.03
        self.take_profit_pct = 0.06
        self.commission = 0.001
        
        # ML specific parameters
        self.prediction_threshold = 0.4  # Lower threshold for more trades
        self.feature_engineer = FeatureEngineer(DATA_DIR)
        self.model = None
        self.model_path = Path(__file__).parent.parent / 'ml_models' / 'saved_models' / 'xgboost_model.pkl'
        
        # Risk management
        self.max_drawdown_limit = 0.15
        self.max_positions = 5
        
        logger.info("MLXGBoostBacktest initialized")
    
    def load_model(self):
        """Load pre-trained XGBoost model"""
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.warning(f"Model not found at {self.model_path}, training new model...")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new XGBoost model if none exists"""
        from ml_models.train_models import train_xgboost_model
        
        # Train on all sacred symbols
        logger.info("Training new XGBoost model...")
        self.model = train_xgboost_model(SACRED_SYMBOLS[:10])  # Train on first 10 symbols
        
        # Save model
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def prepare_features(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Prepare features for ML model"""
        # Use feature engineer to build complete feature matrix
        features_df, targets_df = self.feature_engineer.build_feature_matrix(symbol, timeframe)
        
        if features_df.empty:
            logger.warning(f"No features available for {symbol}")
            return pd.DataFrame()
        
        # Add targets to features for backtest (we need the actual returns)
        if 'signal_1d' in targets_df.columns:
            features_df['target'] = targets_df['signal_1d']  # Use 1-day signal as target
        
        # Load price data to add close prices
        price_data = self.feature_engineer.load_symbol_data(symbol, timeframe)
        if 'price' in price_data:
            features_df['close'] = price_data['price']['close']
        
        return features_df
    
    def generate_ml_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using ML model"""
        if self.model is None:
            self.load_model()
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        if features_df.empty:
            return pd.DataFrame()
        
        # Get feature columns (exclude target if present)
        feature_cols = [col for col in features_df.columns if col not in ['target', 'returns', 'signal']]
        X = features_df[feature_cols]
        
        # Make predictions
        try:
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                predictions_proba = self.model.predict_proba(X)
                # We have 4 classes: 0=sell(-1), 1=hold(0), 2=buy(1), 3=strong_buy(2)
                # Get probability of buy (class 2) + strong buy (class 3)
                buy_confidence = predictions_proba[:, 2] + predictions_proba[:, 3]
                sell_confidence = predictions_proba[:, 0]
            else:
                # For regression models
                predictions = self.model.predict(X)
                buy_confidence = predictions
                sell_confidence = 1 - predictions
            
            # Create signals based on confidence threshold
            signals = pd.DataFrame(index=features_df.index)
            signals['ml_confidence'] = buy_confidence
            signals['signal'] = 0
            
            # Buy signal when confidence is high
            signals.loc[buy_confidence > self.prediction_threshold, 'signal'] = 1
            
            # Sell signal when sell confidence is high
            signals.loc[sell_confidence > self.prediction_threshold, 'signal'] = -1
            
            # Add price data
            signals['close'] = features_df['close'] if 'close' in features_df.columns else 0
            
            # Add position sizing based on confidence
            signals['position_size'] = np.where(
                signals['signal'] != 0,
                np.clip((np.abs(buy_confidence - 0.5) * 2), 0.5, 1.0),  # Scale confidence to position size
                0
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return pd.DataFrame()
    
    async def backtest_symbol(self, symbol: str, days: int) -> List[Dict]:
        """Run ML backtest for single symbol"""
        # Prepare features
        features_df = self.prepare_features(symbol)
        
        if features_df.empty:
            return []
        
        # Generate signals
        signals = self.generate_ml_signals(features_df)
        
        if signals.empty:
            return []
        
        # Filter to requested days
        end_date = signals.index[-1]
        start_date = end_date - timedelta(days=days)
        signals = signals[signals.index >= start_date]
        
        # Simulate trading
        trades = []
        position = None
        capital = self.initial_capital
        
        for date, row in signals.iterrows():
            current_price = row['close']
            
            # Check exit conditions
            if position:
                # Stop loss
                if current_price <= position['stop_loss']:
                    return_pct = (position['stop_loss'] - position['entry_price']) / position['entry_price']
                    exit_reason = 'stop_loss'
                # Take profit
                elif current_price >= position['take_profit']:
                    return_pct = (position['take_profit'] - position['entry_price']) / position['entry_price']
                    exit_reason = 'take_profit'
                # ML sell signal
                elif row['signal'] == -1:
                    return_pct = (current_price - position['entry_price']) / position['entry_price']
                    exit_reason = 'ml_signal'
                else:
                    continue
                
                # Calculate profit
                profit = return_pct * position['position_value']
                capital += profit
                
                trade_result = {
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'return_pct': return_pct,
                    'profit': profit,
                    'exit_reason': exit_reason,
                    'ml_confidence': position['ml_confidence']
                }
                
                trades.append(trade_result)
                position = None
            
            # Check entry conditions
            elif row['signal'] == 1 and position is None:
                # Calculate position size based on ML confidence
                position_size = self.position_size_pct * row['position_size']
                position_value = capital * position_size
                
                position = {
                    'entry_date': date,
                    'entry_price': current_price,
                    'position_value': position_value,
                    'stop_loss': current_price * (1 - self.stop_loss_pct),
                    'take_profit': current_price * (1 + self.take_profit_pct),
                    'ml_confidence': row['ml_confidence']
                }
        
        return trades
    
    async def run(self, symbols: List[str] = None, days: int = 30) -> Dict:
        """Run ML backtest on multiple symbols"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:10]  # Default to first 10 symbols
        
        all_trades = []
        
        # Run backtests concurrently
        tasks = [self.backtest_symbol(symbol, days) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        for trades in results:
            all_trades.extend(trades)
        
        # Calculate metrics
        if not all_trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'total_trades': 0,
                'max_drawdown': 0,
                'backtest_engine': 'ml_xgboost'
            }
        
        # Calculate performance metrics
        df_trades = pd.DataFrame(all_trades)
        
        total_return = (df_trades['profit'].sum() / self.initial_capital) * 100
        winning_trades = df_trades[df_trades['profit'] > 0]
        losing_trades = df_trades[df_trades['profit'] < 0]
        
        win_rate = (len(winning_trades) / len(df_trades)) * 100 if len(df_trades) > 0 else 0
        
        # Calculate Sharpe ratio
        if len(df_trades) > 1:
            returns = df_trades.groupby('exit_date')['return_pct'].sum()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + df_trades.set_index('exit_date')['return_pct']).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdowns.min()) * 100
        
        # Average ML confidence
        avg_confidence = df_trades['ml_confidence'].mean()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': len(df_trades),
            'profitable_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'max_drawdown': max_drawdown,
            'avg_win': winning_trades['return_pct'].mean() * 100 if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['return_pct'].mean() * 100 if len(losing_trades) > 0 else 0,
            'best_trade': df_trades['return_pct'].max() * 100,
            'worst_trade': df_trades['return_pct'].min() * 100,
            'avg_ml_confidence': avg_confidence,
            'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else 0,
            'backtest_engine': 'ml_xgboost'
        }


# Convenience function
async def run_ml_xgboost_backtest(days: int = 30, symbols: List[str] = None) -> Dict:
    """Run ML XGBoost backtest"""
    backtest = MLXGBoostBacktest()
    return await backtest.run(symbols, days)


if __name__ == "__main__":
    # Test backtest
    async def test():
        result = await run_ml_xgboost_backtest(days=30, symbols=SACRED_SYMBOLS[:5])
        print("\n" + "="*50)
        print("ML XGBOOST BACKTEST RESULTS")
        print("="*50)
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    asyncio.run(test())