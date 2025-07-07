"""
DL-based Backtest using LSTM
Predict price movements using trained LSTM model
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
import asyncio
import tensorflow as tf

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
# LSTM model is loaded from h5 file
from sklearn.preprocessing import MinMaxScaler


class DLLSTMBacktest:
    """Deep Learning backtest using LSTM predictions"""
    
    def __init__(self):
        self.initial_capital = 100000
        self.position_size_pct = 0.1
        self.stop_loss_pct = 0.025  # Tighter stop loss for LSTM
        self.take_profit_pct = 0.05
        self.commission = 0.001
        
        # DL specific parameters
        self.sequence_length = 20  # Look back 20 periods
        self.prediction_threshold = 0.55  # Lower threshold for LSTM
        self.model = None
        self.model_path = Path(__file__).parent.parent / 'ml_models' / 'saved_models' / 'lstm_model.h5'
        self.scaler = MinMaxScaler()
        
        # Risk management
        self.max_drawdown_limit = 0.12
        self.max_positions = 3  # Fewer positions for LSTM
        self.use_ensemble = True  # Combine with indicators
        
        logger.info("DLLSTMBacktest initialized")
    
    def load_model(self):
        """Load pre-trained LSTM model"""
        if self.model_path.exists():
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"LSTM model loaded from {self.model_path}")
        else:
            logger.warning(f"Model not found at {self.model_path}, training new model...")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new LSTM model if none exists"""
        from ml_models.train_models import train_lstm_model
        
        # Train on selected symbols
        logger.info("Training new LSTM model...")
        self.model = train_lstm_model(SACRED_SYMBOLS[:5])  # Train on first 5 symbols
        
        # Save model
        self.model_path.parent.mkdir(exist_ok=True, parents=True)
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def prepare_sequences(self, symbol: str, timeframe: str = '4h') -> Tuple[np.ndarray, pd.DataFrame]:
        """Prepare sequences for LSTM model"""
        # Load raw data
        raw_path = DATA_DIR / 'raw' / timeframe / f"{symbol}_{timeframe}_raw.csv"
        if not raw_path.exists():
            logger.warning(f"Raw data not found: {raw_path}")
            return np.array([]), pd.DataFrame()
        
        df = pd.read_csv(raw_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Calculate technical features
        df['returns'] = df['close'].pct_change()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Load indicators
        indicators_df = self._load_indicators_for_lstm(symbol, timeframe)
        if not indicators_df.empty:
            df = df.join(indicators_df, how='left')
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < self.sequence_length + 1:
            return np.array([]), pd.DataFrame()
        
        # Prepare features for LSTM
        feature_cols = [
            'close', 'volume', 'returns', 'volume_ratio', 'price_range', 'close_position'
        ]
        
        # Add indicator features if available
        indicator_cols = [col for col in df.columns if any(ind in col for ind in ['trend', 'adx', 'momentum', 'wt1', 'macd'])]
        feature_cols.extend(indicator_cols)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, dates = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            dates.append(df.index[i])
        
        X = np.array(X)
        
        # Create DataFrame with dates and prices
        result_df = pd.DataFrame(index=dates)
        result_df['close'] = df['close'].iloc[self.sequence_length:].values
        
        # Add original features for ensemble
        if self.use_ensemble:
            for col in ['trend', 'adx', 'squeeze_on', 'momentum']:
                if col in df.columns:
                    result_df[col] = df[col].iloc[self.sequence_length:].values
        
        return X, result_df
    
    def _load_indicators_for_lstm(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load key indicators for LSTM"""
        indicators_dir = DATA_DIR / 'indicators' / timeframe
        result = pd.DataFrame()
        
        # Load only most important indicators
        indicators = {
            'supertrend': ['trend'],
            'adx_di': ['adx'],
            'squeeze_momentum': ['squeeze_on', 'momentum'],
            'wavetrend': ['wt1'],
            'macd_custom': ['macd']
        }
        
        for indicator, cols in indicators.items():
            filepath = indicators_dir / f"{symbol}_{timeframe}_{indicator}.csv"
            if filepath.exists():
                df = pd.read_csv(filepath)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                for col in cols:
                    if col in df.columns:
                        result[col] = df[col]
        
        return result
    
    def generate_lstm_signals(self, X: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using LSTM model"""
        if self.model is None:
            self.load_model()
        
        if len(X) == 0:
            return pd.DataFrame()
        
        # Make predictions
        try:
            predictions = self.model.predict(X, batch_size=32)
            
            # Create signals DataFrame
            signals = pd.DataFrame(index=df.index)
            signals['lstm_prediction'] = predictions.flatten()
            signals['close'] = df['close']
            signals['signal'] = 0
            
            # Basic LSTM signals
            signals.loc[predictions.flatten() > self.prediction_threshold, 'signal'] = 1
            signals.loc[predictions.flatten() < (1 - self.prediction_threshold), 'signal'] = -1
            
            # Ensemble with indicators if available
            if self.use_ensemble and 'trend' in df.columns:
                # Require agreement with Supertrend
                lstm_buy = signals['signal'] == 1
                lstm_sell = signals['signal'] == -1
                
                trend_buy = df['trend'] == 1
                trend_sell = df['trend'] == -1
                
                # Reset signals
                signals['signal'] = 0
                
                # Strong buy: LSTM + Supertrend agree
                signals.loc[lstm_buy & trend_buy, 'signal'] = 1
                signals.loc[lstm_buy & trend_buy, 'signal_strength'] = 'strong'
                
                # Weak buy: Only LSTM
                signals.loc[lstm_buy & ~trend_buy, 'signal'] = 1
                signals.loc[lstm_buy & ~trend_buy, 'signal_strength'] = 'weak'
                
                # Sell signals
                signals.loc[lstm_sell | trend_sell, 'signal'] = -1
                
                # Add ADX filter if available
                if 'adx' in df.columns:
                    low_adx = df['adx'] < 20  # Weak trend
                    signals.loc[low_adx & (signals['signal'] == 1), 'signal'] = 0
            
            # Position sizing based on prediction confidence
            signals['position_size'] = np.where(
                signals['signal'] != 0,
                np.clip(np.abs(predictions.flatten() - 0.5) * 2, 0.3, 1.0),
                0
            )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating LSTM signals: {e}")
            return pd.DataFrame()
    
    async def backtest_symbol(self, symbol: str, days: int) -> List[Dict]:
        """Run LSTM backtest for single symbol"""
        # Prepare sequences
        X, df = self.prepare_sequences(symbol, '4h')  # Use 4h for LSTM
        
        if len(X) == 0:
            return []
        
        # Generate signals
        signals = self.generate_lstm_signals(X, df)
        
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
        high_water_mark = 0
        
        for date, row in signals.iterrows():
            current_price = row['close']
            
            # Update high water mark for trailing stop
            if position:
                high_water_mark = max(high_water_mark, current_price)
            
            # Check exit conditions
            if position:
                # Dynamic stop loss based on signal strength
                stop_mult = 1.5 if position.get('signal_strength') == 'weak' else 1.0
                
                # Stop loss
                if current_price <= position['stop_loss'] * stop_mult:
                    return_pct = (current_price - position['entry_price']) / position['entry_price']
                    exit_reason = 'stop_loss'
                # Take profit
                elif current_price >= position['take_profit']:
                    return_pct = (position['take_profit'] - position['entry_price']) / position['entry_price']
                    exit_reason = 'take_profit'
                # Trailing stop (only for strong signals)
                elif position.get('signal_strength') == 'strong' and current_price < high_water_mark * 0.97:
                    return_pct = (current_price - position['entry_price']) / position['entry_price']
                    exit_reason = 'trailing_stop'
                # LSTM sell signal
                elif row['signal'] == -1:
                    return_pct = (current_price - position['entry_price']) / position['entry_price']
                    exit_reason = 'lstm_signal'
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
                    'lstm_prediction': position['lstm_prediction'],
                    'signal_strength': position.get('signal_strength', 'normal')
                }
                
                trades.append(trade_result)
                position = None
                high_water_mark = 0
            
            # Check entry conditions
            elif row['signal'] == 1 and position is None:
                # Calculate position size
                position_size = self.position_size_pct * row['position_size']
                position_value = capital * position_size
                
                position = {
                    'entry_date': date,
                    'entry_price': current_price,
                    'position_value': position_value,
                    'stop_loss': current_price * (1 - self.stop_loss_pct),
                    'take_profit': current_price * (1 + self.take_profit_pct),
                    'lstm_prediction': row['lstm_prediction'],
                    'signal_strength': row.get('signal_strength', 'normal')
                }
                
                high_water_mark = current_price
        
        return trades
    
    async def run(self, symbols: List[str] = None, days: int = 30) -> Dict:
        """Run LSTM backtest on multiple symbols"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:5]  # Default to first 5 symbols
        
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
                'backtest_engine': 'dl_lstm'
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
        
        # Signal strength analysis
        strong_trades = df_trades[df_trades['signal_strength'] == 'strong']
        weak_trades = df_trades[df_trades['signal_strength'] == 'weak']
        
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
            'avg_lstm_prediction': df_trades['lstm_prediction'].mean(),
            'strong_signal_win_rate': (len(strong_trades[strong_trades['profit'] > 0]) / len(strong_trades) * 100) if len(strong_trades) > 0 else 0,
            'weak_signal_win_rate': (len(weak_trades[weak_trades['profit'] > 0]) / len(weak_trades) * 100) if len(weak_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else 0,
            'backtest_engine': 'dl_lstm'
        }


# Convenience function
async def run_dl_lstm_backtest(days: int = 30, symbols: List[str] = None) -> Dict:
    """Run DL LSTM backtest"""
    backtest = DLLSTMBacktest()
    return await backtest.run(symbols, days)


if __name__ == "__main__":
    # Test backtest
    async def test():
        result = await run_dl_lstm_backtest(days=30, symbols=SACRED_SYMBOLS[:3])
        print("\n" + "="*50)
        print("DL LSTM BACKTEST RESULTS")
        print("="*50)
        for key, value in result.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    asyncio.run(test())