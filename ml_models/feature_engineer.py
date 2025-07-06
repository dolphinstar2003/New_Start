"""
Feature Engineering for ML Models
Create features from price data and indicators
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger


class FeatureEngineer:
    """Feature engineering for trading ML models"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize feature engineer
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir
        self.raw_data_dir = data_dir / 'raw'
        self.indicators_dir = data_dir / 'indicators'
        
        # Feature configuration
        self.lookback_periods = [5, 10, 20, 50]
        self.target_horizons = [1, 3, 5]  # Days ahead for prediction
        
        logger.info("FeatureEngineer initialized")
    
    def load_symbol_data(self, symbol: str, timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Load all data for a symbol (price + indicators)
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            
        Returns:
            Dictionary with price data and indicators
        """
        data = {}
        
        # Load price data
        price_file = self.raw_data_dir / f"{symbol}_{timeframe}_raw.csv"
        if price_file.exists():
            df = pd.read_csv(price_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            data['price'] = df
        
        # Load indicators
        indicator_files = self.indicators_dir.glob(f"{symbol}_{timeframe}_*.csv")
        for file_path in indicator_files:
            indicator_name = file_path.stem.split('_')[-1]
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            data[indicator_name] = df
        
        return data
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features
        
        Args:
            df: Price DataFrame with OHLCV
            
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Volume features
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['price_volume'] = df['close'] * df['volume']
        
        # Volatility features
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
            features[f'high_low_range_{period}'] = (df['high'] - df['low']).rolling(period).mean()
        
        # Price position features
        for period in [10, 20, 50]:
            high_period = df['high'].rolling(period).max()
            low_period = df['low'].rolling(period).min()
            features[f'price_position_{period}'] = (df['close'] - low_period) / (high_period - low_period)
        
        # Moving averages ratios
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'close_sma_{period}_ratio'] = df['close'] / sma
            features[f'sma_{period}_slope'] = sma.diff(5) / sma.shift(5)
        
        # Gap features
        features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        features['gap_filled'] = (
            ((df['open'] > df['close'].shift(1)) & (df['low'] <= df['close'].shift(1))) |
            ((df['open'] < df['close'].shift(1)) & (df['high'] >= df['close'].shift(1)))
        ).astype(int)
        
        return features
    
    def create_technical_features(self, indicators_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create features from technical indicators
        
        Args:
            indicators_data: Dictionary of indicator DataFrames
            
        Returns:
            Combined technical features DataFrame
        """
        # Get common index from all indicators
        indices = [df.index for df in indicators_data.values()]
        common_index = indices[0]
        for idx in indices[1:]:
            common_index = common_index.intersection(idx)
        
        features = pd.DataFrame(index=common_index)
        
        # Supertrend features
        if 'supertrend' in indicators_data:
            st = indicators_data['supertrend']
            features['supertrend_signal'] = st['trend'].reindex(common_index)
            features['supertrend_distance'] = (st['supertrend'] - st.index.to_series().map(
                lambda x: indicators_data.get('price', pd.DataFrame()).loc[x, 'close'] if x in indicators_data.get('price', pd.DataFrame()).index else np.nan
            )).reindex(common_index)
            features['supertrend_buy'] = st['buy_signal'].astype(int).reindex(common_index)
            features['supertrend_sell'] = st['sell_signal'].astype(int).reindex(common_index)
        
        # ADX/DI features
        if 'adx_di' in indicators_data:
            adx = indicators_data['adx_di']
            features['adx'] = adx['adx'].reindex(common_index)
            features['plus_di'] = adx['plus_di'].reindex(common_index)
            features['minus_di'] = adx['minus_di'].reindex(common_index)
            features['di_diff'] = (adx['plus_di'] - adx['minus_di']).reindex(common_index)
            features['adx_strong'] = (adx['adx'] > 25).astype(int).reindex(common_index)
            features['di_bullish'] = adx['di_bullish_cross'].astype(int).reindex(common_index)
            features['di_bearish'] = adx['di_bearish_cross'].astype(int).reindex(common_index)
        
        # Squeeze Momentum features
        if 'squeeze_momentum' in indicators_data:
            sq = indicators_data['squeeze_momentum']
            features['squeeze_on'] = sq['squeeze_on'].astype(int).reindex(common_index)
            features['squeeze_off'] = sq['squeeze_off'].astype(int).reindex(common_index)
            features['momentum'] = sq['momentum'].reindex(common_index)
            features['momentum_positive'] = (sq['momentum'] > 0).astype(int).reindex(common_index)
            features['squeeze_release'] = sq['squeeze_release'].astype(int).reindex(common_index)
        
        # WaveTrend features
        if 'wavetrend' in indicators_data:
            wt = indicators_data['wavetrend']
            features['wt1'] = wt['wt1'].reindex(common_index)
            features['wt2'] = wt['wt2'].reindex(common_index)
            features['wt_diff'] = wt['wt_diff'].reindex(common_index)
            features['wt_overbought'] = wt['overbought'].astype(int).reindex(common_index)
            features['wt_oversold'] = wt['oversold'].astype(int).reindex(common_index)
            features['wt_cross_up'] = wt['cross_up'].astype(int).reindex(common_index)
            features['wt_cross_down'] = wt['cross_down'].astype(int).reindex(common_index)
            features['wt_buy_signal'] = wt['buy_signal'].astype(int).reindex(common_index)
            features['wt_sell_signal'] = wt['sell_signal'].astype(int).reindex(common_index)
        
        # MACD features
        if 'macd_custom' in indicators_data:
            macd = indicators_data['macd_custom']
            features['macd'] = macd['macd'].reindex(common_index)
            features['macd_signal'] = macd['signal'].reindex(common_index)
            features['macd_histogram'] = macd['histogram'].reindex(common_index)
            features['macd_bullish'] = macd['bullish_momentum'].astype(int).reindex(common_index)
            features['macd_bearish'] = macd['bearish_momentum'].astype(int).reindex(common_index)
            features['macd_cross_up'] = macd['macd_cross_up'].astype(int).reindex(common_index)
            features['macd_cross_down'] = macd['macd_cross_down'].astype(int).reindex(common_index)
            features['macd_above_zero'] = macd['macd_above_zero'].astype(int).reindex(common_index)
        
        return features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with time features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic time features
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        features['is_month_end'] = df.index.is_month_end.astype(int)
        features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Cyclical encoding
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
        features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
        
        return features
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features
        
        Args:
            df: Source DataFrame
            columns: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    features[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return features
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features
        
        Args:
            df: Source DataFrame
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=df.index)
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    features[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                    features[f'{col}_std_{window}'] = df[col].rolling(window).std()
                    features[f'{col}_min_{window}'] = df[col].rolling(window).min()
                    features[f'{col}_max_{window}'] = df[col].rolling(window).max()
                    features[f'{col}_skew_{window}'] = df[col].rolling(window).skew()
        
        return features
    
    def create_targets(self, df: pd.DataFrame, horizons: List[int] = None) -> pd.DataFrame:
        """
        Create prediction targets
        
        Args:
            df: Price DataFrame
            horizons: Forward looking periods
            
        Returns:
            DataFrame with target variables
        """
        if horizons is None:
            horizons = self.target_horizons
        
        targets = pd.DataFrame(index=df.index)
        
        for horizon in horizons:
            # Future returns
            future_close = df['close'].shift(-horizon)
            targets[f'return_{horizon}d'] = (future_close - df['close']) / df['close']
            
            # Binary classification targets
            targets[f'up_{horizon}d'] = (targets[f'return_{horizon}d'] > 0).astype(int)
            targets[f'strong_up_{horizon}d'] = (targets[f'return_{horizon}d'] > 0.02).astype(int)
            targets[f'strong_down_{horizon}d'] = (targets[f'return_{horizon}d'] < -0.02).astype(int)
            
            # Multi-class targets
            conditions = [
                targets[f'return_{horizon}d'] > 0.02,
                targets[f'return_{horizon}d'] > 0,
                targets[f'return_{horizon}d'] > -0.02,
            ]
            choices = [2, 1, 0]  # Strong Up, Up, Hold, Down
            targets[f'signal_{horizon}d'] = np.select(conditions, choices, default=-1)
        
        return targets
    
    def build_feature_matrix(self, symbol: str, timeframe: str = '1d') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build complete feature matrix for a symbol
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            
        Returns:
            Tuple of (features, targets)
        """
        logger.info(f"Building feature matrix for {symbol}")
        
        # Load all data
        data = self.load_symbol_data(symbol, timeframe)
        
        if 'price' not in data:
            logger.error(f"No price data found for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
        
        price_df = data['price']
        indicators_data = {k: v for k, v in data.items() if k != 'price'}
        
        # Create feature components
        price_features = self.create_price_features(price_df)
        tech_features = self.create_technical_features(indicators_data)
        time_features = self.create_time_features(price_df)
        
        # Create lag features for important columns
        important_cols = ['returns', 'volume_sma_ratio', 'volatility_20']
        lag_features = self.create_lag_features(price_features, important_cols, [1, 2, 3, 5])
        
        # Create rolling features
        rolling_cols = ['returns', 'volatility_5']
        rolling_features = self.create_rolling_features(price_features, rolling_cols, [5, 10])
        
        # Combine all features
        all_features = [price_features, tech_features, time_features, lag_features, rolling_features]
        features = pd.concat(all_features, axis=1)
        
        # Create targets
        targets = self.create_targets(price_df)
        
        # Align indices
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index]
        
        # Remove rows with too many NaN values
        features = features.dropna(thresh=len(features.columns) * 0.7)
        targets = targets.loc[features.index]
        
        logger.info(f"Feature matrix shape: {features.shape}, Targets shape: {targets.shape}")
        
        return features, targets