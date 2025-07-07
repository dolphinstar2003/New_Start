#!/usr/bin/env python3
"""
Enhanced Feature Engineering for ML Models
Improved feature creation with better quality and consistency
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """Enhanced feature engineering with improved quality control"""
    
    def __init__(self, data_dir: Path):
        """Initialize enhanced feature engineer"""
        self.data_dir = data_dir
        self.raw_data_dir = data_dir / 'raw'
        self.indicators_dir = data_dir / 'indicators'
        
        # Feature configuration
        self.lookback_periods = [5, 10, 20, 50]
        self.target_horizons = [1, 3, 5, 10]
        
        # Quality control
        self.min_data_points = 100
        self.max_nan_ratio = 0.1  # Max 10% NaN allowed
        
        logger.info("Enhanced Feature Engineer initialized")
    
    def load_symbol_data(self, symbol: str, timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """Load and validate symbol data"""
        data = {}
        
        # Load price data with validation
        if timeframe == '1d':
            price_file = self.raw_data_dir / f"{symbol}_1d_raw.csv"
        else:
            price_file = self.raw_data_dir / timeframe / f"{symbol}_{timeframe}_raw.csv"
            
        if price_file.exists():
            df = pd.read_csv(price_file)
            
            # Handle different date column names
            date_col = 'datetime' if 'datetime' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            
            # Remove timezone info to avoid conflicts
            if df[date_col].dt.tz is not None:
                df[date_col] = df[date_col].dt.tz_localize(None)
            
            df.set_index(date_col, inplace=True)
            df.sort_index(inplace=True)
            
            # Validate OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required_cols):
                data['price'] = df[required_cols]
            else:
                logger.warning(f"Missing OHLCV columns for {symbol}")
        
        # Load indicators with error handling
        try:
            indicator_files = list(self.indicators_dir.glob(f"{symbol}_{timeframe}_*.csv"))
            for file_path in indicator_files:
                try:
                    indicator_name = file_path.stem.split('_')[-1]
                    df = pd.read_csv(file_path)
                    
                    date_col = 'datetime' if 'datetime' in df.columns else 'Date'
                    df[date_col] = pd.to_datetime(df[date_col])
                    
                    if df[date_col].dt.tz is not None:
                        df[date_col] = df[date_col].dt.tz_localize(None)
                    
                    df.set_index(date_col, inplace=True)
                    df.sort_index(inplace=True)
                    data[indicator_name] = df
                    
                except Exception as e:
                    logger.warning(f"Failed to load {indicator_name} for {symbol}: {e}")
        except:
            logger.warning(f"No indicators directory found for {symbol}")
        
        return data
    
    def create_enhanced_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced price-based features"""
        features = pd.DataFrame(index=df.index)
        
        # Returns with multiple periods
        for period in [1, 2, 3, 5, 10]:
            features[f'returns_{period}d'] = df['close'].pct_change(period)
            features[f'log_returns_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price ratios
        features['hl_ratio'] = df['high'] / df['low']
        features['co_ratio'] = df['close'] / df['open']
        features['oc_ratio'] = df['open'] / df['close'].shift(1)
        
        # Body and shadow ratios (candlestick analysis)
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - np.maximum(df['close'], df['open'])
        lower_shadow = np.minimum(df['close'], df['open']) - df['low']
        total_range = df['high'] - df['low']
        
        features['body_ratio'] = body / total_range
        features['upper_shadow_ratio'] = upper_shadow / total_range
        features['lower_shadow_ratio'] = lower_shadow / total_range
        features['shadow_ratio'] = (upper_shadow + lower_shadow) / total_range
        
        # Price position within different periods
        for period in [10, 20, 50]:
            high_period = df['high'].rolling(period).max()
            low_period = df['low'].rolling(period).min()
            features[f'price_position_{period}d'] = (df['close'] - low_period) / (high_period - low_period)
        
        # Volatility features
        for period in [5, 10, 20]:
            returns = df['close'].pct_change()
            features[f'volatility_{period}d'] = returns.rolling(period).std()
            features[f'realized_vol_{period}d'] = np.sqrt(252) * features[f'volatility_{period}d']
            
            # True Range based volatility
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            features[f'atr_{period}d'] = true_range.rolling(period).mean()
        
        # Volume features
        features['volume_sma_5_ratio'] = df['volume'] / df['volume'].rolling(5).mean()
        features['volume_sma_20_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_price_trend'] = df['volume'] * np.sign(df['close'].pct_change())
        features['price_volume_product'] = df['close'] * df['volume']
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
            features[f'roc_{period}d'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Moving average features
        for period in [5, 10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'close_sma_{period}_ratio'] = df['close'] / sma
            features[f'sma_{period}_slope'] = (sma - sma.shift(5)) / sma.shift(5)
            
            # Distance from moving average
            features[f'close_sma_{period}_distance'] = (df['close'] - sma) / sma
        
        # Gap analysis
        features['gap_up'] = ((df['open'] > df['close'].shift(1)) & 
                             (df['open'] - df['close'].shift(1)) / df['close'].shift(1) > 0.01).astype(int)
        features['gap_down'] = ((df['open'] < df['close'].shift(1)) & 
                               (df['close'].shift(1) - df['open']) / df['close'].shift(1) > 0.01).astype(int)
        features['gap_size'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        return features
    
    def create_technical_features_enhanced(self, indicators_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Enhanced technical indicator features"""
        # Find common date range
        if not indicators_data:
            return pd.DataFrame()
        
        # Get intersection of all indices
        common_index = None
        for df in indicators_data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=common_index)
        
        # Enhanced SuperTrend features
        if 'supertrend' in indicators_data:
            st = indicators_data['supertrend'].reindex(common_index)
            features['st_trend'] = st['trend'] if 'trend' in st.columns else 0
            features['st_buy'] = st['buy_signal'].astype(int) if 'buy_signal' in st.columns else 0
            features['st_sell'] = st['sell_signal'].astype(int) if 'sell_signal' in st.columns else 0
            
            # SuperTrend momentum
            if 'trend' in st.columns:
                trend_changes = st['trend'].diff()
                features['st_trend_change'] = trend_changes.fillna(0)
                features['st_bullish_days'] = (st['trend'] == 1).rolling(10).sum()
        
        # Enhanced MACD features
        if 'macd_custom' in indicators_data:
            macd = indicators_data['macd_custom'].reindex(common_index)
            
            for col in ['macd', 'signal', 'histogram']:
                if col in macd.columns:
                    features[f'macd_{col}'] = macd[col]
            
            # MACD momentum and crossovers
            if 'macd' in macd.columns and 'signal' in macd.columns:
                features['macd_above_signal'] = (macd['macd'] > macd['signal']).astype(int)
                features['macd_cross_up'] = ((macd['macd'] > macd['signal']) & 
                                           (macd['macd'].shift(1) <= macd['signal'].shift(1))).astype(int)
                features['macd_cross_down'] = ((macd['macd'] < macd['signal']) & 
                                             (macd['macd'].shift(1) >= macd['signal'].shift(1))).astype(int)
            
            if 'histogram' in macd.columns:
                features['macd_hist_increasing'] = (macd['histogram'] > macd['histogram'].shift(1)).astype(int)
                features['macd_hist_positive'] = (macd['histogram'] > 0).astype(int)
        
        # WaveTrend features
        if 'wavetrend' in indicators_data:
            wt = indicators_data['wavetrend'].reindex(common_index)
            
            for col in ['wt1', 'wt2']:
                if col in wt.columns:
                    features[f'wt_{col}'] = wt[col]
                    features[f'wt_{col}_overbought'] = (wt[col] > 60).astype(int)
                    features[f'wt_{col}_oversold'] = (wt[col] < -60).astype(int)
            
            # WaveTrend signals
            signal_cols = ['buy_signal', 'sell_signal', 'cross_up', 'cross_down']
            for col in signal_cols:
                if col in wt.columns:
                    features[f'wt_{col}'] = wt[col].astype(int)
        
        # ADX/DI features
        if 'adx_di' in indicators_data:
            adx = indicators_data['adx_di'].reindex(common_index)
            
            for col in ['adx', 'plus_di', 'minus_di']:
                if col in adx.columns:
                    features[f'adx_{col}'] = adx[col]
            
            if 'plus_di' in adx.columns and 'minus_di' in adx.columns:
                features['di_diff'] = adx['plus_di'] - adx['minus_di']
                features['di_ratio'] = adx['plus_di'] / (adx['minus_di'] + 0.001)
                features['di_sum'] = adx['plus_di'] + adx['minus_di']
            
            if 'adx' in adx.columns:
                features['adx_strong'] = (adx['adx'] > 25).astype(int)
                features['adx_very_strong'] = (adx['adx'] > 40).astype(int)
                features['adx_increasing'] = (adx['adx'] > adx['adx'].shift(1)).astype(int)
        
        # Squeeze Momentum features
        if 'squeeze_momentum' in indicators_data:
            sq = indicators_data['squeeze_momentum'].reindex(common_index)
            
            signal_cols = ['squeeze_on', 'squeeze_off', 'squeeze_release']
            for col in signal_cols:
                if col in sq.columns:
                    features[f'sq_{col}'] = sq[col].astype(int)
            
            if 'momentum' in sq.columns:
                features['sq_momentum'] = sq['momentum']
                features['sq_momentum_positive'] = (sq['momentum'] > 0).astype(int)
                features['sq_momentum_increasing'] = (sq['momentum'] > sq['momentum'].shift(1)).astype(int)
        
        return features
    
    def create_advanced_features(self, price_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced derived features"""
        advanced = pd.DataFrame(index=features_df.index)
        
        # Trend strength
        if 'st_trend' in features_df.columns:
            advanced['trend_consistency_5d'] = features_df['st_trend'].rolling(5).mean()
            advanced['trend_consistency_10d'] = features_df['st_trend'].rolling(10).mean()
        
        # Volatility regime
        if 'volatility_20d' in features_df.columns:
            vol_median = features_df['volatility_20d'].rolling(100).median()
            advanced['vol_regime'] = (features_df['volatility_20d'] / vol_median).fillna(1)
            advanced['low_vol_regime'] = (advanced['vol_regime'] < 0.8).astype(int)
            advanced['high_vol_regime'] = (advanced['vol_regime'] > 1.2).astype(int)
        
        # Momentum clusters
        momentum_cols = [col for col in features_df.columns if 'momentum' in col or 'roc' in col]
        if momentum_cols:
            momentum_df = features_df[momentum_cols]
            advanced['momentum_score'] = momentum_df.mean(axis=1)
            advanced['momentum_consensus'] = (momentum_df > 0).sum(axis=1) / len(momentum_cols)
        
        # Price acceleration
        if 'returns_1d' in features_df.columns:
            returns = features_df['returns_1d']
            advanced['price_acceleration'] = returns - returns.shift(1)
            advanced['returns_consistency_5d'] = returns.rolling(5).apply(lambda x: (x > 0).sum()) / 5
        
        return advanced
    
    def build_enhanced_feature_matrix(self, symbol: str, timeframe: str = '1d') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build enhanced feature matrix with quality control"""
        logger.info(f"Building enhanced feature matrix for {symbol}")
        
        # Load data
        data = self.load_symbol_data(symbol, timeframe)
        
        if 'price' not in data or data['price'].empty:
            logger.warning(f"No price data available for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
        
        price_df = data['price']
        
        # Check minimum data requirement
        if len(price_df) < self.min_data_points:
            logger.warning(f"Insufficient data for {symbol}: {len(price_df)} < {self.min_data_points}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Create feature components
        price_features = self.create_enhanced_price_features(price_df)
        tech_features = self.create_technical_features_enhanced({k: v for k, v in data.items() if k != 'price'})
        time_features = self.create_time_features(price_df)
        
        # Combine features
        all_features = [price_features]
        
        if not tech_features.empty:
            all_features.append(tech_features)
        if not time_features.empty:
            all_features.append(time_features)
        
        # Merge on common index
        if len(all_features) == 1:
            features_df = all_features[0]
        else:
            features_df = all_features[0]
            for feat_df in all_features[1:]:
                features_df = features_df.join(feat_df, how='inner')
        
        # Add advanced features
        if not features_df.empty:
            advanced_features = self.create_advanced_features(price_df, features_df)
            if not advanced_features.empty:
                features_df = features_df.join(advanced_features, how='inner')
        
        # Quality control
        features_df = self.quality_control(features_df)
        
        # Create targets
        targets_df = self.create_targets(price_df, features_df.index)
        
        # Final alignment
        common_index = features_df.index.intersection(targets_df.index)
        features_df = features_df.loc[common_index]
        targets_df = targets_df.loc[common_index]
        
        logger.info(f"Enhanced feature matrix shape: {features_df.shape}, Targets shape: {targets_df.shape}")
        
        return features_df, targets_df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame(index=df.index)
        
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        features['is_month_end'] = df.index.is_month_end.astype(int)
        features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        features['week_of_year'] = df.index.isocalendar().week
        
        return features
    
    def create_targets(self, price_df: pd.DataFrame, feature_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create prediction targets"""
        targets = pd.DataFrame(index=feature_index)
        
        # Align price data with feature index
        aligned_prices = price_df['close'].reindex(feature_index, method='ffill')
        
        # Forward returns for different horizons
        for horizon in self.target_horizons:
            returns = aligned_prices.pct_change(horizon).shift(-horizon)
            targets[f'returns_{horizon}d'] = returns
            
            # Classification targets
            targets[f'signal_{horizon}d'] = np.where(
                returns > 0.02, 2,    # Strong buy
                np.where(returns > 0.005, 1,  # Buy
                np.where(returns < -0.02, -1,  # Sell
                0)))  # Hold
        
        return targets
    
    def quality_control(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control to features"""
        # Remove features with too many NaNs
        nan_ratio = features_df.isnull().sum() / len(features_df)
        valid_features = nan_ratio[nan_ratio <= self.max_nan_ratio].index
        features_df = features_df[valid_features]
        
        # Fill remaining NaNs
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Remove constant features
        constant_features = features_df.columns[features_df.var() == 0]
        if len(constant_features) > 0:
            features_df = features_df.drop(columns=constant_features)
            logger.info(f"Removed {len(constant_features)} constant features")
        
        return features_df