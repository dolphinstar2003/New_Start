"""
Backtest Engine for Trading System
Comprehensive backtesting with realistic trading conditions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))

from strategies.signal_generator import TradingSignalGenerator
from trading.risk_manager import RiskManager
from portfolio.portfolio_manager import PortfolioManager
from ml_models.feature_engineer import FeatureEngineer
from ml_models.ensemble_model import EnsembleTradingModel
from config.settings import SACRED_SYMBOLS, DATA_DIR, RISK_MANAGEMENT


class BacktestEngine:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, start_date: str, end_date: str, 
                 initial_capital: float = 100000.0,
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_rate: float = 0.0005):   # 0.05% slippage
        """
        Initialize backtest engine
        
        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Initial capital
            commission_rate: Commission rate per trade
            slippage_rate: Slippage rate per trade
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(DATA_DIR)
        self.ensemble_model = EnsembleTradingModel(DATA_DIR.parent / 'ml_models')
        self.signal_generator = TradingSignalGenerator(DATA_DIR, DATA_DIR.parent / 'ml_models')
        self.risk_manager = RiskManager(initial_capital)
        self.portfolio_manager = PortfolioManager(initial_capital)
        
        # Backtest state
        self.current_date = self.start_date
        self.is_trained = False
        self.market_data = {}
        self.trading_calendar = []
        
        # Results tracking
        self.trade_log = []
        self.daily_results = []
        self.performance_metrics = {}
        
        logger.info(f"Backtest Engine initialized: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}, Commission: {commission_rate*100:.3f}%, Slippage: {slippage_rate*100:.3f}%")
    
    def load_market_data(self) -> None:
        """Load market data for all symbols"""
        logger.info("Loading market data for backtest...")
        
        for symbol in SACRED_SYMBOLS:
            try:
                # Load price data
                price_file = DATA_DIR / 'raw' / f"{symbol}_1d_raw.csv"
                if price_file.exists():
                    df = pd.read_csv(price_file)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    
                    # Convert timezone-aware index to timezone-naive for comparison
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Filter by date range
                    mask = (df.index >= self.start_date) & (df.index <= self.end_date)
                    df = df[mask]
                    
                    if not df.empty:
                        self.market_data[symbol] = df
                        logger.debug(f"Loaded {len(df)} days of data for {symbol}")
                    else:
                        logger.warning(f"No data in date range for {symbol}")
                else:
                    logger.warning(f"Data file not found for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
        
        if self.market_data:
            # Create trading calendar from available dates
            all_dates = set()
            for df in self.market_data.values():
                all_dates.update(df.index)
            
            self.trading_calendar = sorted(list(all_dates))
            logger.info(f"Market data loaded for {len(self.market_data)} symbols")
            logger.info(f"Trading calendar: {len(self.trading_calendar)} days")
        else:
            raise ValueError("No market data available for backtest")
    
    def prepare_training_data(self, train_end_date: str) -> None:
        """
        Prepare training data for ML models
        
        Args:
            train_end_date: End date for training data
        """
        logger.info("Preparing training data for ML models...")
        
        train_end = pd.to_datetime(train_end_date)
        training_features = []
        training_targets = []
        
        for symbol in SACRED_SYMBOLS:
            try:
                # Build feature matrix for training period
                features, targets = self.feature_engineer.build_feature_matrix(symbol)
                
                if not features.empty and not targets.empty:
                    # Filter training period
                    train_mask = features.index <= train_end
                    symbol_features = features[train_mask]
                    symbol_targets = targets[train_mask]
                    
                    if len(symbol_features) > 50:  # Minimum data requirement
                        # Use 1-day ahead returns as target
                        target_col = 'signal_1d' if 'signal_1d' in symbol_targets.columns else 'up_1d'
                        
                        symbol_features['symbol'] = symbol
                        training_features.append(symbol_features)
                        training_targets.append(symbol_targets[target_col])
                        
                        logger.debug(f"Added {len(symbol_features)} training samples for {symbol}")
            
            except Exception as e:
                logger.error(f"Error preparing training data for {symbol}: {e}")
        
        if training_features:
            # Combine all features and targets
            combined_features = pd.concat(training_features, ignore_index=False)
            combined_targets = pd.concat(training_targets, ignore_index=False)
            
            # Remove symbol column for training
            if 'symbol' in combined_features.columns:
                combined_features = combined_features.drop('symbol', axis=1)
            
            # Align indices
            common_index = combined_features.index.intersection(combined_targets.index)
            combined_features = combined_features.loc[common_index]
            combined_targets = combined_targets.loc[common_index]
            
            logger.info(f"Training data prepared: {len(combined_features)} samples, {len(combined_features.columns)} features")
            
            # Train ensemble model
            training_results = self.ensemble_model.train_all_models(combined_features, combined_targets)
            
            if any('error' not in stats for stats in training_results.values()):
                self.is_trained = True
                logger.info("✅ ML models trained successfully")
            else:
                logger.error("❌ ML model training failed")
        else:
            logger.warning("No training data available")
    
    def get_current_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Get current prices for all symbols on given date"""
        current_prices = {}
        
        for symbol, df in self.market_data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']
        
        return current_prices
    
    def calculate_realistic_execution_price(self, symbol: str, date: pd.Timestamp,
                                          signal: str, shares: int) -> float:
        """
        Calculate realistic execution price including slippage and market impact
        
        Args:
            symbol: Trading symbol
            date: Trade date
            signal: Trading signal
            shares: Number of shares
            
        Returns:
            Realistic execution price
        """
        if symbol not in self.market_data or date not in self.market_data[symbol].index:
            return 0.0
        
        row = self.market_data[symbol].loc[date]
        
        # Base price (use open price for next day execution)
        next_date_idx = None
        dates = list(self.market_data[symbol].index)
        if date in dates:
            current_idx = dates.index(date)
            if current_idx + 1 < len(dates):
                next_date = dates[current_idx + 1]
                if next_date in self.market_data[symbol].index:
                    next_row = self.market_data[symbol].loc[next_date]
                    base_price = next_row['open']  # Execute at next day's open
                else:
                    base_price = row['close']
            else:
                base_price = row['close']
        else:
            base_price = row['close']
        
        # Calculate slippage based on signal direction
        if signal in ['STRONG_BUY', 'WEAK_BUY']:
            # Buy orders: slippage increases price
            slippage_factor = 1 + self.slippage_rate
        else:
            # Sell orders: slippage decreases price
            slippage_factor = 1 - self.slippage_rate
        
        # Market impact based on volume
        if 'volume' in row and row['volume'] > 0:
            # Estimate market impact based on trade size vs average volume
            avg_volume = row['volume']
            trade_volume = shares
            
            if trade_volume > avg_volume * 0.01:  # More than 1% of avg volume
                additional_impact = min(0.002, trade_volume / avg_volume * 0.1)  # Cap at 0.2%
                if signal in ['STRONG_BUY', 'WEAK_BUY']:
                    slippage_factor += additional_impact
                else:
                    slippage_factor -= additional_impact
        
        execution_price = base_price * slippage_factor
        return max(0.01, execution_price)  # Minimum price
    
    def execute_backtest_trade(self, symbol: str, signal: str, confidence: float,
                             date: pd.Timestamp) -> Optional[Dict]:
        """
        Execute a single trade in backtest
        
        Args:
            symbol: Trading symbol
            signal: Trading signal
            confidence: Signal confidence
            date: Trade date
            
        Returns:
            Trade execution details
        """
        current_prices = self.get_current_prices(date)
        if symbol not in current_prices:
            return None
        
        current_price = current_prices[symbol]
        
        # Calculate position size
        position_size_info = self.risk_manager.calculate_position_size(
            signal, confidence, current_price
        )
        
        if not position_size_info['can_trade'] or position_size_info['shares'] <= 0:
            return None
        
        # Validate trade
        validation = self.risk_manager.validate_trade(symbol, signal, position_size_info)
        if not validation['approved']:
            return None
        
        shares = position_size_info['shares']
        
        # Calculate realistic execution price
        execution_price = self.calculate_realistic_execution_price(symbol, date, signal, shares)
        
        # Calculate commission
        trade_value = shares * execution_price
        commission = trade_value * self.commission_rate
        
        # Execute trade
        if signal in ['STRONG_BUY', 'WEAK_BUY']:
            # Buy order
            total_cost = trade_value + commission
            
            if total_cost <= self.portfolio_manager.cash_balance:
                success = self.portfolio_manager.execute_trade(symbol, 'BUY', shares, execution_price, date)
                
                if success:
                    # Deduct commission
                    self.portfolio_manager.cash_balance -= commission
                    
                    trade_record = {
                        'date': date,
                        'symbol': symbol,
                        'signal': signal,
                        'action': 'BUY',
                        'shares': shares,
                        'price': execution_price,
                        'value': trade_value,
                        'commission': commission,
                        'confidence': confidence,
                        'total_cost': total_cost
                    }
                    
                    self.trade_log.append(trade_record)
                    return trade_record
        
        else:  # SELL signals
            # For now, only buy signals in backtest (long-only strategy)
            pass
        
        return None
    
    def run_single_day(self, date: pd.Timestamp) -> Dict[str, Any]:
        """
        Run backtest for a single day
        
        Args:
            date: Trading date
            
        Returns:
            Daily results
        """
        current_prices = self.get_current_prices(date)
        
        if not current_prices:
            return {'date': date, 'trades': 0, 'portfolio_value': self.portfolio_manager.current_capital}
        
        # Update portfolio valuation
        portfolio_value = self.portfolio_manager.calculate_portfolio_value(current_prices)
        
        # Update unrealized P&L
        self.portfolio_manager.cash_balance = portfolio_value - sum(
            holding['shares'] * current_prices.get(symbol, 0) 
            for symbol, holding in self.portfolio_manager.holdings.items()
        )
        
        daily_trades = 0
        
        # Generate signals if models are trained
        if self.is_trained:
            try:
                # Get signals for current date
                signals_dict = {}
                
                for symbol in SACRED_SYMBOLS:
                    if symbol in current_prices:
                        # Get recent features for signal generation
                        features, _ = self.feature_engineer.build_feature_matrix(symbol)
                        
                        if not features.empty:
                            # Filter to current date
                            feature_date_mask = features.index <= date
                            recent_features = features[feature_date_mask].tail(50)  # Last 50 days
                            
                            if len(recent_features) >= 20:  # Minimum for LSTM
                                # Generate signal
                                if self.ensemble_model.is_trained:
                                    try:
                                        ml_probabilities = self.ensemble_model.predict_proba(recent_features.tail(1))
                                        if len(ml_probabilities) > 0:
                                            proba = ml_probabilities[0]
                                            confidence = np.max(proba)
                                            
                                            # Determine signal
                                            if proba[2] > 0.7:  # Strong buy threshold
                                                signal = 'STRONG_BUY'
                                            elif proba[2] > 0.6:
                                                signal = 'WEAK_BUY'
                                            elif proba[0] > 0.7:
                                                signal = 'STRONG_SELL'
                                            elif proba[0] > 0.6:
                                                signal = 'WEAK_SELL'
                                            else:
                                                signal = 'HOLD'
                                            
                                            if signal in ['STRONG_BUY', 'WEAK_BUY']:
                                                trade_result = self.execute_backtest_trade(
                                                    symbol, signal, confidence, date
                                                )
                                                if trade_result:
                                                    daily_trades += 1
                                    
                                    except Exception as e:
                                        logger.debug(f"Signal generation failed for {symbol}: {e}")
                
            except Exception as e:
                logger.debug(f"Daily signal processing error: {e}")
        
        # Update portfolio performance
        performance = self.portfolio_manager.update_portfolio_performance(current_prices)
        
        # Check for stop losses and take profits
        stop_losses = self.risk_manager.check_stop_losses(current_prices)
        for stop_loss in stop_losses:
            symbol = stop_loss['symbol']
            price = stop_loss['current_price']
            
            # Close position
            if symbol in self.portfolio_manager.holdings:
                shares = self.portfolio_manager.holdings[symbol]['shares']
                success = self.portfolio_manager.execute_trade(symbol, 'SELL', shares, price, date)
                
                if success:
                    trade_record = {
                        'date': date,
                        'symbol': symbol,
                        'signal': 'STOP_LOSS',
                        'action': 'SELL',
                        'shares': shares,
                        'price': price,
                        'value': shares * price,
                        'commission': shares * price * self.commission_rate,
                        'confidence': 1.0,
                        'total_cost': shares * price * (1 - self.commission_rate)
                    }
                    self.trade_log.append(trade_record)
                    daily_trades += 1
        
        # Reset daily risk metrics
        self.risk_manager.reset_daily_metrics()
        
        daily_result = {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash_balance': self.portfolio_manager.cash_balance,
            'trades': daily_trades,
            'open_positions': len(self.portfolio_manager.holdings),
            'daily_return': performance.get('daily_return', 0.0) if performance else 0.0
        }
        
        self.daily_results.append(daily_result)
        return daily_result
    
    def run_backtest(self, train_split: float = 0.6) -> Dict[str, Any]:
        """
        Run complete backtest
        
        Args:
            train_split: Fraction of data to use for training
            
        Returns:
            Backtest results
        """
        logger.info("Starting backtest execution...")
        
        # Load market data
        self.load_market_data()
        
        if not self.trading_calendar:
            raise ValueError("No trading calendar available")
        
        # Determine training period
        total_days = len(self.trading_calendar)
        train_days = int(total_days * train_split)
        train_end_date = self.trading_calendar[train_days - 1]
        
        logger.info(f"Training period: {self.trading_calendar[0]} to {train_end_date}")
        logger.info(f"Trading period: {self.trading_calendar[train_days]} to {self.trading_calendar[-1]}")
        
        # Prepare and train models
        self.prepare_training_data(train_end_date.strftime('%Y-%m-%d'))
        
        # Run backtest day by day
        trading_dates = self.trading_calendar[train_days:]
        total_trading_days = len(trading_dates)
        
        logger.info(f"Running backtest for {total_trading_days} trading days...")
        
        for i, date in enumerate(trading_dates):
            if i % 50 == 0:
                logger.info(f"Processing day {i+1}/{total_trading_days}: {date.strftime('%Y-%m-%d')}")
            
            try:
                daily_result = self.run_single_day(date)
            except Exception as e:
                logger.error(f"Error processing {date}: {e}")
                continue
        
        # Calculate final performance metrics
        self.performance_metrics = self.calculate_backtest_performance()
        
        logger.info("✅ Backtest completed successfully")
        logger.info(f"Total trades: {len(self.trade_log)}")
        logger.info(f"Final portfolio value: ${self.performance_metrics['final_value']:,.2f}")
        logger.info(f"Total return: {self.performance_metrics['total_return']*100:.2f}%")
        
        return self.performance_metrics
    
    def calculate_backtest_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest performance metrics"""
        if not self.daily_results:
            return {}
        
        daily_df = pd.DataFrame(self.daily_results)
        trades_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        
        # Basic performance
        initial_value = self.initial_capital
        final_value = daily_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        daily_df['daily_return'] = daily_df['portfolio_value'].pct_change().fillna(0)
        
        # Risk metrics
        returns = daily_df['daily_return'].values
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + daily_df['daily_return']).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # Trading metrics
        if not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            total_trades = len(trades_df)
            total_commission = trades_df['commission'].sum()
            avg_trade_size = trades_df['value'].mean()
        else:
            total_trades = 0
            total_commission = 0
            avg_trade_size = 0
        
        # Win rate (for closed positions)
        win_rate = 0.0
        profit_factor = 0.0
        
        performance = {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_percent': abs(max_drawdown) * 100,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'avg_trade_size': avg_trade_size,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trading_days': len(daily_df),
            'avg_daily_trades': total_trades / len(daily_df) if len(daily_df) > 0 else 0
        }
        
        # God Mode targets comparison
        performance['targets_met'] = {
            'win_rate_target': win_rate >= RISK_MANAGEMENT['target_win_rate'],
            'sharpe_target': sharpe_ratio >= RISK_MANAGEMENT['target_sharpe_ratio'],
            'drawdown_target': abs(max_drawdown) <= RISK_MANAGEMENT['max_drawdown_percent'] / 100,
            'monthly_target': (1 + total_return) ** (1/12) - 1 >= RISK_MANAGEMENT['monthly_target_percent'] / 100
        }
        
        return performance
    
    def save_backtest_results(self, filename: str = None) -> Path:
        """Save backtest results to files"""
        if filename is None:
            filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results_dir = DATA_DIR.parent / 'backtest_results'
        results_dir.mkdir(exist_ok=True)
        
        # Save daily results
        if self.daily_results:
            daily_df = pd.DataFrame(self.daily_results)
            daily_path = results_dir / f"{filename}_daily.csv"
            daily_df.to_csv(daily_path, index=False)
        
        # Save trade log
        if self.trade_log:
            trades_df = pd.DataFrame(self.trade_log)
            trades_path = results_dir / f"{filename}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
        
        # Save performance metrics
        if self.performance_metrics:
            import json
            metrics_path = results_dir / f"{filename}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
        
        logger.info(f"Backtest results saved to {results_dir}")
        return results_dir
    
    def generate_backtest_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.performance_metrics:
            return "No backtest results available"
        
        perf = self.performance_metrics
        
        report = f"""
BACKTEST PERFORMANCE REPORT
{'='*60}
Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE SUMMARY
{'='*30}
Initial Capital: ${perf['initial_value']:,.2f}
Final Value: ${perf['final_value']:,.2f}
Total Return: {perf['total_return_percent']:.2f}%
Annualized Return: {perf['annualized_return']*100:.2f}%

RISK METRICS
{'='*30}
Volatility (Annual): {perf['volatility']*100:.2f}%
Sharpe Ratio: {perf['sharpe_ratio']:.2f}
Maximum Drawdown: {perf['max_drawdown_percent']:.2f}%

TRADING STATISTICS
{'='*30}
Total Trades: {perf['total_trades']}
Trading Days: {perf['trading_days']}
Average Daily Trades: {perf['avg_daily_trades']:.2f}
Total Commission: ${perf['total_commission']:,.2f}
Average Trade Size: ${perf['avg_trade_size']:,.2f}

GOD MODE TARGETS
{'='*30}
Win Rate Target ({RISK_MANAGEMENT['target_win_rate']*100:.0f}%): {'✅' if perf['targets_met']['win_rate_target'] else '❌'}
Sharpe Target ({RISK_MANAGEMENT['target_sharpe_ratio']:.1f}): {'✅' if perf['targets_met']['sharpe_target'] else '❌'}
Drawdown Target (<{RISK_MANAGEMENT['max_drawdown_percent']}%): {'✅' if perf['targets_met']['drawdown_target'] else '❌'}
Monthly Target ({RISK_MANAGEMENT['monthly_target_percent']:.0f}%): {'✅' if perf['targets_met']['monthly_target'] else '❌'}

SUMMARY
{'='*30}
Strategy Performance: {'🚀 EXCELLENT' if perf['total_return'] > 0.2 else '📈 GOOD' if perf['total_return'] > 0.1 else '⚠️ MODERATE' if perf['total_return'] > 0 else '📉 POOR'}
Risk Management: {'✅ GOOD' if perf['max_drawdown'] < 0.15 else '⚠️ MODERATE' if perf['max_drawdown'] < 0.25 else '❌ POOR'}
Trade Frequency: {'⚡ HIGH' if perf['avg_daily_trades'] > 2 else '📊 MODERATE' if perf['avg_daily_trades'] > 0.5 else '🐌 LOW'}
"""
        
        return report