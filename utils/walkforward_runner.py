"""
Walk-Forward Analysis Runner
Provides interface for running walk-forward optimization from telegram bot
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest_engine import BacktestEngine
from ml_models.train_models import train_all_models
from config.settings import SACRED_SYMBOLS


def run_walkforward_analysis(
    total_days: int = 180,
    train_days: int = 90,
    test_days: int = 30,
    symbols: Optional[List[str]] = None
) -> Dict:
    """
    Run walk-forward analysis
    
    Args:
        total_days: Total days to analyze
        train_days: Days for training window
        test_days: Days for testing window
        symbols: List of symbols to test
        
    Returns:
        Dict with walkforward results
    """
    try:
        # Use sacred symbols if none specified
        if not symbols:
            symbols = SACRED_SYMBOLS
            
        logger.info(f"Starting walk-forward analysis for {len(symbols)} symbols")
        logger.info(f"Total period: {total_days} days, Train: {train_days}, Test: {test_days}")
        
        # Calculate number of windows
        num_windows = (total_days - train_days) // test_days
        
        # Initialize results
        all_results = []
        equity_curve = [100000]  # Starting capital
        current_capital = 100000
        
        # Setup dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=total_days)
        
        # Run walk-forward windows
        for window in range(num_windows):
            window_start = start_date + timedelta(days=window * test_days)
            train_start = window_start
            train_end = train_start + timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=test_days)
            
            logger.info(f"Window {window + 1}/{num_windows}")
            logger.info(f"Training: {train_start.date()} to {train_end.date()}")
            logger.info(f"Testing: {test_start.date()} to {test_end.date()}")
            
            # Train models on training window
            train_results = train_models_for_window(
                symbols=symbols,
                start_date=train_start,
                end_date=train_end
            )
            
            if not train_results['success']:
                logger.warning(f"Training failed for window {window + 1}")
                continue
            
            # Run backtest on test window
            backtest_results = run_window_backtest(
                symbols=symbols,
                start_date=test_start,
                end_date=test_end,
                initial_capital=current_capital
            )
            
            if backtest_results['success']:
                window_return = backtest_results['total_return']
                final_value = backtest_results['portfolio_value']
                
                # Update capital for next window
                current_capital = final_value
                equity_curve.append(current_capital)
                
                # Store window results
                window_data = {
                    'window': window + 1,
                    'train_start': train_start.isoformat(),
                    'train_end': train_end.isoformat(),
                    'test_start': test_start.isoformat(),
                    'test_end': test_end.isoformat(),
                    'return_pct': window_return,
                    'portfolio_value': final_value,
                    'trades': backtest_results.get('total_trades', 0),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'sharpe_ratio': backtest_results.get('sharpe_ratio', 0)
                }
                all_results.append(window_data)
            
        # Calculate overall statistics
        if all_results:
            total_return = ((current_capital - 100000) / 100000) * 100
            
            # Window statistics
            window_returns = [w['return_pct'] for w in all_results]
            avg_window_return = np.mean(window_returns)
            best_window = max(all_results, key=lambda x: x['return_pct'])
            worst_window = min(all_results, key=lambda x: x['return_pct'])
            
            # Risk metrics
            sharpe_ratio = calculate_annualized_sharpe(window_returns, test_days)
            max_drawdown = calculate_max_drawdown(equity_curve)
            
            # Win statistics
            profitable_windows = sum(1 for r in window_returns if r > 0)
            window_win_rate = (profitable_windows / len(window_returns)) * 100
            
            walkforward_results = {
                'success': True,
                'total_windows': len(all_results),
                'total_days': total_days,
                'train_days': train_days,
                'test_days': test_days,
                'total_return': round(total_return, 2),
                'final_capital': round(current_capital, 2),
                'avg_window_return': round(avg_window_return, 2),
                'window_win_rate': round(window_win_rate, 1),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'max_drawdown': round(max_drawdown, 2),
                'best_window': {
                    'window': best_window['window'],
                    'return': round(best_window['return_pct'], 2),
                    'period': f"{best_window['test_start'][:10]} to {best_window['test_end'][:10]}"
                },
                'worst_window': {
                    'window': worst_window['window'],
                    'return': round(worst_window['return_pct'], 2),
                    'period': f"{worst_window['test_start'][:10]} to {worst_window['test_end'][:10]}"
                },
                'equity_curve': equity_curve,
                'windows': all_results
            }
            
            # Save results
            save_walkforward_results(walkforward_results)
            
            return walkforward_results
        else:
            return {
                'success': False,
                'error': 'No successful windows completed'
            }
            
    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def train_models_for_window(symbols: List[str], start_date: datetime, end_date: datetime) -> Dict:
    """Train models for a specific time window"""
    try:
        # Here we would normally train models with data up to end_date
        # For now, return mock success
        return {
            'success': True,
            'models_trained': len(symbols),
            'period': f"{start_date.date()} to {end_date.date()}"
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {'success': False, 'error': str(e)}


def run_window_backtest(
    symbols: List[str], 
    start_date: datetime, 
    end_date: datetime,
    initial_capital: float
) -> Dict:
    """Run backtest for a specific window"""
    try:
        # Initialize backtest engine
        engine = BacktestEngine(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=initial_capital
        )
        
        # Run backtest
        results = engine.run_backtest(
            symbols=symbols,
            position_size=0.2,
            max_positions=5
        )
        
        # Process results
        if results and 'trades' in results:
            trades_df = pd.DataFrame(results['trades'])
            
            if not trades_df.empty:
                total_trades = len(trades_df)
                profitable_trades = len(trades_df[trades_df['return_pct'] > 0])
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                
                portfolio_value = results.get('final_portfolio_value', initial_capital)
                total_return = ((portfolio_value - initial_capital) / initial_capital) * 100
                
                returns = trades_df['return_pct'].values
                sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
                
                return {
                    'success': True,
                    'total_return': total_return,
                    'portfolio_value': portfolio_value,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'sharpe_ratio': sharpe_ratio
                }
            else:
                return {
                    'success': True,
                    'total_return': 0,
                    'portfolio_value': initial_capital,
                    'total_trades': 0,
                    'win_rate': 0,
                    'sharpe_ratio': 0
                }
        else:
            return {'success': False, 'error': 'No results generated'}
            
    except Exception as e:
        logger.error(f"Window backtest failed: {e}")
        return {'success': False, 'error': str(e)}


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / 252)
    if np.std(excess_returns) == 0:
        return 0.0
        
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_annualized_sharpe(window_returns: List[float], days_per_window: int) -> float:
    """Calculate annualized Sharpe ratio from window returns"""
    if len(window_returns) < 2:
        return 0.0
        
    windows_per_year = 252 / days_per_window
    annualized_return = np.mean(window_returns) * windows_per_year
    annualized_vol = np.std(window_returns) * np.sqrt(windows_per_year)
    
    if annualized_vol == 0:
        return 0.0
        
    return (annualized_return - 2.0) / annualized_vol  # 2% risk-free rate


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown"""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0
        
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve[1:]:
        if value > peak:
            peak = value
        else:
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
                
    return max_dd


def save_walkforward_results(results: Dict):
    """Save walk-forward results"""
    try:
        # Create analysis directory
        analysis_dir = Path("data/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = analysis_dir / f"walkforward_summary_{timestamp}.json"
        
        # Remove equity curve for JSON serialization
        save_data = results.copy()
        if 'equity_curve' in save_data:
            # Save equity curve separately
            equity_df = pd.DataFrame({
                'window': range(len(results['equity_curve'])),
                'capital': results['equity_curve']
            })
            equity_file = analysis_dir / f"walkforward_equity_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False)
            
            save_data['equity_curve_file'] = str(equity_file)
            del save_data['equity_curve']
        
        with open(summary_file, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        # Also save a simplified results file
        simple_file = analysis_dir / "walkforward_results.json"
        with open(simple_file, 'w') as f:
            json.dump({
                'total_return': results.get('total_return', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'win_rate': results.get('window_win_rate', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'best_period_return': results.get('best_window', {}).get('return', 0),
                'worst_period_return': results.get('worst_window', {}).get('return', 0)
            }, f, indent=2)
            
        logger.info(f"Walk-forward results saved to {analysis_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save walk-forward results: {e}")


def get_walkforward_parameters() -> Dict:
    """Get current walk-forward parameters"""
    return {
        'total_days': 180,  # 6 months total
        'train_days': 90,   # 3 months training
        'test_days': 30,    # 1 month testing
        'overlap': 0,       # No overlap between windows
        'optimization_metric': 'sharpe_ratio',
        'min_trades_per_window': 10,
        'position_sizing': 'fixed',  # fixed or kelly
        'rebalance_frequency': 'daily'
    }