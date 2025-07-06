"""
Backtest Runner Module
Provides simplified interface for running backtests from telegram bot
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent))

from backtest.backtest_engine import BacktestEngine
from config.settings import SACRED_SYMBOLS


def run_backtest(days: int = 30, symbols: Optional[List[str]] = None) -> Dict:
    """
    Run backtest for specified number of days
    
    Args:
        days: Number of days to backtest
        symbols: List of symbols to test (defaults to SACRED_SYMBOLS)
        
    Returns:
        Dict with backtest results
    """
    try:
        # Setup dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Use sacred symbols if none specified
        if not symbols:
            symbols = SACRED_SYMBOLS
            
        logger.info(f"Running {days}-day backtest for {len(symbols)} symbols")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            initial_capital=100000.0
        )
        
        # Run backtest
        results = engine.run_backtest(
            symbols=symbols,
            position_size=0.2,  # 20% per position
            max_positions=5
        )
        
        # Calculate summary statistics
        if results and 'trades' in results:
            trades_df = pd.DataFrame(results['trades'])
            
            if not trades_df.empty:
                total_trades = len(trades_df)
                profitable_trades = len(trades_df[trades_df['return_pct'] > 0])
                win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Portfolio metrics
                portfolio_value = results.get('final_portfolio_value', 100000)
                total_return = ((portfolio_value - 100000) / 100000) * 100
                
                # Risk metrics
                returns = trades_df['return_pct'].values
                sharpe_ratio = calculate_sharpe_ratio(returns) if len(returns) > 1 else 0
                max_drawdown = calculate_max_drawdown(results.get('equity_curve', [])) if 'equity_curve' in results else 0
                
                # Trade statistics
                avg_win = trades_df[trades_df['return_pct'] > 0]['return_pct'].mean() if profitable_trades > 0 else 0
                avg_loss = trades_df[trades_df['return_pct'] < 0]['return_pct'].mean() if (total_trades - profitable_trades) > 0 else 0
                best_trade = trades_df['return_pct'].max() if total_trades > 0 else 0
                worst_trade = trades_df['return_pct'].min() if total_trades > 0 else 0
                
                backtest_results = {
                    'success': True,
                    'days': days,
                    'symbols_tested': len(symbols),
                    'total_return': round(total_return, 2),
                    'portfolio_value': round(portfolio_value, 2),
                    'total_trades': total_trades,
                    'profitable_trades': profitable_trades,
                    'win_rate': round(win_rate, 1),
                    'sharpe_ratio': round(sharpe_ratio, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2),
                    'best_trade': round(best_trade, 2),
                    'worst_trade': round(worst_trade, 2),
                    'profit_factor': round(abs(avg_win / avg_loss) if avg_loss != 0 else 0, 2),
                    'trades_per_day': round(total_trades / days, 2) if days > 0 else 0
                }
                
                # Save detailed results
                save_backtest_results(backtest_results, trades_df)
                
                return backtest_results
            else:
                return {
                    'success': False,
                    'error': 'No trades generated during backtest period'
                }
        else:
            return {
                'success': False,
                'error': 'Backtest failed to generate results'
            }
            
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio from returns"""
    if len(returns) < 2:
        return 0.0
        
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if np.std(excess_returns) == 0:
        return 0.0
        
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve"""
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


def save_backtest_results(results: Dict, trades_df: pd.DataFrame):
    """Save backtest results to file"""
    try:
        # Create analysis directory
        analysis_dir = Path("data/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = analysis_dir / f"backtest_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save trades
        trades_file = analysis_dir / f"backtest_trades_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"Backtest results saved to {analysis_dir}")
        
    except Exception as e:
        logger.error(f"Failed to save backtest results: {e}")


def get_backtest_parameters() -> Dict:
    """Get current backtest parameters"""
    return {
        'stop_loss': 0.03,  # 3%
        'take_profit': 0.08,  # 8%
        'position_size': 0.20,  # 20%
        'max_positions': 5,
        'min_volume': 1000000,  # Minimum daily volume
        'min_score': 65,  # Minimum ML score
        'commission': 0.001,  # 0.1%
        'slippage': 0.0005  # 0.05%
    }