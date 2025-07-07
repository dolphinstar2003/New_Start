#!/usr/bin/env python3
"""
Performance Tracker for Paper Trading
Tracks and analyzes portfolio performance over time
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks and analyzes trading performance"""
    
    def __init__(self, data_dir: str = "paper_trading/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Performance history
        self.performance_history = self.load_performance_history()
        
    def load_performance_history(self) -> dict:
        """Load performance history from file"""
        history_file = self.data_dir / "performance_history.json"
        
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
                return {}
        
        return {}
    
    def save_performance_history(self):
        """Save performance history to file"""
        history_file = self.data_dir / "performance_history.json"
        
        try:
            with open(history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance history: {e}")
    
    def update_daily_performance(self, portfolio_manager):
        """Update daily performance for all portfolios"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        if date_str not in self.performance_history:
            self.performance_history[date_str] = {}
        
        # Update each portfolio
        for name, portfolio in portfolio_manager.portfolios.items():
            metrics = portfolio.get_performance_metrics()
            
            self.performance_history[date_str][name] = {
                'portfolio_value': metrics['portfolio_value'],
                'total_return': metrics['total_return'],
                'daily_pnl': metrics['total_pnl'],
                'positions_count': metrics['num_positions'],
                'cash': metrics['cash'],
                'win_rate': metrics['win_rate'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'portfolio_heat': metrics['portfolio_heat']
            }
        
        self.save_performance_history()
        logger.info(f"Updated performance history for {date_str}")
    
    def calculate_period_performance(self, portfolio_name: str, days: int = 30) -> dict:
        """Calculate performance over specified period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        performance_data = []
        
        for date_str, daily_data in self.performance_history.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            if start_date <= date <= end_date and portfolio_name in daily_data:
                data = daily_data[portfolio_name].copy()
                data['date'] = date
                performance_data.append(data)
        
        if not performance_data:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(performance_data)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # Calculate metrics
        total_return = ((df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1) * 100
        
        # Daily returns
        daily_returns = df['portfolio_value'].pct_change().dropna()
        
        # Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe ratio
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win days
        win_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = (win_days / total_days * 100) if total_days > 0 else 0
        
        return {
            'period_days': days,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': df['positions_count'].sum(),
            'avg_positions': df['positions_count'].mean(),
            'final_value': df['portfolio_value'].iloc[-1],
            'initial_value': df['portfolio_value'].iloc[0]
        }
    
    def generate_performance_chart(self, portfolio_names: List[str] = None, 
                                  days: int = 30, save_path: str = None):
        """Generate performance comparison chart"""
        if portfolio_names is None:
            portfolio_names = ['aggressive', 'balanced', 'conservative']
        
        # Prepare data
        plot_data = {}
        
        for portfolio_name in portfolio_names:
            dates = []
            values = []
            
            for date_str, daily_data in sorted(self.performance_history.items()):
                if portfolio_name in daily_data:
                    dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    values.append(daily_data[portfolio_name]['portfolio_value'])
            
            if dates and values:
                plot_data[portfolio_name] = pd.Series(values, index=dates)
        
        if not plot_data:
            logger.warning("No data available for chart generation")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot portfolio values
        for name, series in plot_data.items():
            # Normalize to percentage returns
            returns = (series / series.iloc[0] - 1) * 100
            plt.plot(returns.index, returns.values, label=name.capitalize(), linewidth=2)
        
        plt.title('Portfolio Performance Comparison', fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return (%)', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_metrics_table(self, days: int = 30) -> pd.DataFrame:
        """Generate performance metrics table for all portfolios"""
        metrics_data = []
        
        portfolio_names = ['aggressive', 'balanced', 'conservative']
        
        for portfolio_name in portfolio_names:
            metrics = self.calculate_period_performance(portfolio_name, days)
            
            if metrics:
                metrics['portfolio'] = portfolio_name
                metrics_data.append(metrics)
        
        if not metrics_data:
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        df.set_index('portfolio', inplace=True)
        
        # Format for display
        display_columns = {
            'total_return': 'Return %',
            'volatility': 'Volatility %',
            'sharpe_ratio': 'Sharpe',
            'max_drawdown': 'Max DD %',
            'win_rate': 'Win Rate %',
            'avg_positions': 'Avg Positions',
            'final_value': 'Final Value'
        }
        
        df = df[list(display_columns.keys())]
        df.rename(columns=display_columns, inplace=True)
        
        return df
    
    def analyze_trade_performance(self, portfolio_manager) -> dict:
        """Analyze individual trade performance"""
        trade_analysis = {}
        
        for name, portfolio in portfolio_manager.portfolios.items():
            trades = []
            
            # Analyze closed positions
            for pos in portfolio.closed_positions:
                trade_data = {
                    'symbol': pos.symbol,
                    'entry_date': pos.entry_date,
                    'exit_date': pos.exit_date,
                    'hold_days': (pos.exit_date - pos.entry_date).days,
                    'entry_price': pos.entry_price,
                    'exit_price': pos.exit_price,
                    'shares': pos.shares,
                    'pnl': pos.get_pnl(),
                    'pnl_pct': pos.get_pnl_percentage(),
                    'strategy': pos.strategy
                }
                trades.append(trade_data)
            
            if trades:
                df = pd.DataFrame(trades)
                
                # Calculate statistics
                winning_trades = df[df['pnl'] > 0]
                losing_trades = df[df['pnl'] < 0]
                
                trade_analysis[name] = {
                    'total_trades': len(df),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': len(winning_trades) / len(df) * 100 if len(df) > 0 else 0,
                    'avg_win': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
                    'avg_loss': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
                    'largest_win': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
                    'largest_loss': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
                    'avg_hold_days': df['hold_days'].mean(),
                    'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
                }
            else:
                trade_analysis[name] = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'largest_win': 0,
                    'largest_loss': 0,
                    'avg_hold_days': 0,
                    'profit_factor': 0
                }
        
        return trade_analysis
    
    def generate_summary_report(self, portfolio_manager, save_path: str = None):
        """Generate comprehensive performance summary report"""
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'portfolios': {}
        }
        
        # Performance metrics for different periods
        periods = [7, 30, 90]
        
        for name in portfolio_manager.portfolios.keys():
            portfolio_report = {
                'current_metrics': portfolio_manager.portfolios[name].get_performance_metrics(),
                'period_performance': {},
                'trade_analysis': {}
            }
            
            # Period performance
            for days in periods:
                period_metrics = self.calculate_period_performance(name, days)
                if period_metrics:
                    portfolio_report['period_performance'][f'{days}_days'] = period_metrics
            
            report['portfolios'][name] = portfolio_report
        
        # Add trade analysis
        trade_analysis = self.analyze_trade_performance(portfolio_manager)
        for name, analysis in trade_analysis.items():
            if name in report['portfolios']:
                report['portfolios'][name]['trade_analysis'] = analysis
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report saved to {save_path}")
        
        return report


if __name__ == "__main__":
    # Test performance tracker
    tracker = PerformanceTracker()
    
    # Generate sample data
    sample_history = {
        "2025-01-01": {
            "aggressive": {
                "portfolio_value": 50000,
                "total_return": 0,
                "daily_pnl": 0,
                "positions_count": 0
            }
        },
        "2025-01-02": {
            "aggressive": {
                "portfolio_value": 51000,
                "total_return": 2.0,
                "daily_pnl": 1000,
                "positions_count": 3
            }
        }
    }
    
    tracker.performance_history = sample_history
    tracker.save_performance_history()
    
    print("Performance tracker initialized with sample data")