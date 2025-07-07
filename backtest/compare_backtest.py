"""
Compare Backtest Results
Comprehensive comparison of different backtest strategies
Outputs detailed CSV report
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from loguru import logger
import asyncio
from typing import Dict, List, Tuple
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import DATA_DIR
from backtest.realistic_backtest import run_realistic_backtest
from backtest.backtest_sirali import run_backtest_4h, run_backtest_1d, run_backtest_hybrid
from backtest.backtest_oncelikli import run_priority_backtest
from backtest.backtest_ml_xgboost import run_ml_xgboost_backtest
from backtest.backtest_dl_lstm import run_dl_lstm_backtest
from backtest.rotation_backtest import run_rotation_backtest


class BacktestComparator:
    """Compare multiple backtest strategies"""
    
    def __init__(self):
        self.results_dir = DATA_DIR / 'analysis'
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Backtest configurations
        self.backtest_configs = {
            # Current realistic backtest
            'realistic': {
                'function': run_realistic_backtest,
                'params': {},
                'description': 'Current 5-indicator equal weight'
            },
            
            # Hierarchical backtests
            'hierarchical_4h': {
                'function': run_backtest_4h,
                'params': {'risk_strategy': 'sl_plus_conditional_trailing'},
                'description': '4h timeframe sequential'
            },
            'hierarchical_1d': {
                'function': run_backtest_1d,
                'params': {'risk_strategy': 'sl_plus_conditional_trailing'},
                'description': '1d timeframe sequential'
            },
            'hierarchical_hybrid': {
                'function': run_backtest_hybrid,
                'params': {'risk_strategy': 'sl_plus_conditional_trailing'},
                'description': '1d trend + 4h entry'
            },
            
            # Priority-based backtest
            'priority_core': {
                'function': run_priority_backtest,
                'params': {},
                'description': 'Core indicators (ST+SQZ) priority'
            },
            
            # Different risk strategies for hierarchical
            'hier_4h_signals_only': {
                'function': run_backtest_4h,
                'params': {'risk_strategy': 'only_signals'},
                'description': '4h no stop loss'
            },
            'hier_4h_sl_only': {
                'function': run_backtest_4h,
                'params': {'risk_strategy': 'with_sl'},
                'description': '4h with stop loss only'
            },
            'hier_4h_sl_trailing': {
                'function': run_backtest_4h,
                'params': {'risk_strategy': 'sl_plus_trailing'},
                'description': '4h with SL + trailing'
            },
            
            # Machine Learning backtests
            'ml_xgboost': {
                'function': run_ml_xgboost_backtest,
                'params': {},
                'description': 'XGBoost ML predictions'
            },
            
            # Deep Learning backtests
            'dl_lstm': {
                'function': run_dl_lstm_backtest,
                'params': {},
                'description': 'LSTM deep learning'
            },
            
            # Dynamic Rotation Strategy
            'rotation_top10': {
                'function': run_rotation_backtest,
                'params': {},
                'description': 'Dynamic top 10 rotation'
            }
        }
        
        # Metrics to compare
        self.key_metrics = [
            'total_return',
            'sharpe_ratio',
            'win_rate',
            'max_drawdown',
            'total_trades',
            'profit_factor',
            'best_trade',
            'worst_trade',
            'avg_trade',
            'calmar_ratio',
            'sortino_ratio',
            'avg_win_loss_ratio'
        ]
        
        logger.info(f"BacktestComparator initialized with {len(self.backtest_configs)} strategies")
        logger.info(f"Including {len([k for k in self.backtest_configs if 'ml' in k or 'dl' in k])} ML/DL strategies")
    
    def calculate_additional_metrics(self, result: Dict) -> Dict:
        """Calculate additional metrics for comparison"""
        # Calmar Ratio = Annual Return / Max Drawdown
        annual_return = result.get('total_return', 0) * 12 / 30  # Approximate annual
        max_dd = result.get('max_drawdown', 1)
        result['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0
        
        # Average trade
        if result.get('total_trades', 0) > 0:
            result['avg_trade'] = result.get('total_return', 0) / result['total_trades']
        else:
            result['avg_trade'] = 0
        
        # Win/Loss ratio
        wins = result.get('profitable_trades', 0)
        losses = result.get('losing_trades', 1)
        result['avg_win_loss_ratio'] = wins / losses if losses > 0 else wins
        
        # Sortino Ratio (placeholder - would need downside deviation)
        result['sortino_ratio'] = result.get('sharpe_ratio', 0) * 1.2  # Approximation
        
        # Profit factor (if not already calculated)
        if 'profit_factor' not in result:
            if wins > 0 and losses > 0:
                avg_win = abs(result.get('best_trade', 0))
                avg_loss = abs(result.get('worst_trade', 1))
                result['profit_factor'] = (wins * avg_win) / (losses * avg_loss) if avg_loss > 0 else 0
            else:
                result['profit_factor'] = 0
        
        return result
    
    async def run_single_backtest(self, name: str, config: Dict, days: int) -> Dict:
        """Run a single backtest configuration"""
        logger.info(f"Running {name}: {config['description']}")
        
        try:
            # Run backtest
            func = config['function']
            params = config['params'].copy()
            result = await func(days=days, **params)
            
            # Add metadata
            result['strategy_name'] = name
            result['description'] = config['description']
            result['days'] = days
            
            # Calculate additional metrics
            result = self.calculate_additional_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            return {
                'strategy_name': name,
                'description': config['description'],
                'error': str(e),
                'days': days
            }
    
    async def run_all_backtests(self, days: int = 30) -> pd.DataFrame:
        """Run all backtest configurations"""
        results = []
        
        # Run backtests concurrently in batches
        batch_size = 3
        backtest_items = list(self.backtest_configs.items())
        
        for i in range(0, len(backtest_items), batch_size):
            batch = backtest_items[i:i+batch_size]
            
            tasks = [
                self.run_single_backtest(name, config, days)
                for name, config in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Reorder columns
        id_cols = ['strategy_name', 'description', 'days', 'backtest_engine']
        metric_cols = [col for col in self.key_metrics if col in df.columns]
        other_cols = [col for col in df.columns if col not in id_cols + metric_cols]
        
        df = df[id_cols + metric_cols + other_cols]
        
        return df
    
    def generate_comparison_report(self, df: pd.DataFrame) -> Dict:
        """Generate detailed comparison report"""
        report = {
            'summary': {},
            'rankings': {},
            'recommendations': []
        }
        
        # Best strategy by key metrics
        for metric in self.key_metrics:
            if metric in df.columns:
                if metric in ['max_drawdown', 'worst_trade']:  # Lower is better
                    best_idx = df[metric].idxmin()
                else:  # Higher is better
                    best_idx = df[metric].idxmax()
                
                if pd.notna(best_idx):
                    best_strategy = df.loc[best_idx, 'strategy_name']
                    best_value = df.loc[best_idx, metric]
                    report['rankings'][metric] = {
                        'best_strategy': best_strategy,
                        'value': best_value
                    }
        
        # Overall score (weighted average ranking)
        weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'win_rate': 0.15,
            'max_drawdown': 0.20,
            'profit_factor': 0.10,
            'calmar_ratio': 0.10
        }
        
        scores = {}
        for idx, row in df.iterrows():
            score = 0
            for metric, weight in weights.items():
                if metric in row and pd.notna(row[metric]):
                    # Normalize to 0-100 scale
                    if metric == 'max_drawdown':
                        normalized = 100 - min(row[metric], 100)
                    else:
                        normalized = min(row[metric], 100)
                    score += normalized * weight
            scores[row['strategy_name']] = score
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        report['summary']['overall_ranking'] = sorted_scores
        
        # Recommendations
        best_overall = sorted_scores[0][0]
        report['recommendations'].append(
            f"Best overall strategy: {best_overall} (Score: {sorted_scores[0][1]:.2f})"
        )
        
        # Risk-adjusted recommendation
        if 'sharpe_ratio' in df.columns:
            best_sharpe_idx = df['sharpe_ratio'].idxmax()
            if pd.notna(best_sharpe_idx):
                best_risk_adj = df.loc[best_sharpe_idx, 'strategy_name']
                report['recommendations'].append(
                    f"Best risk-adjusted: {best_risk_adj} (Sharpe: {df.loc[best_sharpe_idx, 'sharpe_ratio']:.2f})"
                )
        
        # Conservative recommendation
        if 'max_drawdown' in df.columns:
            safest_idx = df['max_drawdown'].idxmin()
            if pd.notna(safest_idx):
                safest = df.loc[safest_idx, 'strategy_name']
                report['recommendations'].append(
                    f"Most conservative: {safest} (Max DD: {df.loc[safest_idx, 'max_drawdown']:.2f}%)"
                )
        
        return report
    
    def save_results(self, df: pd.DataFrame, report: Dict, days: int):
        """Save results to CSV and JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed comparison CSV
        csv_path = self.results_dir / f'backtest_comparison_{days}d_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison CSV to {csv_path}")
        
        # Save summary report
        report_path = self.results_dir / f'backtest_report_{days}d_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved report JSON to {report_path}")
        
        # Create a summary CSV for easy viewing
        summary_data = []
        for strategy in df['strategy_name'].unique():
            row = df[df['strategy_name'] == strategy].iloc[0]
            summary_data.append({
                'Strategy': strategy,
                'Return %': f"{row.get('total_return', 0):.2f}",
                'Sharpe': f"{row.get('sharpe_ratio', 0):.2f}",
                'Win Rate %': f"{row.get('win_rate', 0):.1f}",
                'Max DD %': f"{row.get('max_drawdown', 0):.2f}",
                'Trades': row.get('total_trades', 0),
                'Score': next((s[1] for s in report['summary']['overall_ranking'] if s[0] == strategy), 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Score', ascending=False)
        summary_path = self.results_dir / f'backtest_summary_{days}d_{timestamp}.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary CSV to {summary_path}")
        
        return csv_path, report_path, summary_path
    
    async def compare_all(self, days: int = 30) -> Tuple[pd.DataFrame, Dict]:
        """Run complete comparison"""
        logger.info(f"Starting backtest comparison for {days} days...")
        
        # Run all backtests
        df = await self.run_all_backtests(days)
        
        # Generate report
        report = self.generate_comparison_report(df)
        
        # Save results
        csv_path, report_path, summary_path = self.save_results(df, report, days)
        
        # Print summary
        print("\n" + "="*80)
        print(f"BACKTEST COMPARISON RESULTS ({days} days)")
        print("="*80)
        
        print("\n📊 TOP 5 STRATEGIES BY OVERALL SCORE:")
        for i, (strategy, score) in enumerate(report['summary']['overall_ranking'][:5], 1):
            row = df[df['strategy_name'] == strategy].iloc[0]
            print(f"\n{i}. {strategy} (Score: {score:.2f})")
            print(f"   Return: {row.get('total_return', 0):.2f}% | "
                  f"Sharpe: {row.get('sharpe_ratio', 0):.2f} | "
                  f"Win Rate: {row.get('win_rate', 0):.1f}% | "
                  f"Max DD: {row.get('max_drawdown', 0):.2f}%")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"• {rec}")
        
        print(f"\n📁 Results saved to:")
        print(f"• Detailed: {csv_path}")
        print(f"• Summary: {summary_path}")
        print(f"• Report: {report_path}")
        print("="*80)
        
        return df, report


# Convenience function
async def compare_backtests(days: int = 30):
    """Run backtest comparison"""
    comparator = BacktestComparator()
    return await comparator.compare_all(days)


if __name__ == "__main__":
    # Run comparison
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(compare_backtests(days))