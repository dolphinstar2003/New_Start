"""
Trading Report Generator
Creates comprehensive PDF/HTML reports for the trading system
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional, Any
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_full_report(paper_trader=None, format='html') -> str:
    """
    Generate comprehensive trading report
    
    Args:
        paper_trader: Paper trader instance
        format: Output format ('html' or 'csv')
        
    Returns:
        Path to generated report
    """
    try:
        # Create reports directory
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'html':
            report_path = reports_dir / f"trading_report_{timestamp}.html"
            html_content = generate_html_report(paper_trader)
            with open(report_path, 'w') as f:
                f.write(html_content)
        else:
            # Generate CSV reports
            report_path = reports_dir / f"trading_report_{timestamp}"
            report_path.mkdir(exist_ok=True)
            
            # Portfolio summary
            if paper_trader:
                portfolio_df = pd.DataFrame([paper_trader.get_portfolio_status()])
                portfolio_df.to_csv(report_path / "portfolio_summary.csv", index=False)
                
                # Trade history
                if paper_trader.trade_history:
                    trades_df = pd.DataFrame(paper_trader.trade_history)
                    trades_df.to_csv(report_path / "trade_history.csv", index=False)
                
                # Open positions
                if paper_trader.portfolio['positions']:
                    positions_data = []
                    for symbol, pos in paper_trader.portfolio['positions'].items():
                        positions_data.append({
                            'symbol': symbol,
                            'shares': pos['shares'],
                            'average_price': pos['average_price'],
                            'current_price': pos.get('current_price', pos['average_price']),
                            'unrealized_pnl': pos['unrealized_pnl'],
                            'unrealized_pnl_pct': pos['unrealized_pnl_pct'],
                            'stop_loss': pos.get('stop_loss', 0),
                            'take_profit': pos.get('take_profit', 0)
                        })
                    positions_df = pd.DataFrame(positions_data)
                    positions_df.to_csv(report_path / "open_positions.csv", index=False)
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise


def generate_html_report(paper_trader) -> str:
    """Generate HTML format report"""
    
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Get portfolio data
    if paper_trader:
        portfolio_status = paper_trader.get_portfolio_status()
        performance_metrics = paper_trader.get_performance_metrics()
        
        # Create HTML content
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trading Report - {timestamp}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #666;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #f44336;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .section {{
            margin: 40px 0;
        }}
        .timestamp {{
            color: #999;
            font-size: 12px;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Trading System Report</h1>
        <div class="timestamp">Generated: {timestamp}</div>
        
        <div class="section">
            <h2>Portfolio Overview</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Value</div>
                    <div class="metric-value">${portfolio_status['portfolio_value']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Cash Available</div>
                    <div class="metric-value">${portfolio_status['cash']:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if portfolio_status['total_return_pct'] >= 0 else 'negative'}">
                        {portfolio_status['total_return_pct']:+.2f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Open Positions</div>
                    <div class="metric-value">{portfolio_status['num_positions']}</div>
                </div>
            </div>
        </div>
"""
        
        # Add performance metrics if available
        if performance_metrics:
            html += f"""
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{performance_metrics['win_rate']:.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{performance_metrics['sharpe_ratio']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{performance_metrics['max_drawdown']:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{performance_metrics.get('profit_factor', 0):.2f}</div>
                </div>
            </div>
        </div>
"""
        
        # Add open positions table
        if paper_trader.portfolio['positions']:
            html += """
        <div class="section">
            <h2>Open Positions</h2>
            <table>
                <tr>
                    <th>Symbol</th>
                    <th>Shares</th>
                    <th>Entry Price</th>
                    <th>Current Price</th>
                    <th>P&L</th>
                    <th>P&L %</th>
                    <th>Stop Loss</th>
                </tr>
"""
            for symbol, pos in paper_trader.portfolio['positions'].items():
                pnl_class = 'positive' if pos['unrealized_pnl'] >= 0 else 'negative'
                html += f"""
                <tr>
                    <td><strong>{symbol}</strong></td>
                    <td>{pos['shares']}</td>
                    <td>${pos['average_price']:.2f}</td>
                    <td>${pos.get('current_price', pos['average_price']):.2f}</td>
                    <td class="{pnl_class}">${pos['unrealized_pnl']:+.2f}</td>
                    <td class="{pnl_class}">{pos['unrealized_pnl_pct']:+.1f}%</td>
                    <td>${pos.get('stop_loss', 0):.2f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""
        
        # Add recent trades
        if paper_trader.trade_history:
            recent_trades = paper_trader.trade_history[-10:]  # Last 10 trades
            html += """
        <div class="section">
            <h2>Recent Trades</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Shares</th>
                    <th>Price</th>
                    <th>Total Value</th>
                    <th>Commission</th>
                </tr>
"""
            for trade in reversed(recent_trades):
                html += f"""
                <tr>
                    <td>{trade['date']}</td>
                    <td><strong>{trade['symbol']}</strong></td>
                    <td>{trade['action'].upper()}</td>
                    <td>{trade['shares']}</td>
                    <td>${trade['price']:.2f}</td>
                    <td>${trade['total_value']:.2f}</td>
                    <td>${trade['commission']:.2f}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""
    else:
        # No paper trader, generate basic report
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
        .error { color: red; padding: 20px; background-color: #ffe6e6; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Trading System Report</h1>
        <div class="error">
            No trading data available. Please start the trading system first.
        </div>
    </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


def generate_performance_chart(equity_curve: List[float], save_path: str):
    """Generate performance chart"""
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, linewidth=2)
        plt.title('Portfolio Equity Curve', fontsize=16)
        plt.xlabel('Time Period')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Performance chart saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate performance chart: {e}")


def generate_trade_analysis(trades_df: pd.DataFrame) -> Dict:
    """Analyze trades and generate statistics"""
    if trades_df.empty:
        return {}
    
    analysis = {
        'total_trades': len(trades_df),
        'winning_trades': len(trades_df[trades_df['return_pct'] > 0]),
        'losing_trades': len(trades_df[trades_df['return_pct'] < 0]),
        'avg_win': trades_df[trades_df['return_pct'] > 0]['return_pct'].mean(),
        'avg_loss': trades_df[trades_df['return_pct'] < 0]['return_pct'].mean(),
        'largest_win': trades_df['return_pct'].max(),
        'largest_loss': trades_df['return_pct'].min(),
        'avg_trade_duration': trades_df['duration_hours'].mean() if 'duration_hours' in trades_df else 0,
        'most_traded_symbol': trades_df['symbol'].value_counts().index[0] if 'symbol' in trades_df else 'N/A'
    }
    
    return analysis