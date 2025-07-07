# Paper Trading Module

Real-time paper trading system with AlgoLab API integration for live market data.

## Features

- **Real-time Data**: Live prices from AlgoLab during market hours (10:00-18:00 Istanbul time)
- **Multiple Strategies**: Balanced, Aggressive, and Conservative portfolios
- **Portfolio Management**: Automatic position sizing and risk management
- **Performance Tracking**: Detailed metrics and trade history
- **Dashboard**: Web-based monitoring interface

## Quick Start

### 1. Run Paper Trading

```bash
python paper_trader.py
```

This will:
- Initialize AlgoLab connection
- Start monitoring all three portfolios
- Execute trades based on signals
- Generate daily reports at 18:15

### 2. Monitor with Dashboard

```bash
./run_dashboard.sh
# or
streamlit run dashboard_simple.py
```

Open http://localhost:8501 in your browser.

## Portfolio Strategies

### Balanced Portfolio
- Strategy: VixFix Enhanced Supertrend
- Risk: Medium
- Target: 56,908% backtest return

### Aggressive Portfolio  
- Strategy: Supertrend Only
- Risk: High
- Target: 41,347% average return

### Conservative Portfolio
- Strategy: MACD + ADX Balanced
- Risk: Low
- Target: 3,585% + 3,603% average return

## Configuration

### AlgoLab API Setup

Add to `.env` file:
```
ALGOLAB_API_KEY=API-XXXXX
ALGOLAB_USERNAME=TC_NUMBER
ALGOLAB_PASSWORD=PASSWORD
```

### Portfolio Parameters

Each portfolio has:
- Initial Capital: ₺50,000
- Max Positions: 5-10
- Position Size: 10-30% of portfolio
- Stop Loss: 3%
- Take Profit: 8%

## Data Sources

- **Market Hours (10:00-18:00)**: AlgoLab real-time API
- **After Hours**: Yahoo Finance fallback
- **Rate Limiting**: 5 seconds between AlgoLab requests

## Files

- `paper_trader.py` - Main trading engine
- `portfolio_manager.py` - Portfolio and position management
- `signal_generator.py` - Trading signal generation
- `data_fetcher.py` - Market data fetching (AlgoLab/Yahoo)
- `data_fetcher_algolab.py` - AlgoLab specific implementation
- `performance_tracker.py` - Performance metrics
- `dashboard_simple.py` - Streamlit monitoring interface
- `telegram_bot.py` - Telegram notifications

## Monitoring

### Console Output
```
🚀 Paper Trader Started!
============================================================
Trading will run every 5 minutes during market hours
Daily reports will be generated at 18:15
```

### Dashboard Features
- Real-time portfolio values
- Current positions with P&L
- Trade history and statistics
- Performance charts
- AlgoLab connection status

## Troubleshooting

### AlgoLab Authentication
- SMS authentication required (max 2 attempts)
- Session cached for 24 hours
- Check logs/algolab_api.log for details

### Data Issues
- Verify market hours (10:00-18:00 Istanbul)
- Check AlgoLab rate limiting (5s between requests)
- Fallback to Yahoo Finance if AlgoLab fails

### Performance
- Each portfolio saved to data/[name]_portfolio.json
- Trade history in portfolio files
- Daily reports in reports/daily_report_YYYY-MM-DD.json