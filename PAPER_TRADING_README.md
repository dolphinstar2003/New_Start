# 📈 Paper Trading System - Dynamic Portfolio Optimizer

## 🚀 Overview

This paper trading system implements the **Dynamic Portfolio Optimizer** strategy with real-time AlgoLab data integration and Telegram notifications/control.

### Key Features
- ✅ Real-time AlgoLab data streaming
- ✅ Telegram bot for notifications and trade confirmations
- ✅ Up to 10 concurrent positions
- ✅ Dynamic position sizing (20-30%)
- ✅ Portfolio rotation for maximum returns
- ✅ 3% stop loss with trailing stops
- ✅ 8% take profit targets
- ✅ Automatic state persistence

## 📋 Prerequisites

1. **AlgoLab Account**
   - Valid AlgoLab credentials
   - API key set as environment variable:
     ```bash
     export ALGOLAB_API_KEY="your_api_key_here"
     ```

2. **Telegram Bot** (Optional but recommended)
   - Create a bot via [@BotFather](https://t.me/botfather)
   - Get your chat ID
   - Bot will be configured during first run

3. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Quick Start

### Method 1: Easy Launcher (Recommended)
```bash
python start_paper_trading.py
```

This will:
- Check all requirements
- Setup Telegram if needed
- Initialize connections
- Start paper trading with your preferred settings

### Method 2: Direct Run
```bash
python paper_trading_module.py
```

### Method 3: Dashboard (Visual Interface)
```bash
streamlit run paper_trading_dashboard.py
```

## 📱 Telegram Commands

Once your bot is running, send these commands to your Telegram bot:

### Portfolio Management
- `/start` - Initialize bot and show help
- `/status` - Show current portfolio status
- `/positions` - List all open positions
- `/trades` - Show recent trade history
- `/performance` - Display performance metrics

### Trading Control
- `/start_trading` - Enable automatic trading
- `/stop_trading` - Disable automatic trading
- `/force_check` - Force position evaluation

### Market Analysis
- `/opportunities` - Show top market opportunities
- `/analyze SYMBOL` - Analyze specific symbol (e.g., `/analyze GARAN`)

### System
- `/settings` - Show current trading parameters
- `/help` - Show all available commands

## ⚙️ Configuration

### Trading Parameters (in `paper_trading_module.py`)
```python
PORTFOLIO_PARAMS = {
    'base_position_pct': 0.20,      # Base 20% position
    'max_position_pct': 0.30,       # Max 30% (strong signals)
    'min_position_pct': 0.05,       # Min 5% (weak signal)
    'max_positions': 10,            # Max 10 positions
    'stop_loss': 0.03,              # 3% stop loss
    'take_profit': 0.08,            # 8% take profit
    'trailing_start': 0.04,         # 4% trailing start
    'trailing_distance': 0.02,      # 2% trailing distance
}
```

### Telegram Setup
First run will prompt you to configure Telegram:
1. Enter your bot token from @BotFather
2. Enter your chat ID
3. Optionally set admin user IDs

Configuration is saved to `telegram_config.json`

## 📊 Trading Strategy

### Entry Signals
The system evaluates opportunities based on:
1. **Price Momentum** (30% weight)
   - 1-hour and daily price changes
2. **Volume Activity** (20% weight)
   - Volume ratio vs 20-period average
3. **Technical Indicators** (30% weight)
   - Supertrend signals
   - RSI levels
4. **Market Depth** (10% weight)
   - Bid/ask ratios
5. **Trade Flow** (10% weight)
   - Buy/sell volume ratios

### Position Management
- **Entry**: Score > 1.0 (configurable)
- **Strong Entry**: Score > 2.5 (larger position)
- **Exit Conditions**:
  - Stop Loss: -3%
  - Take Profit: +8%
  - Trailing Stop: Activates at +4%
  - Exit Signal: Score < -0.5

### Portfolio Rotation
Every 2 days, the system checks for rotation opportunities:
- Compares existing positions with new opportunities
- Rotates if score improvement > 3.0
- Minimum holding period: 3 days
- Won't rotate profitable positions (>15%)

## 🛡️ Safety Features

### Trade Confirmations
All trades require Telegram confirmation by default:
- 60-second timeout for confirmations
- Can be disabled via `/confirm` command

### Risk Management
- Maximum 95% of portfolio can be invested
- Automatic position sizing based on signal strength
- Trailing stops to protect profits
- Daily drawdown monitoring

### State Persistence
- Automatic state saving
- Resume from last state on restart
- Manual save via `save` command

## 📈 Performance Monitoring

### Real-time Updates
- Hourly portfolio updates via Telegram
- Daily summary at 18:30 Istanbul time
- Console updates every 30 minutes

### Metrics Tracked
- Total Return ($ and %)
- Win Rate
- Sharpe Ratio
- Maximum Drawdown
- Average Win/Loss
- Profit Factor

## 🐛 Troubleshooting

### AlgoLab Connection Issues
```bash
# Check API key
echo $ALGOLAB_API_KEY

# Test authentication
python test_algolab_auth.py
```

### Telegram Bot Not Responding
1. Check bot token is correct
2. Ensure you've started conversation with `/start`
3. Verify chat ID in `telegram_config.json`

### No Market Data
- Verify market hours (10:00-18:00 Istanbul time)
- Check WebSocket connection in logs
- Ensure symbols have `.IS` suffix

## 📝 Logs

Detailed logs are written to console with levels:
- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors

## 🔧 Advanced Usage

### Custom Position Sizing
```python
# In paper_trading_module.py
def calculate_position_size(...):
    # Modify position sizing logic
```

### Add Custom Indicators
```python
# In calculate_opportunity_score()
# Add your custom scoring logic
```

### Modify Trading Hours
```python
# In trading_loop()
if 10 <= current_time.hour < 18:  # Change hours here
```

## 📞 Support

For issues or questions:
1. Check logs for error messages
2. Review this README
3. Check AlgoLab API status
4. Verify Telegram bot configuration

## ⚠️ Disclaimer

This is a PAPER TRADING system for testing strategies. No real money is involved. Always test thoroughly before considering live trading.

## 🎯 Best Practices

1. **Start Conservative**
   - Begin with trade confirmations enabled
   - Monitor for a few days before full auto

2. **Regular Monitoring**
   - Check Telegram notifications
   - Review daily summaries
   - Analyze losing trades

3. **Optimize Gradually**
   - Adjust parameters based on performance
   - Test one change at a time
   - Keep records of modifications

Good luck with your paper trading! 🚀📈