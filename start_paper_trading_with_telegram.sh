#!/bin/bash
# Paper Trading Launcher with Telegram Config

# Load Telegram config from JSON
export TELEGRAM_BOT_TOKEN="7805877166:AAEgy8Ic31qQv-ImINy8VrDnOUoZfC18bxo"
export TELEGRAM_CHAT_ID="547884175"

# Activate virtual environment
source venv/bin/activate

echo "=================================="
echo "Starting Paper Trading with Telegram"
echo "=================================="
echo "Bot Token: ${TELEGRAM_BOT_TOKEN:0:10}..."
echo "Chat ID: $TELEGRAM_CHAT_ID"
echo "=================================="

# Run paper trading
python run_paper_trading_auto.py