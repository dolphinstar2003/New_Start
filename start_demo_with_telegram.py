#!/usr/bin/env python3
"""
Start Paper Trading Demo with Telegram
"""
import os
import json
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# Load and set Telegram config
config_file = Path(__file__).parent / 'telegram_config.json'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
    os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')
    
    # Import after setting env vars
    from telegram_integration import TELEGRAM_CONFIG
    TELEGRAM_CONFIG.update(config)

# Now import and run demo
from paper_trading_demo import main

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 STARTING PAPER TRADING DEMO WITH TELEGRAM")
    print("="*80)
    print("✅ Telegram notifications enabled")
    print("✅ Trade confirmations disabled (demo mode)")
    print("✅ Using simulated market data")
    print("="*80)
    print("\nCheck your Telegram for notifications!")
    print("Use /help command in Telegram to see available commands")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    
    asyncio.run(main())