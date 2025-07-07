#!/usr/bin/env python3
"""
Test Telegram Bot Connection
"""
import os
import json
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# Load config from JSON file
config_file = Path(__file__).parent / 'telegram_config.json'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Set environment variables
    os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
    os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')
    
    print(f"Loaded config from {config_file}")
    print(f"Bot Token: {config['bot_token'][:10]}...")
    print(f"Chat ID: {config['chat_id']}")
else:
    print(f"Config file not found: {config_file}")
    sys.exit(1)

# Now import after setting env vars
from telegram_integration import TelegramBot, TELEGRAM_CONFIG

# Update global config
TELEGRAM_CONFIG.update(config)

async def test_bot():
    """Test Telegram bot"""
    print("\n" + "="*50)
    print("TESTING TELEGRAM BOT")
    print("="*50)
    
    try:
        # Create bot instance
        bot = TelegramBot()
        print("✅ Bot created successfully")
        
        # Send test message
        await bot.send_notification(
            "🎉 *Test Successful!*\n\n"
            "Paper Trading Telegram bot is working correctly.\n"
            "You can now use the paper trading system with Telegram notifications!\n\n"
            "Try these commands:\n"
            "/start - Initialize bot\n"
            "/help - Show all commands",
            "success"
        )
        
        print("✅ Test message sent!")
        print("\nCheck your Telegram for the message.")
        print("If you received it, the bot is working correctly!")
        
        # Close bot
        await bot.bot.close_session()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPossible issues:")
        print("1. Invalid bot token")
        print("2. Wrong chat ID")
        print("3. Bot not started (send /start to your bot first)")

if __name__ == "__main__":
    asyncio.run(test_bot())