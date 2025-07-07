#!/usr/bin/env python3
"""
Test Telegram Bot Commands
"""
import os
import json
import asyncio
from pathlib import Path

# Load config
config_file = Path(__file__).parent / 'telegram_config.json'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
    os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')

import sys
sys.path.append(str(Path(__file__).parent))

from telegram_bot_advanced import AdvancedTelegramBot


async def test_commands():
    """Test telegram commands"""
    print("Starting Telegram bot test...")
    
    # Create bot
    bot = AdvancedTelegramBot()
    
    print("Bot created, starting...")
    print(f"Token: {bot.bot_token[:20]}...")
    print(f"Chat ID: {bot.chat_id}")
    
    # Send test message
    await bot.send_notification(
        "🧪 *Bot Test Started*\n\n"
        "Testing command handlers...\n"
        "Try these commands:\n"
        "/help - Show all commands\n"
        "/status - Show status",
        "info"
    )
    
    print("\n✅ Bot is running!")
    print("Try sending commands in Telegram")
    print("Press Ctrl+C to stop\n")
    
    # Start bot
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(test_commands())