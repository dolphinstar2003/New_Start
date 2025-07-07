#!/usr/bin/env python3
"""
Test Telegram Integration for Paper Trading
"""
import sys
import asyncio
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from paper_trader import PaperTrader
from telegram_bot import PaperTradingBot
from data_fetcher import FALLBACK_PRICES
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


async def test_telegram_integration():
    """Test Telegram bot integration"""
    print("\n" + "="*80)
    print("🧪 TESTING TELEGRAM INTEGRATION")
    print("="*80)
    
    # Check config
    config_file = Path(__file__).parent.parent / 'telegram_config.json'
    if not config_file.exists():
        print("\n❌ Telegram configuration not found!")
        print("Create telegram_config.json with:")
        print("{")
        print('  "bot_token": "YOUR_BOT_TOKEN",')
        print('  "chat_id": "YOUR_CHAT_ID",')
        print('  "admin_users": []')
        print("}")
        return
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    if not config.get('bot_token') or not config.get('chat_id'):
        print("\n❌ Invalid Telegram configuration!")
        print("Please fill in bot_token and chat_id in telegram_config.json")
        return
    
    print("\n✅ Telegram configuration found!")
    
    # Create paper trader
    trader = PaperTrader(enable_telegram=False)  # Don't auto-init for testing
    
    # Override with fallback prices for testing
    trader.data_fetcher.get_current_prices = lambda: FALLBACK_PRICES
    trader.check_trading_hours = lambda: True
    
    # Create Telegram bot manually
    bot = PaperTradingBot(trader)
    
    # Test notifications
    print("\n1️⃣ Testing basic notification...")
    await bot.send_notification(
        "*Test Notification* ✅\\n\\nIf you see this, notifications are working!",
        "success"
    )
    
    print("\n2️⃣ Running trading cycle...")
    trader.run_trading_cycle()
    
    # Get portfolio status
    summary_df = trader.portfolio_manager.get_summary()
    
    print("\n3️⃣ Testing portfolio status notification...")
    await bot.send_daily_summary()
    
    print("\n4️⃣ Testing trade notifications...")
    for _, row in summary_df.iterrows():
        portfolio = trader.portfolio_manager.get_portfolio(row['name'])
        if portfolio and portfolio.positions:
            for symbol, pos in portfolio.positions.items():
                await bot.send_trade_notification(
                    row['name'], "BUY", symbol,
                    pos.shares, pos.entry_price, "Test Signal"
                )
                break  # Just test one
            break
    
    print("\n✅ Telegram integration test completed!")
    print("Check your Telegram for the test messages.")


def main():
    """Main function"""
    try:
        asyncio.run(test_telegram_integration())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()