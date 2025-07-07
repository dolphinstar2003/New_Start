#!/usr/bin/env python3
"""
Run Paper Trading with Telegram Integration
"""
import sys
import asyncio
import threading
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from paper_trader import PaperTrader
from telegram_bot import PaperTradingBot, run_bot
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_telegram_bot(trader):
    """Run Telegram bot in separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_bot(trader))
    except Exception as e:
        logger.error(f"Telegram bot error: {e}")


def main():
    """Main function"""
    print("\n" + "="*80)
    print("🚀 PAPER TRADING WITH TELEGRAM INTEGRATION")
    print("="*80)
    
    # Check if telegram config exists
    config_file = Path(__file__).parent.parent / 'telegram_config.json'
    if not config_file.exists():
        print("\n⚠️  Telegram configuration not found!")
        print("Run 'python telegram_bot.py' to configure Telegram first.")
        print("\nRunning without Telegram integration...")
        
        # Run without Telegram
        trader = PaperTrader(enable_telegram=False)
        trader.run_continuous()
    else:
        print("\n✅ Telegram configuration found!")
        print("Starting paper trading with Telegram notifications...")
        
        # Create trader with Telegram enabled
        trader = PaperTrader(enable_telegram=True)
        
        # Start Telegram bot in separate thread
        telegram_thread = threading.Thread(
            target=run_telegram_bot,
            args=(trader,),
            daemon=True
        )
        telegram_thread.start()
        
        # Run paper trader
        try:
            trader.run_continuous()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            logger.info("Paper trading stopped by user")


if __name__ == "__main__":
    main()