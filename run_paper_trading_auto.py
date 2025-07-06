#!/usr/bin/env python3
"""
Paper Trading Auto Runner
Runs paper trading in automatic mode without user interaction
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent))

from paper_trading_module import PaperTradingModule
from telegram_integration import TELEGRAM_CONFIG


async def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🚀 DYNAMIC PORTFOLIO OPTIMIZER - PAPER TRADING (AUTO MODE)")
    print("="*80)
    print("Starting in automatic mode with trade confirmations via Telegram")
    print("="*80)
    
    # Load Telegram config if exists
    telegram_config_file = Path(__file__).parent / 'telegram_config.json'
    if telegram_config_file.exists():
        with open(telegram_config_file, 'r') as f:
            config = json.load(f)
            TELEGRAM_CONFIG.update(config)
            # Set environment variables
            os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
            os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')
    
    # Create paper trader
    paper_trader = PaperTradingModule(initial_capital=100000)
    
    # Try to load previous state
    if paper_trader.load_state():
        print("✅ Previous state loaded")
    else:
        print("ℹ️  Starting with fresh portfolio")
    
    # Initialize system
    print("\n🔌 Initializing connections...")
    success = await paper_trader.initialize()
    
    if not success:
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully!")
    
    # Configure for automatic trading
    paper_trader.auto_trade_enabled = True
    paper_trader.require_confirmation = True  # Require Telegram confirmations
    paper_trader.telegram_notifications = True
    
    # Show initial status
    status = paper_trader.get_portfolio_status()
    print(f"\n📊 Portfolio Status:")
    print(f"   Value: ${status['portfolio_value']:,.2f}")
    print(f"   Cash: ${status['cash']:,.2f}")
    print(f"   Positions: {status['num_positions']}")
    print(f"   Total Return: ${status['total_return']:,.2f} ({status['total_return_pct']:.2f}%)")
    
    # Send startup notification
    if paper_trader.telegram_bot:
        await paper_trader.telegram_bot.send_notification(
            "🚀 *Paper Trading Started (Auto Mode)*\n\n"
            "✅ Auto trading: Enabled\n"
            "✅ Confirmations: Required\n"
            "✅ Notifications: Enabled\n\n"
            "Send /help for available commands\n"
            "Good luck! 📈",
            "success"
        )
        print(f"\n📱 Telegram Bot Active - Check your Telegram for notifications")
    
    print("\n📊 Starting automatic trading...")
    print("   - Monitoring all 20 symbols")
    print("   - Max 10 positions with 20-30% sizing")
    print("   - 3% stop loss, 8% take profit")
    print("   - Portfolio rotation every 2 days")
    print("\nPress Ctrl+C to stop")
    print("="*80)
    
    # Start trading loop
    trading_task = asyncio.create_task(paper_trader.trading_loop())
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            # Print periodic status
            if paper_trader.is_running:
                status = paper_trader.get_portfolio_status()
                current_time = datetime.now()
                
                print(f"\n[{current_time.strftime('%H:%M:%S')}] "
                      f"Value: ${status['portfolio_value']:,.2f} | "
                      f"Return: {status['total_return_pct']:+.2f}% | "
                      f"Positions: {status['num_positions']} | "
                      f"Trading: {'ON' if paper_trader.auto_trade_enabled else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Received shutdown signal...")
    
    finally:
        # Stop paper trading
        print("📝 Saving state...")
        await paper_trader.stop()
        trading_task.cancel()
        
        # Final summary
        print("\n" + "="*80)
        print("📊 TRADING SESSION SUMMARY")
        print("="*80)
        
        status = paper_trader.get_portfolio_status()
        print(f"Final Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Total Return: ${status['total_return']:,.2f} ({status['total_return_pct']:.2f}%)")
        print(f"Total Trades: {status['total_trades']}")
        print(f"Win Rate: {status['win_rate']:.1f}%")
        
        metrics = paper_trader.get_performance_metrics()
        if metrics:
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Send shutdown notification
        if paper_trader.telegram_bot:
            await paper_trader.telegram_bot.send_notification(
                f"🛑 *Paper Trading Stopped*\n\n"
                f"📊 Session Summary:\n"
                f"• Final Value: ${status['portfolio_value']:,.2f}\n"
                f"• Return: {status['total_return_pct']:+.2f}%\n"
                f"• Trades: {status['total_trades']}\n"
                f"• Win Rate: {status['win_rate']:.1f}%\n\n"
                f"State saved successfully ✅",
                "info"
            )
        
        print("="*80)
        print("\n✅ Paper trading session ended. State saved.")


if __name__ == "__main__":
    # Check if running during market hours
    current_time = datetime.now()
    if not (10 <= current_time.hour < 18 and current_time.weekday() < 5):
        print("\n⚠️  WARNING: Market is currently closed!")
        print("   Trading hours: Mon-Fri 10:00-18:00 Istanbul time")
        print("   The system will run but no trades will be executed until market opens.")
        input("\nPress Enter to continue anyway...")
    
    # Run the main function
    asyncio.run(main())