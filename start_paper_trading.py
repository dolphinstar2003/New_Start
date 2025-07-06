#!/usr/bin/env python3
"""
Paper Trading Launcher with Telegram Integration
Easy setup and launch for the paper trading system
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
from telegram_integration import setup_telegram_config, TELEGRAM_CONFIG


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*80)
    print("🚀 DYNAMIC PORTFOLIO OPTIMIZER - PAPER TRADING SYSTEM")
    print("="*80)
    print("Version: 1.0 - AlgoLab Data + Telegram Integration")
    print("Strategy: Up to 10 positions, 20-30% sizing, 3% stop loss")
    print("="*80)


def check_requirements():
    """Check if all requirements are met"""
    print("\n📋 Checking requirements...")
    
    # Check AlgoLab auth
    try:
        from utils.algolab_auth import AlgoLabAuth
        auth = AlgoLabAuth()
        if auth.api_key:
            print("✅ AlgoLab API key found")
        else:
            print("❌ AlgoLab API key not found")
            print("   Please set ALGOLAB_API_KEY environment variable")
            return False
    except Exception as e:
        print(f"❌ AlgoLab auth check failed: {e}")
        return False
    
    # Check Telegram config
    telegram_config_file = Path(__file__).parent / 'telegram_config.json'
    if telegram_config_file.exists():
        print("✅ Telegram configuration found")
    else:
        print("⚠️  Telegram configuration not found")
        setup_telegram = input("\nWould you like to setup Telegram now? (y/n): ").lower()
        if setup_telegram == 'y':
            config = setup_telegram_config()
            TELEGRAM_CONFIG.update(config)
        else:
            print("⚠️  Continuing without Telegram integration")
    
    # Check data directory
    from config.settings import DATA_DIR
    if DATA_DIR.exists():
        print("✅ Data directory found")
    else:
        print("❌ Data directory not found")
        return False
    
    print("\n✅ All requirements met!")
    return True


async def main():
    """Main entry point"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please fix the requirements and try again.")
        return
    
    # Load Telegram config if exists
    telegram_config_file = Path(__file__).parent / 'telegram_config.json'
    if telegram_config_file.exists():
        with open(telegram_config_file, 'r') as f:
            config = json.load(f)
            TELEGRAM_CONFIG.update(config)
            # Set environment variables
            os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
            os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')
    
    print("\n🚀 Starting Paper Trading System...")
    
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
    
    # Show initial status
    status = paper_trader.get_portfolio_status()
    print(f"\n📊 Initial Portfolio Status:")
    print(f"   Value: ${status['portfolio_value']:,.2f}")
    print(f"   Cash: ${status['cash']:,.2f}")
    print(f"   Positions: {status['num_positions']}")
    print(f"   Total Return: ${status['total_return']:,.2f} ({status['total_return_pct']:.2f}%)")
    
    # Trading options
    print("\n" + "="*80)
    print("TRADING OPTIONS")
    print("="*80)
    print("1. Start with AUTO TRADING (recommended)")
    print("2. Start in MANUAL mode (monitor only)")
    print("3. Configure settings")
    print("4. Exit")
    print("="*80)
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        paper_trader.auto_trade_enabled = True
        paper_trader.require_confirmation = True
        print("\n✅ Auto trading ENABLED with trade confirmations")
        if paper_trader.telegram_bot:
            await paper_trader.telegram_bot.send_notification(
                "🚀 Paper Trading Started\n\n"
                "Auto trading: ✅ Enabled\n"
                "Confirmations: ✅ Required\n\n"
                "Good luck! 📈",
                "success"
            )
    
    elif choice == "2":
        paper_trader.auto_trade_enabled = False
        print("\n⚠️  Auto trading DISABLED - monitoring mode only")
        if paper_trader.telegram_bot:
            await paper_trader.telegram_bot.send_notification(
                "👁️ Paper Trading Started\n\n"
                "Auto trading: ❌ Disabled\n"
                "Mode: Monitoring only\n\n"
                "Use /start_trading to enable",
                "info"
            )
    
    elif choice == "3":
        print("\n⚙️  Settings Configuration")
        print("1. Telegram notifications:", "Enabled" if paper_trader.telegram_notifications else "Disabled")
        print("2. Trade confirmations:", "Required" if paper_trader.require_confirmation else "Not required")
        print("3. Max positions:", paper_trader.PORTFOLIO_PARAMS['max_positions'])
        print("4. Stop loss:", f"{paper_trader.PORTFOLIO_PARAMS['stop_loss']*100:.0f}%")
        print("5. Take profit:", f"{paper_trader.PORTFOLIO_PARAMS['take_profit']*100:.0f}%")
        
        modify = input("\nModify settings? (y/n): ").lower()
        if modify == 'y':
            paper_trader.telegram_notifications = input("Enable Telegram notifications? (y/n): ").lower() == 'y'
            paper_trader.require_confirmation = input("Require trade confirmations? (y/n): ").lower() == 'y'
            print("✅ Settings updated")
        
        # Ask again for trading mode
        start_trading = input("\nStart auto trading? (y/n): ").lower()
        paper_trader.auto_trade_enabled = (start_trading == 'y')
    
    elif choice == "4":
        print("Exiting...")
        return
    
    # Start trading loop
    print("\n" + "="*80)
    print("💼 PAPER TRADING ACTIVE")
    print("="*80)
    print("Commands:")
    print("  /status  - Show portfolio status (Telegram)")
    print("  /help    - Show all commands (Telegram)")
    print("  Ctrl+C   - Stop and exit")
    print("="*80)
    
    if paper_trader.telegram_bot:
        print(f"\n📱 Telegram Bot: @{paper_trader.telegram_bot.bot._token.split(':')[0]}")
        print("   Send /start to your bot to begin")
    
    print("\n📊 Monitoring markets...")
    
    # Start trading loop
    trading_task = asyncio.create_task(paper_trader.trading_loop())
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(60)
            
            # Print periodic status
            if paper_trader.is_running:
                status = paper_trader.get_portfolio_status()
                current_time = datetime.now()
                
                # Print to console every 30 minutes
                if current_time.minute in [0, 30]:
                    print(f"\n[{current_time.strftime('%H:%M')}] "
                          f"Value: ${status['portfolio_value']:,.2f} | "
                          f"Return: {status['total_return_pct']:+.2f}% | "
                          f"Positions: {status['num_positions']}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Shutting down...")
    
    finally:
        # Stop paper trading
        await paper_trader.stop()
        trading_task.cancel()
        
        # Final summary
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY")
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
        
        print("="*80)
        print("\n✅ Paper trading session ended. State saved automatically.")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())