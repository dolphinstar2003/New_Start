#!/usr/bin/env python3
"""
Paper Trading Demo Mode
Runs paper trading with simulated market data for testing
"""
import asyncio
import random
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from paper_trading_module import PaperTradingModule
from config.settings import SACRED_SYMBOLS


class DemoPaperTrader(PaperTradingModule):
    """Paper trader with simulated market data"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_mode = True
        
    async def initialize(self):
        """Initialize in demo mode"""
        print("🎮 Running in DEMO MODE - Using simulated market data")
        
        # Skip AlgoLab authentication
        self.algolab_api = None
        self.algolab_socket = None
        self.auth = None
        
        # Initialize Telegram if available
        try:
            from telegram_integration import TelegramBot
            self.telegram_bot = TelegramBot(self)
            asyncio.create_task(self.telegram_bot.start())
            print("✅ Telegram bot initialized")
        except Exception as e:
            print(f"⚠️  Telegram bot initialization failed: {e}")
            self.telegram_bot = None
            self.telegram_notifications = False
        
        # Initialize demo market data
        self._init_demo_data()
        
        return True
    
    def _init_demo_data(self):
        """Initialize demo market data"""
        base_prices = {
            'GARAN.IS': 120.0,
            'AKBNK.IS': 60.0,
            'ISCTR.IS': 14.0,
            'YKBNK.IS': 30.0,
            'SAHOL.IS': 95.0,
            'SISE.IS': 35.0,
            'EREGL.IS': 22.0,
            'ASELS.IS': 120.0,
            'TUPRS.IS': 130.0,
            'TCELL.IS': 95.0,
            'MGROS.IS': 550.0,
            'BIMAS.IS': 500.0,
            'PETKM.IS': 17.0,
            'KOZAL.IS': 25.0,
            'ULKER.IS': 110.0,
            'THYAO.IS': 300.0,
            'AKSEN.IS': 35.0,
            'ENKAI.IS': 60.0,
            'KRDMD.IS': 30.0,
            'KCHOL.IS': 140.0
        }
        
        for symbol, base_price in base_prices.items():
            self.market_data[symbol] = {
                'last_price': base_price,
                'open_price': base_price * random.uniform(0.98, 1.02),
                'high_price': base_price * random.uniform(1.01, 1.03),
                'low_price': base_price * random.uniform(0.97, 0.99),
                'volume': random.randint(1000000, 10000000),
                'volume_ma': random.randint(800000, 8000000),
                'price_history': [],
                'last_update': datetime.now()
            }
            
            # Calculate derived values
            self.market_data[symbol]['volume_ratio'] = (
                self.market_data[symbol]['volume'] / 
                self.market_data[symbol]['volume_ma']
            )
            self.market_data[symbol]['price_change_day'] = (
                (self.market_data[symbol]['last_price'] - 
                 self.market_data[symbol]['open_price']) / 
                self.market_data[symbol]['open_price']
            )
            self.market_data[symbol]['price_change_1h'] = random.uniform(-0.02, 0.02)
            
            # Add some technical indicators
            self.indicators[symbol] = {
                'supertrend_trend': random.choice([-1, 1]),
                'supertrend_buy_signal': random.random() > 0.8,
                'rsi': random.uniform(30, 70),
                'adx_di_adx': random.uniform(15, 35),
                'adx_di_plus_di': random.uniform(20, 30),
                'adx_di_minus_di': random.uniform(15, 25)
            }
    
    async def update_market_data_via_api(self):
        """Update demo market data with random movements"""
        for symbol in SACRED_SYMBOLS:
            if symbol in self.market_data:
                # Simulate price movements
                current_price = self.market_data[symbol]['last_price']
                change = random.uniform(-0.02, 0.02)  # ±2% max change
                new_price = current_price * (1 + change)
                
                self.market_data[symbol]['last_price'] = new_price
                self.market_data[symbol]['volume'] += random.randint(-100000, 100000)
                self.market_data[symbol]['last_update'] = datetime.now()
                
                # Update price changes
                self.market_data[symbol]['price_change_day'] = (
                    (new_price - self.market_data[symbol]['open_price']) / 
                    self.market_data[symbol]['open_price']
                )
                self.market_data[symbol]['price_change_1h'] = change
                
                # Update some indicators randomly
                self.indicators[symbol]['rsi'] += random.uniform(-5, 5)
                self.indicators[symbol]['rsi'] = max(0, min(100, self.indicators[symbol]['rsi']))
                
                # Randomly generate signals
                if random.random() > 0.95:
                    self.indicators[symbol]['supertrend_buy_signal'] = True
                    self.indicators[symbol]['supertrend_trend'] = 1
                elif random.random() < 0.05:
                    self.indicators[symbol]['supertrend_buy_signal'] = False
                    self.indicators[symbol]['supertrend_trend'] = -1
                
                # Update current prices
                self.portfolio['current_prices'][symbol] = new_price
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Demo market data updated")


async def main():
    """Run demo paper trading"""
    print("\n" + "="*80)
    print("🎮 PAPER TRADING DEMO MODE")
    print("="*80)
    print("Running with simulated market data for testing")
    print("="*80)
    
    # Create demo paper trader
    trader = DemoPaperTrader(initial_capital=100000)
    
    # Load previous state if exists
    if trader.load_state():
        print("✅ Previous state loaded")
    else:
        print("ℹ️  Starting with fresh portfolio")
    
    # Initialize
    success = await trader.initialize()
    if not success:
        print("❌ Failed to initialize")
        return
    
    # Configure trading
    trader.auto_trade_enabled = True
    trader.require_confirmation = False  # No confirmations in demo
    trader.telegram_notifications = True
    
    # Show initial status
    status = trader.get_portfolio_status()
    print(f"\n📊 Initial Portfolio:")
    print(f"   Value: ${status['portfolio_value']:,.2f}")
    print(f"   Cash: ${status['cash']:,.2f}")
    print(f"   Positions: {status['num_positions']}")
    
    if trader.telegram_bot:
        await trader.telegram_bot.send_notification(
            "🎮 *Demo Mode Started*\n\n"
            "Paper trading with simulated data\n"
            "Auto trading: ✅ Enabled\n"
            "Confirmations: ❌ Disabled (demo)\n\n"
            "Good luck! 📈",
            "info"
        )
    
    print("\n🚀 Starting demo trading...")
    print("   - Simulated market data")
    print("   - Automatic trading enabled")
    print("   - No trade confirmations")
    print("\nPress Ctrl+C to stop")
    print("="*80)
    
    # Override the trading loop to always run (ignore market hours)
    async def demo_trading_loop():
        trader.is_running = True
        check_interval = 30  # Check every 30 seconds in demo
        last_check = datetime.now()
        last_api_update = datetime.now()
        
        while trader.is_running:
            try:
                current_time = datetime.now()
                
                # Update demo market data every 10 seconds
                if (current_time - last_api_update).seconds >= 10:
                    await trader.update_market_data_via_api()
                    last_api_update = current_time
                
                # Check positions every interval
                if (current_time - last_check).seconds >= check_interval:
                    if trader.auto_trade_enabled:
                        await trader.check_positions_for_exit()
                        await trader.check_for_rotation()
                        await trader.check_for_new_entries()
                    
                    await trader.update_portfolio_value()
                    last_check = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in demo loop: {e}")
                await asyncio.sleep(5)
    
    # Start demo trading loop
    trading_task = asyncio.create_task(demo_trading_loop())
    
    try:
        while True:
            await asyncio.sleep(60)
            
            # Print status every minute
            status = trader.get_portfolio_status()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Value: ${status['portfolio_value']:,.2f} | "
                  f"Return: {status['total_return_pct']:+.2f}% | "
                  f"Positions: {status['num_positions']}")
    
    except KeyboardInterrupt:
        print("\n\nStopping demo...")
    
    finally:
        await trader.stop()
        trading_task.cancel()
        
        # Final summary
        print("\n" + "="*80)
        print("DEMO SESSION SUMMARY")
        print("="*80)
        
        status = trader.get_portfolio_status()
        print(f"Final Value: ${status['portfolio_value']:,.2f}")
        print(f"Total Return: {status['total_return_pct']:.2f}%")
        print(f"Total Trades: {status['total_trades']}")
        
        if trader.telegram_bot:
            await trader.telegram_bot.send_notification(
                f"🎮 *Demo Session Ended*\n\n"
                f"Final Value: ${status['portfolio_value']:,.2f}\n"
                f"Return: {status['total_return_pct']:+.2f}%\n"
                f"Trades: {status['total_trades']}",
                "info"
            )
        
        print("="*80)


if __name__ == "__main__":
    asyncio.run(main())