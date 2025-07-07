#!/usr/bin/env python3
"""
Start Enhanced Paper Trading V2 with Telegram Bot
Gerçekçi limit order simülasyonu + Telegram kontrolü
"""
import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from paper_trading_module_v2 import EnhancedPaperTradingModule

def print_banner():
    """Print startup banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          ENHANCED PAPER TRADING V2 + TELEGRAM BOT            ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  • Gerçekçi Limit Order Simülasyonu                         ║
    ║  • Order Book Depth & Market Microstructure                 ║  
    ║  • Slippage, Commission, Partial Fills                      ║
    ║  • AlgoLab Gerçek Zamanlı Veri                             ║
    ║  • Telegram Bot ile Tam Kontrol                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

async def main():
    """Main entry point"""
    print_banner()
    
    # Initialize enhanced paper trading
    logger.info("Starting Enhanced Paper Trading V2...")
    paper_trader = EnhancedPaperTradingModule(initial_capital=100000)
    
    # Load previous state if exists
    if paper_trader.load_state():
        logger.info("✓ Previous state loaded successfully")
        print("✓ Önceki durum yüklendi")
    else:
        logger.info("! Starting with fresh portfolio")
        print("! Yeni portfolio ile başlanıyor")
    
    # Initialize connections
    print("\n🔌 Bağlantılar kuruluyor...")
    success = await paper_trader.initialize()
    
    if not success:
        logger.error("Failed to initialize system")
        print("\n❌ Sistem başlatılamadı!")
        return
    
    print("✓ AlgoLab bağlantısı kuruldu")
    
    # Check Telegram status
    if paper_trader.telegram_bot:
        print("✓ Telegram bot aktif")
        print("\n📱 Telegram'dan /help yazarak komutları görebilirsiniz")
    else:
        print("! Telegram bot aktif değil")
    
    # Enable auto trading by default
    paper_trader.auto_trade_enabled = True
    print("\n🤖 Otomatik trading aktif")
    
    print("\n" + "="*60)
    print("Sistem hazır! Trading loop başlatılıyor...")
    print("Durdurmak için Ctrl+C")
    print("="*60 + "\n")
    
    # Start trading loop
    try:
        await paper_trader.trading_loop()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        print("\n\n⏹️  Sistem kapatılıyor...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Beklenmeyen hata: {e}")
    finally:
        # Clean shutdown
        await paper_trader.stop()
        
        # Print final summary
        print("\n" + "="*60)
        print("KAPANIŞ ÖZETİ")
        print("="*60)
        
        status = paper_trader.get_portfolio_status()
        metrics = paper_trader.get_execution_metrics()
        
        print(f"Portfolio Değeri: ${status['portfolio_value']:,.2f}")
        print(f"Toplam Getiri: %{status['total_return_pct']:.2f}")
        print(f"Toplam İşlem: {status['total_trades']}")
        print(f"Açık Pozisyon: {status['num_positions']}")
        print(f"Aktif Emir: {status['active_orders']}")
        
        if metrics:
            print(f"\nEmir Dolum Oranı: %{metrics.get('fill_rate', 0)*100:.1f}")
            print(f"Ortalama Slippage: {metrics.get('avg_slippage_bps', 0):.1f} bps")
        
        print("\n✓ Sistem başarıyla kapatıldı")


if __name__ == "__main__":
    # Windows color support
    if sys.platform == "win32":
        import os
        os.system("color")
    
    # Configure logging
    logger.remove()
    logger.add(
        "logs/paper_trading_v2.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    # Run
    asyncio.run(main())