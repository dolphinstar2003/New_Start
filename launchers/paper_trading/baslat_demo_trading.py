#!/usr/bin/env python3
"""
Demo Paper Trading Başlatıcı
Piyasa kapalıyken test için sahte veriyle çalışır
"""
import os
import json
import asyncio
import random
from pathlib import Path
from datetime import datetime

# Telegram config'i otomatik yükle
config_file = Path(__file__).parent / 'telegram_config.json'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
    os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')

import sys
sys.path.append(str(Path(__file__).parent))

from paper_trading_demo import DemoPaperTrader
from telegram_integration import TELEGRAM_CONFIG

# Telegram config güncelle
if config_file.exists():
    TELEGRAM_CONFIG.update(config)


async def demo_trading():
    """Demo modda paper trading"""
    
    print("\n" + "="*60)
    print("🎮 DEMO PAPER TRADING")
    print("="*60)
    print("Sahte veriyle test - Gerçek para YOK!")
    print("="*60)
    
    # Demo trader oluştur
    trader = DemoPaperTrader(initial_capital=100000)
    
    # Önceki durumu yükle
    if trader.load_state():
        print("✅ Önceki durum yüklendi")
    else:
        print("ℹ️  Yeni demo portföy (100,000 $)")
    
    # Sistemi başlat
    success = await trader.initialize()
    if not success:
        print("❌ Başlatılamadı!")
        return
    
    print("✅ Demo sistem hazır!")
    
    # Otomatik trading
    trader.auto_trade_enabled = True
    trader.require_confirmation = False  # Demo'da onay yok
    trader.telegram_notifications = True
    
    # Durum göster
    status = trader.get_portfolio_status()
    print(f"\n💼 BAŞLANGIÇ:")
    print(f"   Değer: ${status['portfolio_value']:,.2f}")
    print(f"   Nakit: ${status['cash']:,.2f}")
    
    # Telegram bildirimi
    if trader.telegram_bot:
        await trader.telegram_bot.send_notification(
            "🎮 Demo Trading Başladı!\n\n"
            "• Sahte veriyle test\n"
            "• Otomatik işlem: ✅\n" 
            "• Onay gerektirmez\n\n"
            "/help - Komutları gör",
            "info"
        )
    
    print("\n🚀 Demo trading başlıyor...")
    print("• Her 10 saniyede fiyat güncellenir")
    print("• Rastgele alım/satım sinyalleri")
    print("• Ctrl+C ile durdur")
    print("="*60)
    
    # Demo trading döngüsü
    trader.is_running = True
    last_update = datetime.now()
    last_check = datetime.now()
    
    try:
        while trader.is_running:
            now = datetime.now()
            
            # Her 10 saniyede fiyat güncelle
            if (now - last_update).seconds >= 10:
                await trader.update_market_data_via_api()
                last_update = now
            
            # Her 30 saniyede işlem kontrolü
            if (now - last_check).seconds >= 30:
                if trader.auto_trade_enabled:
                    await trader.check_positions_for_exit()
                    await trader.check_for_rotation()
                    await trader.check_for_new_entries()
                
                await trader.update_portfolio_value()
                
                # Durum göster
                status = trader.get_portfolio_status()
                print(f"\n[{now.strftime('%H:%M:%S')}] "
                      f"${status['portfolio_value']:,.0f} "
                      f"(%{status['total_return_pct']:+.1f}) "
                      f"Poz: {status['num_positions']}")
                
                last_check = now
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nDurduruluyor...")
    
    finally:
        await trader.stop()
        
        # Özet
        status = trader.get_portfolio_status()
        print("\n" + "="*60)
        print("DEMO ÖZET")
        print("="*60)
        print(f"Son Değer: ${status['portfolio_value']:,.2f}")
        print(f"Getiri: %{status['total_return_pct']:.2f}")
        print(f"İşlem: {status['total_trades']}")
        print("="*60)


if __name__ == "__main__":
    print("\n🎮 DEMO MOD - Sahte veriyle test")
    print("Gerçek paper trading için: python baslat_paper_trading.py")
    
    asyncio.run(demo_trading())