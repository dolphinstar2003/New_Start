#!/usr/bin/env python3
"""
Paper Trading Başlatıcı
Kolay kullanım için basitleştirilmiş versiyon
"""
import os
import json
import asyncio
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

from paper_trading_module import PaperTradingModule
from telegram_integration import TELEGRAM_CONFIG

# Telegram config güncelle
if config_file.exists():
    TELEGRAM_CONFIG.update(config)


async def basit_paper_trading():
    """Basit paper trading başlatıcı"""
    
    print("\n" + "="*60)
    print("📈 PAPER TRADING SİSTEMİ")
    print("="*60)
    
    # Paper trader oluştur
    trader = PaperTradingModule(initial_capital=100000)
    
    # Önceki durumu yükle
    if trader.load_state():
        print("✅ Önceki durum yüklendi")
    else:
        print("ℹ️  Yeni portföy oluşturuldu (100,000 $)")
    
    print("\n🔌 Bağlantılar kuruluyor...")
    
    # Sistemi başlat
    try:
        success = await trader.initialize()
        if not success:
            print("❌ Sistem başlatılamadı!")
            return
    except Exception as e:
        print(f"❌ Hata: {e}")
        return
    
    print("✅ Sistem hazır!")
    
    # Otomatik trading'i başlat
    trader.auto_trade_enabled = True
    trader.require_confirmation = True  # Telegram onayı gerekli
    trader.telegram_notifications = True
    
    # Mevcut durumu göster
    status = trader.get_portfolio_status()
    print(f"\n💼 PORTFÖY DURUMU:")
    print(f"   Değer: ${status['portfolio_value']:,.2f}")
    print(f"   Nakit: ${status['cash']:,.2f}")
    print(f"   Pozisyon: {status['num_positions']}")
    print(f"   Getiri: %{status['total_return_pct']:.2f}")
    
    # Telegram bildirimi gönder
    if trader.telegram_bot:
        await trader.telegram_bot.send_notification(
            "🚀 Paper Trading Başladı!\n\n"
            f"💰 Başlangıç: ${trader.portfolio['initial_capital']:,.2f}\n"
            f"📊 Mevcut: ${status['portfolio_value']:,.2f}\n"
            f"📈 Getiri: %{status['total_return_pct']:+.2f}\n"
            f"🎯 Pozisyon: {status['num_positions']}/10\n\n"
            "Komutlar için /help yazın",
            "success"
        )
        print("\n📱 Telegram bot aktif!")
        print("   Telegram'da /help yazarak komutları görebilirsiniz")
    
    print("\n" + "="*60)
    print("🤖 OTOMATİK TİCARET BAŞLADI")
    print("="*60)
    print("• Max 10 pozisyon, %20-30 pozisyon büyüklüğü")
    print("• %3 stop loss, %8 kar al")
    print("• Her işlem Telegram'dan onay bekleyecek")
    print("• Durdurmak için Ctrl+C")
    print("="*60)
    
    # Trading döngüsünü başlat
    trading_task = asyncio.create_task(trader.trading_loop())
    
    try:
        # Sürekli çalış
        while True:
            await asyncio.sleep(300)  # 5 dakikada bir kontrol
            
            # Periyodik durum göster
            if trader.is_running:
                status = trader.get_portfolio_status()
                saat = datetime.now().strftime('%H:%M')
                
                print(f"\n[{saat}] Değer: ${status['portfolio_value']:,.2f} | "
                      f"Getiri: %{status['total_return_pct']:+.2f} | "
                      f"Pozisyon: {status['num_positions']}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Durduruluyor...")
    
    finally:
        # Sistemi durdur
        await trader.stop()
        trading_task.cancel()
        
        # Son özet
        print("\n" + "="*60)
        print("📊 ÖZET")
        print("="*60)
        
        status = trader.get_portfolio_status()
        metrics = trader.get_performance_metrics()
        
        print(f"Son Değer: ${status['portfolio_value']:,.2f}")
        print(f"Toplam Getiri: %{status['total_return_pct']:.2f}")
        print(f"İşlem Sayısı: {status['total_trades']}")
        
        if metrics:
            print(f"Kazanma Oranı: %{metrics['win_rate']:.1f}")
            print(f"Sharpe Oranı: {metrics['sharpe_ratio']:.2f}")
        
        print("="*60)
        print("\n✅ Paper trading durduruldu. Durum kaydedildi.")


if __name__ == "__main__":
    # Piyasa saatlerini kontrol et
    saat = datetime.now().hour
    gun = datetime.now().weekday()
    
    if not (10 <= saat < 18 and gun < 5):
        print("\n⚠️  UYARI: Piyasa kapalı!")
        print("   Piyasa saatleri: Pazartesi-Cuma 10:00-18:00")
        print("   Sistem çalışacak ama piyasa açılana kadar işlem yapılmayacak.")
        
        devam = input("\nYine de devam etmek istiyor musunuz? (e/h): ")
        if devam.lower() != 'e':
            print("İptal edildi.")
            exit()
    
    # Paper trading'i başlat
    asyncio.run(basit_paper_trading())