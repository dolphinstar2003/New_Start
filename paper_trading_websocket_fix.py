#!/usr/bin/env python3
"""
Paper Trading - WebSocket Hatasız Versiyon
Sadece API kullanarak çalışır
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


async def paper_trading_no_websocket():
    """WebSocket kullanmadan paper trading"""
    
    print("\n" + "="*60)
    print("📈 PAPER TRADING SİSTEMİ (WebSocket Kapalı)")
    print("="*60)
    
    # Paper trader oluştur
    trader = PaperTradingModule(initial_capital=100000)
    
    # WebSocket'i devre dışı bırak
    trader.use_websocket = False
    
    # Önceki durumu yükle
    if trader.load_state():
        print("✅ Önceki durum yüklendi")
    else:
        print("ℹ️  Yeni portföy oluşturuldu (100,000 $)")
    
    print("\n🔌 API bağlantısı kuruluyor...")
    
    # Sistemi başlat (WebSocket olmadan)
    try:
        # Sadece API ve Telegram'ı başlat
        trader.algolab_api = await trader._initialize_algolab_api()
        if trader.algolab_api:
            print("✅ AlgoLab API bağlantısı kuruldu")
        
        # Telegram bot başlat
        await trader._initialize_telegram()
        if trader.telegram_bot:
            print("✅ Telegram bot başlatıldı")
        
        trader.is_running = True
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return
    
    print("✅ Sistem hazır! (API-only mod)")
    
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
        from utils.telegram_utils import escape_markdown_v1
        await trader.telegram_bot.send_notification(
            f"🚀 Paper Trading Başladı\\! \\(WebSocket Kapalı\\)\n\n"
            f"💰 Başlangıç: \\${trader.portfolio['initial_capital']:,.2f}\n"
            f"📊 Mevcut: \\${status['portfolio_value']:,.2f}\n"
            f"📈 Getiri: %{status['total_return_pct']:+.2f}\n"
            f"🎯 Pozisyon: {status['num_positions']}/10\n\n"
            f"⚠️ WebSocket kapalı \\- sadece API kullanılıyor\n"
            f"Komutlar için /help yazın",
            "success"
        )
        print("\n📱 Telegram bot aktif!")
    
    print("\n" + "="*60)
    print("🤖 OTOMATİK TİCARET BAŞLADI (API-ONLY)")
    print("="*60)
    print("• WebSocket KAPALI - Sadece API kullanılıyor")
    print("• Her 30 saniyede bir fiyat güncellemesi")
    print("• Durdurmak için Ctrl+C")
    print("="*60)
    
    # Trading döngüsü (WebSocket olmadan)
    last_update = datetime.now()
    last_check = datetime.now()
    
    try:
        while trader.is_running:
            now = datetime.now()
            
            # Her 30 saniyede fiyat güncelle (API ile)
            if (now - last_update).seconds >= 30:
                print(f"\n[{now.strftime('%H:%M:%S')}] Fiyatlar güncelleniyor...")
                await trader.update_market_data_via_api()
                last_update = now
            
            # Her 60 saniyede işlem kontrolü
            if (now - last_check).seconds >= 60:
                if trader.auto_trade_enabled:
                    await trader.check_positions_for_exit()
                    await trader.check_for_rotation()
                    await trader.check_for_new_entries()
                
                await trader.update_portfolio_value()
                
                # Durum göster
                status = trader.get_portfolio_status()
                print(f"[{now.strftime('%H:%M:%S')}] "
                      f"Değer: ${status['portfolio_value']:,.0f} "
                      f"(%{status['total_return_pct']:+.1f}) "
                      f"Poz: {status['num_positions']}")
                
                last_check = now
            
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Durduruluyor...")
    
    finally:
        # Sistemi durdur
        trader.is_running = False
        if trader.telegram_bot:
            await trader.telegram_bot.stop()
        
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
        
        # Durumu kaydet
        trader.save_state()
        
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
    
    # Paper trading'i başlat (WebSocket olmadan)
    asyncio.run(paper_trading_no_websocket())