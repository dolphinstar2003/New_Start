#!/usr/bin/env python3
"""
Full Control Paper Trading System
Tüm sistem kontrolü Telegram üzerinden
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


async def full_control_trading():
    """Full control paper trading with Telegram"""
    
    print("\n" + "="*60)
    print("🎮 FULL CONTROL PAPER TRADING SYSTEM")
    print("="*60)
    print("Telegram üzerinden tam kontrol!")
    print("="*60)
    
    # Paper trader oluştur
    trader = PaperTradingModule(initial_capital=100000)
    
    # Önceki durumu yükle
    if trader.load_state():
        print("✅ Önceki durum yüklendi")
    else:
        print("ℹ️  Yeni portföy oluşturuldu (100,000 $)")
    
    print("\n🔌 Sistem başlatılıyor...")
    
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
    
    # Otomatik trading ayarları
    trader.auto_trade_enabled = True
    trader.require_confirmation = True  # Telegram onayı
    trader.telegram_notifications = True
    
    # Mevcut durumu göster
    status = trader.get_portfolio_status()
    print(f"\n💼 PORTFÖY DURUMU:")
    print(f"   Değer: ${status['portfolio_value']:,.2f}")
    print(f"   Nakit: ${status['cash']:,.2f}")
    print(f"   Pozisyon: {status['num_positions']}")
    print(f"   Getiri: %{status['total_return_pct']:.2f}")
    
    print("\n" + "="*60)
    print("📱 TELEGRAM FULL CONTROL AKTİF!")
    print("="*60)
    print("\nKullanılabilir komutlar:")
    print("\n📊 Trading:")
    print("  /status, /positions, /trades, /performance")
    print("  /start_trading, /stop_trading, /force_check")
    print("\n🎮 Demo & Test:")
    print("  /start_demo, /stop_demo, /demo_status")
    print("\n🧠 Analiz:")
    print("  /train - ML model eğitimi")
    print("  /walkforward - Walkforward analizi")
    print("  /backtest [gün] - Backtest çalıştır")
    print("  /optimize - Parametre optimizasyonu")
    print("\n💻 Sistem:")
    print("  /system_info, /logs, /restart, /shutdown")
    print("\n📄 Raporlar:")
    print("  /download_report, /export_trades")
    print("\n⚙️ Ayarlar:")
    print("  /set_param [isim] [değer]")
    print("  /help - Tüm komutlar")
    print("\nDurdurmak için Ctrl+C")
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
        print("\n✅ Full control trading durduruldu. Durum kaydedildi.")


if __name__ == "__main__":
    print("\n🎮 FULL CONTROL TRADING SYSTEM")
    print("Telegram'dan /help yazarak başlayın!")
    
    # Paper trading'i başlat
    asyncio.run(full_control_trading())