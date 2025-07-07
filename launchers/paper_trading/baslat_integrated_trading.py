#!/usr/bin/env python3
"""
Integrated Trading System Başlatıcı
Tüm modüllere erişimli Telegram bot ile
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


async def integrated_trading():
    """Integrated paper trading with full system control"""
    
    print("\n" + "="*70)
    print("🚀 BIST100 INTEGRATED TRADING SYSTEM")
    print("="*70)
    print("Full modüler sistem kontrolü - Telegram üzerinden")
    print("="*70)
    
    # Paper trader oluştur
    trader = PaperTradingModule(initial_capital=100000)
    
    # Önceki durumu yükle
    if trader.load_state():
        print("✅ Önceki portföy durumu yüklendi")
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
    print(f"   Pozisyon: {status['num_positions']}/10")
    print(f"   Getiri: %{status['total_return_pct']:+.2f}")
    
    print("\n" + "="*70)
    print("📱 TELEGRAM KOMUTLARI AKTİF!")
    print("="*70)
    print("\n🎮 Temel Komutlar:")
    print("  /help - Tüm komutları detaylı göster")
    print("  /status - Detaylı portföy durumu")
    print("  /positions - Açık pozisyonlar")
    print("  /backtest 30 - 30 günlük gerçek backtest")
    print("  /train - ML modellerini eğit")
    print("  /market_overview - Piyasa genel durumu")
    print("\n📊 Analiz Komutları:")
    print("  /analyze GARAN - Hisse analizi")
    print("  /opportunities - En iyi fırsatlar")
    print("  /walkforward - 6 aylık walkforward")
    print("  /optimize - Parametre optimizasyonu")
    print("\n💻 Sistem Komutları:")
    print("  /system_info - CPU, RAM, Disk durumu")
    print("  /data_status - Veri durumu kontrolü")
    print("  /logs 50 - Son 50 satır log")
    print("\n📄 Raporlar:")
    print("  /daily_report - Günlük rapor")
    print("  /export_trades csv - İşlemleri dışa aktar")
    print("\n⚙️ Ayarlar:")
    print("  /set_param stop_loss 0.05 - Parametre ayarla")
    print("  /get_params - Mevcut parametreler")
    print("\nDurdurmak için Ctrl+C")
    print("="*70)
    
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
        print("\n\n⏹️  Sistem durduruluyor...")
    
    finally:
        # Sistemi durdur
        await trader.stop()
        trading_task.cancel()
        
        # Son özet
        print("\n" + "="*70)
        print("📊 KAPANIŞ ÖZETİ")
        print("="*70)
        
        status = trader.get_portfolio_status()
        metrics = trader.get_performance_metrics()
        
        print(f"Son Değer: ${status['portfolio_value']:,.2f}")
        print(f"Toplam Getiri: %{status['total_return_pct']:.2f}")
        print(f"İşlem Sayısı: {status['total_trades']}")
        
        if metrics:
            print(f"Kazanma Oranı: %{metrics['win_rate']:.1f}")
            print(f"Sharpe Oranı: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: %{metrics['max_drawdown']:.2f}")
        
        print("="*70)
        print("\n✅ Integrated trading sistemi kapatıldı.")


if __name__ == "__main__":
    print("\n🚀 BIST100 INTEGRATED TRADING SYSTEM")
    print("Gerçek modüller, gerçek backtest, gerçek ML!")
    print("\nTelegram'dan /help yazarak başlayın")
    
    # Check market hours
    now = datetime.now()
    if not (10 <= now.hour < 18 and now.weekday() < 5):
        print("\n⚠️  DİKKAT: Piyasa kapalı!")
        print("   Sistem çalışacak ama işlem yapılmayacak")
    
    # Start integrated trading
    asyncio.run(integrated_trading())