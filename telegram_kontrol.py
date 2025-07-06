#!/usr/bin/env python3
"""
Telegram Bot Kontrol
Bot'un çalışıp çalışmadığını kontrol et
"""
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Config yükle
config_file = Path(__file__).parent / 'telegram_config.json'
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)
    os.environ['TELEGRAM_BOT_TOKEN'] = config.get('bot_token', '')
    os.environ['TELEGRAM_CHAT_ID'] = config.get('chat_id', '')
else:
    print("❌ telegram_config.json bulunamadı!")
    print("\nÖnce Telegram ayarlarını yapın:")
    print("python telegram_integration.py")
    exit(1)

import sys
sys.path.append(str(Path(__file__).parent))

from telegram_integration import TelegramBot, TELEGRAM_CONFIG
TELEGRAM_CONFIG.update(config)


async def kontrol():
    """Telegram bot kontrolü"""
    print("\n" + "="*50)
    print("TELEGRAM BOT KONTROL")
    print("="*50)
    
    print(f"Bot Token: {config['bot_token'][:10]}...")
    print(f"Chat ID: {config['chat_id']}")
    
    try:
        # Bot oluştur
        bot = TelegramBot()
        
        # Test mesajı gönder
        await bot.send_notification(
            f"✅ Bot Çalışıyor!\n\n"
            f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"Paper Trading Komutları:\n"
            f"/status - Portföy durumu\n"
            f"/positions - Açık pozisyonlar\n"
            f"/trades - İşlem geçmişi\n"
            f"/start_trading - Trading başlat\n"
            f"/stop_trading - Trading durdur\n"
            f"/help - Tüm komutlar",
            "success"
        )
        
        print("\n✅ Mesaj gönderildi!")
        print("Telegram'ı kontrol edin.")
        
        # Bot kapat
        await bot.bot.close_session()
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        print("\nOlası sebepler:")
        print("1. Bot token yanlış")
        print("2. Chat ID yanlış") 
        print("3. Bot'a /start yazmadınız")
        print("4. İnternet bağlantısı yok")


if __name__ == "__main__":
    asyncio.run(kontrol())