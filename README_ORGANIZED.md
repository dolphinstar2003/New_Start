# BIST100 Trading System V2

## 🚀 Hızlı Başlangıç

```bash
python main_launcher.py
```

## 📁 Proje Yapısı

```
.
├── main_launcher.py          # Ana başlatıcı
├── paper_trading_module_v2.py # Gelişmiş paper trading
├── telegram_bot_integrated.py # Telegram bot
│
├── core/                     # Çekirdek modüller
│   ├── algolab_api.py       # AlgoLab API
│   ├── order_book_simulator.py # Order simülasyonu
│   └── limit_order_manager.py  # Limit order yönetimi
│
├── strategies/              
│   ├── proven/              # Kanıtlanmış stratejiler
│   │   ├── ultra_aggressive_backtest.py    # %8-10 aylık
│   │   ├── realistic_high_return_backtest.py # %5-6 aylık
│   │   └── enhanced_mtf_backtest.py        # %3-4 aylık
│   └── experimental/        # Deneysel stratejiler
│
├── ml_models/               # Machine Learning
├── indicators/              # Teknik indikatörler
├── data/                    # Veri dosyaları
├── logs/                    # Log dosyaları
└── config/                  # Ayarlar
```

## 📊 En İyi Stratejiler

1. **Ultra Aggressive** (%8-10 aylık hedef)
   - Yüksek risk/ödül
   - 2x kaldıraç kullanır
   - Agresif pozisyon boyutları

2. **Realistic High Return** (%5-6 aylık hedef)
   - Dengeli yaklaşım
   - Orta risk seviyesi
   - Kanıtlanmış performans

3. **Enhanced MTF** (%3-4 aylık hedef)
   - Düşük risk
   - Multi-timeframe analiz
   - Tutarlı getiri

## 🤖 Telegram Bot Komutları

Tüm komutlar çalışır durumda:
- `/status` - Portfolio durumu
- `/backtest` - Backtest çalıştır
- `/opportunities` - En iyi fırsatlar
- Ve 23 komut daha...

## 📈 Paper Trading V2 Özellikleri

- Gerçekçi limit order simülasyonu
- Order book depth
- Slippage ve commission
- Market microstructure
- Kısmi dolumlar (partial fills)
