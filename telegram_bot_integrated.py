#!/usr/bin/env python3
"""
Integrated Telegram Bot for Full Trading System Control
Connects to all real modules and provides detailed feedback
"""
import os
import sys
import json
import threading
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import telebot
from loguru import logger
import pandas as pd
import psutil

# Add project path
sys.path.append('/home/yunus/Belgeler/New_Start')

# Configuration
config_file = Path(__file__).parent / 'telegram_config.json'
config = {}
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)

BOT_TOKEN = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
CHAT_ID = config.get('chat_id', os.getenv('TELEGRAM_CHAT_ID', ''))

# Import real modules
from config.settings import SACRED_SYMBOLS, BACKTEST_CONFIG


class IntegratedTelegramBot:
    """Integrated Telegram bot with full system control"""
    
    def __init__(self, paper_trader=None):
        self.paper_trader = paper_trader
        self.bot = telebot.TeleBot(BOT_TOKEN)
        self.chat_id = CHAT_ID
        self.is_running = False
        
        # Process tracking
        self.active_processes = {}
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup all command handlers with detailed responses"""
        
        # ===== HELP COMMAND =====
        @self.bot.message_handler(commands=['help'])
        def handle_help(message):
            help_text = """
🎮 *BIST100 Trading System - Full Control*

*📊 Trading Commands:*
/status - Detaylı portföy durumu ve performans
/positions - Açık pozisyonlar ve kar/zarar
/trades - Son işlemler (son 20)
/performance - Detaylı performans metrikleri
/opportunities - En iyi trading fırsatları
/analyze GARAN - Hisse analizi

*🎮 Trading Control:*
/start\\_trading - Otomatik trading başlat
/stop\\_trading - Trading durdur
/force\\_check - Pozisyonları hemen kontrol et
/close\\_all - Tüm pozisyonları kapat
/close GARAN - Belirli pozisyonu kapat

*📈 Backtest & Analysis:*
/backtest 30 - Son 30 gün backtest
/backtest\\_range 2024-01-01 2024-12-31
/walkforward - Walkforward analizi (6 aylık)
/optimize - Parametre optimizasyonu

*🧠 ML & Training:*
/train - ML modellerini eğit
/train\\_status - Eğitim durumu
/model\\_performance - Model performansı
/predict GARAN - Tahmin al

*📊 Data Management:*
/update\\_data - Veri güncelle
/data\\_status - Veri durumu
/missing\\_data - Eksik veriler
/download\\_data - Veri indir

*💻 System Control:*
/system\\_info - CPU, RAM, Disk durumu
/logs 50 - Son 50 satır log
/processes - Aktif işlemler
/restart - Sistemi yeniden başlat
/shutdown - Sistemi kapat

*📄 Reports & Export:*
/daily\\_report - Günlük rapor
/weekly\\_report - Haftalık rapor
/export\\_trades csv - İşlemleri dışa aktar
/export\\_positions - Pozisyonları dışa aktar

*⚙️ Settings:*
/get\\_params - Trading parametreleri
/set\\_param stop\\_loss 0.05
/risk\\_settings - Risk ayarları
/notification\\_settings - Bildirim ayarları

*🔍 Market Analysis:*
/market\\_overview - Piyasa genel durumu
/top\\_movers - En çok hareket edenler
/volume\\_leaders - Hacim liderleri
/sector\\_analysis - Sektör analizi

Detaylı bilgi için: /help [komut]
Örnek: /help backtest
"""
            self.bot.reply_to(message, help_text, parse_mode='Markdown')
            logger.info(f"Help command from {message.from_user.username}")
        
        # ===== STATUS COMMAND WITH DETAILS =====
        @self.bot.message_handler(commands=['status'])
        def handle_status(message):
            if self.paper_trader:
                status = self.paper_trader.get_portfolio_status()
                metrics = self.paper_trader.get_performance_metrics() or {}
                
                # Market hours check
                now = datetime.now()
                market_open = now.weekday() < 5 and 10 <= now.hour < 18
                market_status = "🟢 Açık" if market_open else "🔴 Kapalı"
                
                text = f"""
💼 *DETAYLI PORTFÖY DURUMU*

📊 *Değer ve Getiri:*
• Toplam Değer: ${status['portfolio_value']:,.2f}
• Başlangıç: ${status.get('initial_capital', 100000):,.2f}
• Nakit: ${status['cash']:,.2f} ({(status['cash']/status['portfolio_value']*100):.1f}%)
• Toplam Getiri: ${status['portfolio_value'] - status.get('initial_capital', 100000):+,.2f}
• Getiri %: {status['total_return_pct']:+.2f}%

📈 *Pozisyonlar:*
• Açık: {status['num_positions']}/10
• Değer: ${status['portfolio_value'] - status['cash']:,.2f}
• Ortalama Pozisyon: ${(status['portfolio_value'] - status['cash'])/max(1, status['num_positions']):,.2f}

📊 *İşlem İstatistikleri:*
• Toplam İşlem: {status['total_trades']}
• Bugünkü İşlem: {status.get('trades_today', 0)}
• Son İşlem: {status.get('last_trade_time', 'N/A')}
"""

                if metrics:
                    text += f"""
🎯 *Performans Metrikleri:*
• Kazanma Oranı: %{metrics.get('win_rate', 0):.1f}
• Ortalama Kazanç: ${metrics.get('avg_win', 0):.2f}
• Ortalama Kayıp: ${metrics.get('avg_loss', 0):.2f}
• Kar Faktörü: {metrics.get('profit_factor', 0):.2f}
• Sharpe Oranı: {metrics.get('sharpe_ratio', 0):.2f}
• Max Drawdown: %{metrics.get('max_drawdown', 0):.2f}
• Sortino Oranı: {metrics.get('sortino_ratio', 0):.2f}
"""

                text += f"""
⚙️ *Sistem Durumu:*
• Trading: {'✅ Aktif' if self.paper_trader.auto_trade_enabled else '❌ Pasif'}
• Piyasa: {market_status}
• Onay Modu: {'✅ Açık' if self.paper_trader.require_confirmation else '❌ Kapalı'}
• Bot Uptime: {self._get_uptime()}

🕐 *Zaman:* {now.strftime('%Y-%m-%d %H:%M:%S')}
"""
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "❌ Trading sistemi bağlı değil")
        
        # ===== DETAILED POSITIONS =====
        @self.bot.message_handler(commands=['positions'])
        def handle_positions(message):
            if self.paper_trader and self.paper_trader.portfolio['positions']:
                text = "*📈 AÇIK POZİSYONLAR*\n\n"
                total_value = 0
                total_pnl = 0
                
                sorted_positions = sorted(
                    self.paper_trader.portfolio['positions'].items(),
                    key=lambda x: x[1]['unrealized_pnl_pct'],
                    reverse=True
                )
                
                for i, (symbol, pos) in enumerate(sorted_positions, 1):
                    pnl = pos['unrealized_pnl']
                    pnl_pct = pos['unrealized_pnl_pct']
                    current_price = pos.get('current_price', pos['average_price'])
                    value = pos['shares'] * current_price
                    total_value += value
                    total_pnl += pnl
                    
                    # Calculate holding time
                    entry_date = pos.get('entry_date', datetime.now())
                    if isinstance(entry_date, str):
                        entry_date = datetime.fromisoformat(entry_date)
                    holding_days = (datetime.now() - entry_date).days
                    
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    trend = "📈" if pnl >= 0 else "📉"
                    
                    text += f"{i}. {emoji} *{symbol}* {trend}\n"
                    text += f"   💰 Değer: ${value:,.2f}\n"
                    text += f"   📊 Adet: {pos['shares']} @ ${pos['average_price']:.2f}\n"
                    text += f"   💵 Güncel: ${current_price:.2f}\n"
                    text += f"   💹 K/Z: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                    text += f"   🛑 Stop: ${pos.get('stop_loss', pos['average_price']*0.97):.2f}\n"
                    text += f"   🎯 Hedef: ${pos.get('take_profit', pos['average_price']*1.08):.2f}\n"
                    text += f"   📅 Gün: {holding_days}\n"
                    text += f"   📊 Score: {pos.get('entry_score', 'N/A')}\n\n"
                
                # Summary
                avg_pnl_pct = (total_pnl / total_value * 100) if total_value > 0 else 0
                text += f"*📊 ÖZET:*\n"
                text += f"• Toplam Değer: ${total_value:,.2f}\n"
                text += f"• Toplam K/Z: ${total_pnl:+.2f} ({avg_pnl_pct:+.1f}%)\n"
                text += f"• Ortalama Pozisyon: ${total_value/len(sorted_positions):,.2f}\n"
                
                # Risk metrics
                text += f"\n*⚠️ RİSK:*\n"
                text += f"• Toplam Risk: ${total_value * 0.03:,.2f} (%3 SL)\n"
                text += f"• Portfolio Risk: %{(total_value / self.paper_trader.portfolio['cash']) * 3:.1f}\n"
                
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "📭 Açık pozisyon yok")
        
        # ===== BACKTEST COMMAND =====
        @self.bot.message_handler(commands=['backtest'])
        def handle_backtest(message):
            parts = message.text.split()
            days = 30  # default
            if len(parts) > 1:
                try:
                    days = int(parts[1])
                    days = min(days, 365)  # Max 1 year
                except:
                    pass
            
            self.bot.reply_to(message, f"📈 *{days} Günlük Backtest Başlatılıyor...*\n\nBu işlem 1-2 dakika sürebilir.", parse_mode='Markdown')
            
            # Run backtest in thread
            def run_backtest_thread():
                try:
                    # Import and run real backtest
                    from backtest import run_backtest
                    
                    # Run async function in new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    results = loop.run_until_complete(run_backtest(days=days))
                    loop.close()
                    
                    # Format results
                    if results['backtest_engine'] == 'real':
                        text = f"""
✅ *Backtest Tamamlandı ({days} gün)*

📊 *Sonuçlar:*
• Başlangıç: ${results.get('initial_capital', 100000):,.2f}
• Bitiş: ${results.get('final_value', 100000):,.2f}
• Toplam Getiri: %{results['total_return']:.2f}
• Yıllık Getiri: %{(results['total_return'] / days * 365):.2f}

📈 *Performans:*
• Sharpe Oranı: {results['sharpe_ratio']:.2f}
• Kazanma Oranı: %{results['win_rate']:.1f}
• Max Drawdown: %{results.get('max_drawdown', 0):.2f}

💰 *İşlemler:*
• Toplam: {results['total_trades']}
• Karlı: {results['profitable_trades']}
• Zararlı: {results.get('losing_trades', 0)}
• Günlük Ort: {results['total_trades'] / max(1, days):.1f}

🎯 *En İyi/Kötü:*
• En İyi: %{results['best_trade']:.2f}
• En Kötü: %{results['worst_trade']:.2f}

📊 *Analiz:*
• Kar Faktörü: {results['profitable_trades'] / max(1, results.get('losing_trades', 1)):.2f}
• Beklenen Değer: %{(results['win_rate']/100 * abs(results['best_trade']) - (1-results['win_rate']/100) * abs(results['worst_trade'])):.2f}

⚙️ Motor: Gerçek Backtest Engine
📅 Tarih: {results.get('timestamp', datetime.now().isoformat())[:19]}
"""
                    else:
                        text = f"""
⚠️ *Backtest Tamamlandı (Demo Mod)*

📊 *Simülasyon Sonuçları ({days} gün):*
• Toplam Getiri: %{results['total_return']:.2f}
• Sharpe Oranı: {results['sharpe_ratio']:.2f}
• Kazanma Oranı: %{results['win_rate']:.1f}
• Toplam İşlem: {results['total_trades']}

*Not: Gerçek veri kullanılamadı, simülasyon sonuçları*
"""
                    
                    self.bot.send_message(self.chat_id, text, parse_mode='Markdown')
                    
                except Exception as e:
                    error_text = f"❌ *Backtest Hatası:*\n\n`{str(e)}`\n\nLütfen daha sonra tekrar deneyin."
                    self.bot.send_message(self.chat_id, error_text, parse_mode='Markdown')
                    logger.error(f"Backtest error: {e}")
            
            backtest_thread = threading.Thread(target=run_backtest_thread)
            backtest_thread.daemon = True
            backtest_thread.start()
            
            logger.info(f"Backtest started by {message.from_user.username} for {days} days")
        
        # ===== TRAIN ML MODELS =====
        @self.bot.message_handler(commands=['train'])
        def handle_train(message):
            if 'train' in self.active_processes and self.active_processes['train'].poll() is None:
                self.bot.reply_to(message, "⚠️ Eğitim zaten devam ediyor!")
                return
            
            self.bot.reply_to(message, """
🧠 *ML Model Eğitimi Başlatılıyor...*

Eğitilecek modeller:
• XGBoost (Ana model)
• LightGBM (Yedek model)
• LSTM (Derin öğrenme)
• Ensemble (Birleşik model)

Bu işlem 5-10 dakika sürebilir.
""", parse_mode='Markdown')
            
            # Start training in subprocess  
            cmd = ['/home/yunus/Belgeler/New_Start/venv/bin/python', 'ml_models/train_all_models.py']
            self.active_processes['train'] = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd='/home/yunus/Belgeler/New_Start'
            )
            
            # Monitor training
            def monitor_training():
                process = self.active_processes['train']
                start_time = datetime.now()
                
                while process.poll() is None:
                    import time
                    time.sleep(10)
                
                duration = (datetime.now() - start_time).seconds
                
                if process.returncode == 0:
                    # Read training results if available
                    results_file = Path("models/training_results.json")
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        text = f"""
✅ *ML Model Eğitimi Tamamlandı!*

⏱️ Süre: {duration//60} dakika {duration%60} saniye

📊 *Model Performansları:*
• XGBoost: %{results.get('xgboost', {}).get('accuracy', 0)*100:.1f} doğruluk
• LightGBM: %{results.get('lightgbm', {}).get('accuracy', 0)*100:.1f} doğruluk
• LSTM: %{results.get('lstm', {}).get('accuracy', 0)*100:.1f} doğruluk

🎯 *En İyi Model:* {results.get('best_model', 'XGBoost')}

📁 Modeller kaydedildi: `models/` klasörü
"""
                    else:
                        text = f"✅ Eğitim tamamlandı! (Süre: {duration//60}dk {duration%60}sn)"
                    
                    self.bot.send_message(self.chat_id, text, parse_mode='Markdown')
                else:
                    stderr = process.stderr.read()
                    self.bot.send_message(
                        self.chat_id,
                        f"❌ Eğitim başarısız!\n\nHata: {stderr[:500]}..."
                    )
            
            monitor_thread = threading.Thread(target=monitor_training)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            logger.info(f"ML training started by {message.from_user.username}")
        
        # ===== SYSTEM INFO =====
        @self.bot.message_handler(commands=['system_info'])
        def handle_system_info(message):
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get process info
            current_process = psutil.Process()
            process_memory = current_process.memory_info().rss / 1024**3  # GB
            
            # Network info
            net_io = psutil.net_io_counters()
            
            text = f"""
💻 *SİSTEM BİLGİLERİ*

🖥️ *CPU:*
• Kullanım: {cpu}%
• Çekirdek: {psutil.cpu_count()} ({psutil.cpu_count(logical=False)} fiziksel)
• Frekans: {psutil.cpu_freq().current:.0f} MHz

🧠 *RAM:*
• Kullanım: {memory.percent}%
• Kullanılan: {memory.used/1024**3:.1f} GB
• Toplam: {memory.total/1024**3:.1f} GB
• Bot Kullanımı: {process_memory:.2f} GB

💾 *Disk:*
• Kullanım: {disk.percent}%
• Kullanılan: {disk.used/1024**3:.1f} GB
• Toplam: {disk.total/1024**3:.1f} GB
• Boş: {disk.free/1024**3:.1f} GB

🌐 *Network:*
• Gönderilen: {net_io.bytes_sent/1024**3:.1f} GB
• Alınan: {net_io.bytes_recv/1024**3:.1f} GB

📊 *Trading Durumu:*
• Paper Trading: {'✅ Aktif' if self.paper_trader and self.paper_trader.is_running else '❌ Pasif'}
• WebSocket: {'✅ Bağlı' if self.paper_trader and self.paper_trader.algolab_socket else '❌ Bağlı Değil'}
• Aktif İşlemler: {len(self.active_processes)}

🐍 *Python:*
• Version: {sys.version.split()[0]}
• Venv: {'✅ Aktif' if 'venv' in sys.prefix else '❌ Pasif'}

🕐 *Uptime:* {self._get_uptime()}
📅 *Tarih:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            self.bot.reply_to(message, text, parse_mode='Markdown')
            logger.info(f"System info requested by {message.from_user.username}")
        
        # ===== DATA STATUS =====
        @self.bot.message_handler(commands=['data_status'])
        def handle_data_status(message):
            try:
                self.bot.reply_to(message, "📊 Veri durumu kontrol ediliyor...")
                
                # Check data files
                data_dir = Path("data")
                csv_files = list(data_dir.glob("**/*.csv"))
                total_size = sum(f.stat().st_size for f in csv_files) / 1024**2  # MB
                
                # Check each symbol
                symbol_status = {}
                for symbol in SACRED_SYMBOLS[:10]:  # Top 10
                    daily_file = data_dir / f"{symbol}_1d.csv"
                    hourly_file = data_dir / f"{symbol}_1h.csv"
                    
                    if daily_file.exists():
                        df = pd.read_csv(daily_file)
                        last_date = pd.to_datetime(df['timestamp']).max()
                        days_old = (datetime.now() - last_date).days
                        symbol_status[symbol] = {
                            'daily': len(df),
                            'last_update': last_date,
                            'days_old': days_old
                        }
                
                text = f"""
📊 *VERİ DURUMU*

📁 *Genel:*
• Toplam Dosya: {len(csv_files)}
• Toplam Boyut: {total_size:.1f} MB
• Veri Klasörü: `data/`

📈 *Sembol Durumu:*
"""
                for symbol, status in symbol_status.items():
                    emoji = "🟢" if status['days_old'] <= 1 else "🟡" if status['days_old'] <= 7 else "🔴"
                    text += f"\n{emoji} *{symbol}:*\n"
                    text += f"  • Günlük: {status['daily']} kayıt\n"
                    text += f"  • Son: {status['last_update'].strftime('%Y-%m-%d')}\n"
                    text += f"  • Yaş: {status['days_old']} gün\n"
                
                text += f"\n💡 *Öneri:* "
                if any(s['days_old'] > 7 for s in symbol_status.values()):
                    text += "Veri güncellenmeli! /update\\_data"
                else:
                    text += "Veriler güncel ✅"
                
                self.bot.send_message(self.chat_id, text, parse_mode='Markdown')
                
            except Exception as e:
                self.bot.reply_to(message, f"❌ Hata: {str(e)}")
        
        # ===== MARKET OVERVIEW =====
        @self.bot.message_handler(commands=['market_overview'])
        def handle_market_overview(message):
            if not self.paper_trader:
                self.bot.reply_to(message, "❌ Trading sistemi bağlı değil")
                return
            
            try:
                text = "*🏛️ PİYASA GENEL DURUMU*\n\n"
                
                # Get market data for all symbols
                market_data = self.paper_trader.market_data
                
                if not market_data:
                    self.bot.reply_to(message, "❌ Piyasa verisi yok")
                    return
                
                # Calculate market statistics
                gainers = []
                losers = []
                volume_leaders = []
                
                for symbol, data in market_data.items():
                    if symbol not in SACRED_SYMBOLS:
                        continue
                    
                    change = data.get('price_change_pct', 0)
                    volume = data.get('volume', 0)
                    
                    if change > 0:
                        gainers.append((symbol, change))
                    else:
                        losers.append((symbol, change))
                    
                    volume_leaders.append((symbol, volume, change))
                
                # Sort lists
                gainers.sort(key=lambda x: x[1], reverse=True)
                losers.sort(key=lambda x: x[1])
                volume_leaders.sort(key=lambda x: x[1], reverse=True)
                
                # Market summary
                total_gainers = len(gainers)
                total_losers = len(losers)
                market_sentiment = "Yükseliş" if total_gainers > total_losers else "Düşüş"
                
                text += f"📊 *Genel Durum:*\n"
                text += f"• Piyasa: {'🟢' if total_gainers > total_losers else '🔴'} {market_sentiment}\n"
                text += f"• Yükselenler: {total_gainers}\n"
                text += f"• Düşenler: {total_losers}\n\n"
                
                # Top gainers
                text += "📈 *En Çok Yükselenler:*\n"
                for symbol, change in gainers[:5]:
                    text += f"• {symbol}: %{change:+.2f}\n"
                
                # Top losers
                text += "\n📉 *En Çok Düşenler:*\n"
                for symbol, change in losers[:5]:
                    text += f"• {symbol}: %{change:+.2f}\n"
                
                # Volume leaders
                text += "\n📊 *Hacim Liderleri:*\n"
                for symbol, volume, change in volume_leaders[:5]:
                    text += f"• {symbol}: {volume:,.0f} lot (%{change:+.2f})\n"
                
                self.bot.reply_to(message, text, parse_mode='Markdown')
                
            except Exception as e:
                self.bot.reply_to(message, f"❌ Hata: {str(e)}")
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_unknown(message):
            suggestions = {
                'durum': '/status',
                'pozisyon': '/positions',
                'help': '/help',
                'yardım': '/help',
                'başlat': '/start_trading',
                'durdur': '/stop_trading'
            }
            
            suggestion = suggestions.get(message.text.lower().strip('/'), '')
            
            if suggestion:
                self.bot.reply_to(
                    message,
                    f"❓ Bunu mu demek istediniz: {suggestion}"
                )
            else:
                self.bot.reply_to(
                    message,
                    "❓ Bilinmeyen komut. /help yazarak komutları görebilirsiniz."
                )
    
    def _get_uptime(self):
        """Get bot uptime"""
        if hasattr(self, '_start_time'):
            uptime = datetime.now() - self._start_time
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            return f"{uptime.days}g {hours}s {minutes}d"
        return "N/A"
    
    def send_notification(self, message, notification_type="info"):
        """Send notification to Telegram"""
        try:
            # Escape markdown
            message = message.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]')
            
            emojis = {
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "info": "ℹ️",
                "trade": "💰"
            }
            emoji = emojis.get(notification_type, "📢")
            
            self.bot.send_message(self.chat_id, f"{emoji} {message}", parse_mode='Markdown')
            logger.info(f"Notification sent: {notification_type}")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def start(self):
        """Start the bot"""
        self.is_running = True
        self._start_time = datetime.now()
        
        bot_thread = threading.Thread(target=self._polling_loop)
        bot_thread.daemon = True
        bot_thread.start()
        logger.info("Integrated Telegram bot started")
        
        # Send start notification
        self.send_notification(
            "*🤖 Integrated Trading Bot Başladı*\n\n"
            "Full sistem kontrolü aktif!\n"
            "/help yazarak başlayın",
            "success"
        )
    
    def _polling_loop(self):
        """Polling loop"""
        while self.is_running:
            try:
                self.bot.polling(non_stop=True, interval=1, timeout=30)
            except Exception as e:
                logger.error(f"Polling error: {e}")
                if self.is_running:
                    import time
                    time.sleep(5)
    
    def stop(self):
        """Stop the bot"""
        self.is_running = False
        
        # Stop all subprocesses
        for name, process in self.active_processes.items():
            if process.poll() is None:
                process.terminate()
                logger.info(f"Terminated process: {name}")
        
        self.bot.stop_polling()
        logger.info("Integrated Telegram bot stopped")


if __name__ == "__main__":
    # Test the bot
    bot = IntegratedTelegramBot()
    bot.start()
    
    print("\n" + "="*60)
    print("🤖 INTEGRATED TELEGRAM BOT")
    print("="*60)
    print("Full sistem kontrolü aktif!")
    print("\nKomutlar:")
    print("• /help - Tüm komutları göster")
    print("• /status - Detaylı durum")
    print("• /backtest 30 - 30 günlük backtest")
    print("• /train - ML modellerini eğit")
    print("• /market_overview - Piyasa durumu")
    print("\nDurdurmak için Ctrl+C")
    print("="*60)
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nBot durduruluyor...")
        bot.stop()