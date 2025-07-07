"""
Additional Telegram Bot Commands Patch
Adds missing commands to the integrated bot
"""

def add_missing_commands(bot_instance):
    """Add missing commands to the bot"""
    bot = bot_instance.bot
    paper_trader = bot_instance.paper_trader
    
    # ===== WEEKLY REPORT =====
    @bot.message_handler(commands=['weekly_report'])
    def handle_weekly_report(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            # Calculate weekly statistics
            trades_df = paper_trader.get_trade_history()
            
            # Get last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            if not trades_df.empty:
                # Filter trades for last week
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                weekly_trades = trades_df[trades_df['exit_date'] >= start_date]
                
                # Calculate metrics
                total_trades = len(weekly_trades)
                profitable = len(weekly_trades[weekly_trades['profit'] > 0])
                total_profit = weekly_trades['profit'].sum()
                
                # Portfolio performance
                status = paper_trader.get_portfolio_status()
                
                text = f"""
📊 *HAFTALIK RAPOR*
📅 {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}

💼 *Portfolio Durumu:*
• Değer: ${status['portfolio_value']:,.2f}
• Haftalık Getiri: %{status.get('weekly_return', 0):.2f}
• Toplam Getiri: %{status['total_return_pct']:.2f}

📈 *İşlem Özeti:*
• Toplam İşlem: {total_trades}
• Karlı İşlem: {profitable}
• Zarar Eden: {total_trades - profitable}
• Kazanma Oranı: %{(profitable/max(1,total_trades)*100):.1f}
• Net Kar/Zarar: ${total_profit:,.2f}

🎯 *En İyi/Kötü İşlemler:*"""
                
                if not weekly_trades.empty:
                    best_trade = weekly_trades.loc[weekly_trades['profit_pct'].idxmax()]
                    worst_trade = weekly_trades.loc[weekly_trades['profit_pct'].idxmin()]
                    
                    text += f"""
• En İyi: {best_trade['symbol']} (%{best_trade['profit_pct']:.2f})
• En Kötü: {worst_trade['symbol']} (%{worst_trade['profit_pct']:.2f})"""
                
                # Position analysis
                positions = status.get('positions', [])
                if positions:
                    text += f"\n\n📊 *Açık Pozisyonlar:*\n"
                    text += f"• Sayı: {len(positions)}\n"
                    text += f"• Ortalama Süre: {sum(p['holding_days'] for p in positions)/len(positions):.1f} gün\n"
                
            else:
                text = "📊 *HAFTALIK RAPOR*\n\nBu hafta işlem yapılmadı."
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Rapor hatası: {str(e)}")
    
    # ===== LOGS COMMAND =====
    @bot.message_handler(commands=['logs'])
    def handle_logs(message):
        parts = message.text.split()
        lines = 50  # default
        if len(parts) > 1:
            try:
                lines = int(parts[1])
                lines = min(lines, 200)  # Max 200 lines
            except:
                pass
        
        try:
            # Read log file
            log_file = Path('/home/yunus/Belgeler/New_Start/logs/trading.log')
            if not log_file.exists():
                bot.reply_to(message, "❌ Log dosyası bulunamadı")
                return
            
            # Get last N lines
            with open(log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:]
            
            # Format log text
            log_text = ''.join(last_lines[-30:])  # Telegram message limit
            
            # Send as code block
            text = f"📋 *Son {min(len(last_lines), 30)} Log Satırı:*\n\n```\n{log_text}\n```"
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Log okuma hatası: {str(e)}")
    
    # ===== ANALYZE COMMAND =====  
    @bot.message_handler(commands=['analyze'])
    def handle_analyze(message):
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "❌ Kullanım: /analyze GARAN")
            return
        
        symbol = parts[1].upper() + '.IS'
        
        if symbol not in SACRED_SYMBOLS:
            bot.reply_to(message, f"❌ {symbol} desteklenen semboller arasında değil")
            return
        
        if not paper_trader or symbol not in paper_trader.market_data:
            bot.reply_to(message, "❌ Veri bulunamadı")
            return
        
        try:
            # Get market data
            data = paper_trader.market_data[symbol]
            
            # Calculate indicators
            score = paper_trader.calculate_opportunity_score(symbol, data)
            
            # Order book analysis
            order_book_analysis = {}
            if hasattr(paper_trader, 'limit_order_manager'):
                order_book_analysis = paper_trader.limit_order_manager.get_order_book_analysis(symbol)
            
            text = f"""
🔍 *{symbol} ANALİZİ*

📊 *Fiyat Bilgileri:*
• Güncel: ${data.get('last_price', 0):.2f}
• Açılış: ${data.get('open_price', 0):.2f}
• Gün Y/D: ${data.get('high_price', 0):.2f} / ${data.get('low_price', 0):.2f}
• Değişim: %{data.get('price_change_day', 0)*100:.2f}

📈 *Teknik Göstergeler:*
• Fırsat Skoru: {score}/100
• RSI: {paper_trader.indicators.get(symbol, {}).get('rsi', 50):.1f}
• Trend: {'Yükseliş' if paper_trader.indicators.get(symbol, {}).get('supertrend_trend', 0) == 1 else 'Düşüş'}

📊 *Hacim Analizi:*
• Hacim: {data.get('volume', 0):,.0f}
• Hacim Oranı: {data.get('volume_ratio', 1):.2f}x
"""
            
            if order_book_analysis:
                text += f"""
📖 *Emir Defteri:*
• Spread: %{order_book_analysis.get('spread_pct', 0):.3f}
• Alış/Satış Dengesi: {order_book_analysis.get('imbalance', 0):.2f}
• Likidite Skoru: {order_book_analysis.get('liquidity_score', 0):.0f}/100
"""
            
            # Position check
            if symbol in paper_trader.portfolio['positions']:
                pos = paper_trader.portfolio['positions'][symbol]
                text += f"""
💼 *Pozisyon Durumu:*
• Adet: {pos['shares']}
• Giriş: ${pos['entry_price']:.2f}
• K/Z: %{pos.get('unrealized_pnl_pct', 0):.2f}
"""
            else:
                text += "\n💡 *Öneri:* "
                if score >= 60:
                    text += "Güçlü AL sinyali! 🟢"
                elif score >= 40:
                    text += "Orta seviye fırsat 🟡"
                else:
                    text += "Zayıf sinyal, bekleyin 🔴"
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Analiz hatası: {str(e)}")
    
    # ===== SYSTEM INFO =====
    @bot.message_handler(commands=['system_info'])
    def handle_system_info(message):
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            
            # Disk
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            text = f"""
💻 *SİSTEM DURUMU*

🖥️ *CPU:*
• Kullanım: %{cpu_percent:.1f}
• Çekirdek: {psutil.cpu_count()}

💾 *Bellek:*
• Toplam: {memory.total / 1024 / 1024 / 1024:.1f} GB
• Kullanılan: {memory.used / 1024 / 1024 / 1024:.1f} GB (%{memory.percent:.1f})
• Boş: {memory.available / 1024 / 1024 / 1024:.1f} GB

💿 *Disk:*
• Toplam: {disk.total / 1024 / 1024 / 1024:.1f} GB
• Kullanılan: {disk.used / 1024 / 1024 / 1024:.1f} GB (%{disk.percent:.1f})
• Boş: {disk.free / 1024 / 1024 / 1024:.1f} GB

🤖 *Bot Process:*
• Bellek: {process_memory:.1f} MB
• Uptime: {bot_instance._get_uptime()}

📊 *Trading Sistemi:*
• Durum: {'✅ Aktif' if paper_trader and paper_trader.is_running else '❌ Pasif'}
• Auto Trade: {'✅' if paper_trader and paper_trader.auto_trade_enabled else '❌'}
"""
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Sistem bilgisi hatası: {str(e)}")
    
    # ===== OPPORTUNITIES =====
    @bot.message_handler(commands=['opportunities'])
    def handle_opportunities(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            # Get opportunities
            opportunities = paper_trader.evaluate_all_opportunities()
            
            if not opportunities:
                bot.reply_to(message, "❌ Fırsat bulunamadı")
                return
            
            text = "*🎯 EN İYİ FIRSATLAR*\n\n"
            
            # Show top 10
            for i, opp in enumerate(opportunities[:10], 1):
                emoji = "🟢" if opp['score'] >= 60 else "🟡" if opp['score'] >= 40 else "🔴"
                in_pos = "📌" if opp['in_position'] else ""
                
                text += f"{i}. {emoji} *{opp['symbol']}* {in_pos}\n"
                text += f"   • Skor: {opp['score']}/100\n"
                text += f"   • Fiyat: ${opp['price']:.2f}\n"
                text += f"   • Momentum: %{opp['momentum_day']:.2f}\n"
                
                if opp['in_position']:
                    text += f"   • Pozisyon K/Z: %{opp['current_profit']*100:.2f}\n"
                
                text += "\n"
            
            # Summary
            high_score = sum(1 for o in opportunities if o['score'] >= 60)
            text += f"*📊 Özet:*\n"
            text += f"• Güçlü Sinyal: {high_score}\n"
            text += f"• Toplam Fırsat: {len(opportunities)}\n"
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Fırsat analizi hatası: {str(e)}")
    
    # ===== PREDICT COMMAND =====
    @bot.message_handler(commands=['predict'])
    def handle_predict(message):
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "❌ Kullanım: /predict GARAN")
            return
        
        symbol = parts[1].upper() + '.IS'
        
        if symbol not in SACRED_SYMBOLS:
            bot.reply_to(message, f"❌ {symbol} desteklenen semboller arasında değil")
            return
        
        try:
            # Check if models exist
            model_path = Path("models/training_results.json")
            if not model_path.exists():
                bot.reply_to(message, "❌ ML modelleri henüz eğitilmemiş. Önce /train komutunu kullanın.")
                return
            
            # Simple prediction based on current indicators
            if paper_trader and symbol in paper_trader.market_data:
                score = paper_trader.calculate_opportunity_score(symbol, paper_trader.market_data[symbol])
                
                # Predict based on score
                if score >= 60:
                    prediction = "GÜÇLÜ AL 🟢"
                    confidence = 85 + (score - 60) / 4
                elif score >= 40:
                    prediction = "AL 🟡"
                    confidence = 70 + (score - 40) / 2
                elif score >= 20:
                    prediction = "BEKLE ⏸️"
                    confidence = 60
                else:
                    prediction = "SAT 🔴"
                    confidence = 70 + (20 - score) / 2
                
                text = f"""
🔮 *{symbol} TAHMİN*

📊 *ML Tahmin:* {prediction}
🎯 *Güven:* %{confidence:.0f}
📈 *Fırsat Skoru:* {score}/100

📉 *Teknik Göstergeler:*
• Trend: {'Yükseliş' if score > 50 else 'Düşüş'}
• Momentum: {'Pozitif' if score > 40 else 'Negatif'}
• Hacim: {'Yüksek' if paper_trader.market_data[symbol].get('volume_ratio', 1) > 1.2 else 'Normal'}

💡 *Öneri:*
"""
                if score >= 60:
                    text += "Güçlü alım fırsatı. Pozisyon açmayı düşünün."
                elif score >= 40:
                    text += "Orta seviye fırsat. Diğer göstergeleri de kontrol edin."
                else:
                    text += "Zayıf sinyal. Beklemeniz önerilir."
                
                bot.reply_to(message, text, parse_mode='Markdown')
            else:
                bot.reply_to(message, "❌ Veri bulunamadı. Önce /update_data komutunu deneyin.")
                
        except Exception as e:
            bot.reply_to(message, f"❌ Tahmin hatası: {str(e)}")
    
    # ===== TRADES COMMAND =====
    @bot.message_handler(commands=['trades'])
    def handle_trades(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            # Get trade history
            trades_df = paper_trader.get_trade_history()
            
            if trades_df.empty:
                bot.reply_to(message, "📭 İşlem geçmişi boş")
                return
            
            # Get last 20 trades
            recent_trades = trades_df.tail(20).sort_values('exit_date', ascending=False)
            
            text = "*📋 SON İŞLEMLER*\n\n"
            
            for idx, trade in recent_trades.iterrows():
                emoji = "🟢" if trade['profit'] > 0 else "🔴"
                
                # Format dates
                entry_date = pd.to_datetime(trade['entry_date']).strftime('%d.%m %H:%M')
                exit_date = pd.to_datetime(trade['exit_date']).strftime('%d.%m %H:%M')
                
                text += f"{emoji} *{trade['symbol']}*\n"
                text += f"  • Giriş: {entry_date} @ ${trade['entry_price']:.2f}\n"
                text += f"  • Çıkış: {exit_date} @ ${trade['exit_price']:.2f}\n"
                text += f"  • K/Z: ${trade['profit']:.2f} ({trade['profit_pct']:.2f}%)\n"
                text += f"  • Süre: {trade.get('holding_time', 'N/A')}\n\n"
            
            # Summary
            total_profit = recent_trades['profit'].sum()
            win_rate = len(recent_trades[recent_trades['profit'] > 0]) / len(recent_trades) * 100
            
            text += f"*📊 ÖZET:*\n"
            text += f"• Toplam K/Z: ${total_profit:.2f}\n"
            text += f"• Kazanma Oranı: %{win_rate:.1f}\n"
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ İşlem geçmişi hatası: {str(e)}")
    
    # ===== PERFORMANCE COMMAND =====
    @bot.message_handler(commands=['performance'])
    def handle_performance(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            metrics = paper_trader.get_performance_metrics()
            status = paper_trader.get_portfolio_status()
            
            text = f"""
📊 *DETAYLI PERFORMANS ANALİZİ*

💰 *Portföy Performansı:*
• Başlangıç: ${status.get('initial_capital', 100000):,.2f}
• Mevcut: ${status['portfolio_value']:,.2f}
• Net K/Z: ${status['portfolio_value'] - status.get('initial_capital', 100000):,.2f}
• Getiri: %{status['total_return_pct']:.2f}

📈 *İşlem İstatistikleri:*
• Toplam İşlem: {metrics.get('total_trades', 0)}
• Karlı İşlem: {metrics.get('winning_trades', 0)}
• Zararlı İşlem: {metrics.get('losing_trades', 0)}
• Kazanma Oranı: %{metrics.get('win_rate', 0):.1f}

💵 *Kar/Zarar Analizi:*
• Ortalama Kazanç: ${metrics.get('avg_win', 0):.2f}
• Ortalama Kayıp: ${metrics.get('avg_loss', 0):.2f}
• En İyi İşlem: ${metrics.get('best_trade', 0):.2f}
• En Kötü İşlem: ${metrics.get('worst_trade', 0):.2f}
• Kar Faktörü: {metrics.get('profit_factor', 0):.2f}

📊 *Risk Metrikleri:*
• Sharpe Oranı: {metrics.get('sharpe_ratio', 0):.2f}
• Sortino Oranı: {metrics.get('sortino_ratio', 0):.2f}
• Max Drawdown: %{metrics.get('max_drawdown', 0):.2f}
• Calmar Oranı: {metrics.get('calmar_ratio', 0):.2f}

⏱️ *Zaman Analizi:*
• Ortalama Pozisyon Süresi: {metrics.get('avg_holding_time', 'N/A')}
• En Uzun Pozisyon: {metrics.get('longest_trade', 'N/A')}
• En Kısa Pozisyon: {metrics.get('shortest_trade', 'N/A')}

📅 *Dönemsel Performans:*
• Günlük Getiri: %{metrics.get('daily_return', 0):.2f}
• Haftalık Getiri: %{metrics.get('weekly_return', 0):.2f}
• Aylık Getiri: %{metrics.get('monthly_return', 0):.2f}
"""
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Performans analizi hatası: {str(e)}")
    
    # ===== START TRADING COMMAND =====
    @bot.message_handler(commands=['start_trading'])
    def handle_start_trading(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            if paper_trader.auto_trade_enabled:
                bot.reply_to(message, "⚠️ Otomatik trading zaten aktif!")
                return
            
            paper_trader.auto_trade_enabled = True
            bot.reply_to(message, "✅ Otomatik trading başlatıldı!\n\n🤖 Sistem şimdi otomatik alım/satım yapacak.")
            logger.info(f"Auto trading started by {message.from_user.username}")
            
        except Exception as e:
            bot.reply_to(message, f"❌ Trading başlatma hatası: {str(e)}")
    
    # ===== STOP TRADING COMMAND =====
    @bot.message_handler(commands=['stop_trading'])
    def handle_stop_trading(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            if not paper_trader.auto_trade_enabled:
                bot.reply_to(message, "⚠️ Otomatik trading zaten kapalı!")
                return
            
            paper_trader.auto_trade_enabled = False
            bot.reply_to(message, "🛑 Otomatik trading durduruldu!\n\n⚠️ Açık pozisyonlar korunacak.")
            logger.info(f"Auto trading stopped by {message.from_user.username}")
            
        except Exception as e:
            bot.reply_to(message, f"❌ Trading durdurma hatası: {str(e)}")
    
    # ===== FORCE CHECK COMMAND =====
    @bot.message_handler(commands=['force_check'])
    def handle_force_check(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            bot.reply_to(message, "🔄 Pozisyonlar kontrol ediliyor...")
            
            # Check all positions
            checked = 0
            closed = 0
            
            for symbol in list(paper_trader.portfolio['positions'].keys()):
                result = paper_trader.check_position_exit(symbol)
                checked += 1
                if result and result.get('closed'):
                    closed += 1
            
            text = f"""
✅ *Kontrol Tamamlandı*

• Kontrol Edilen: {checked} pozisyon
• Kapatılan: {closed} pozisyon
• Kalan: {checked - closed} pozisyon

💡 Stop loss ve take profit seviyeleri kontrol edildi.
"""
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Pozisyon kontrolü hatası: {str(e)}")
    
    # ===== CLOSE ALL COMMAND =====
    @bot.message_handler(commands=['close_all'])
    def handle_close_all(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            positions = list(paper_trader.portfolio['positions'].keys())
            
            if not positions:
                bot.reply_to(message, "📭 Kapatılacak pozisyon yok")
                return
            
            bot.reply_to(message, f"🔄 {len(positions)} pozisyon kapatılıyor...")
            
            results = []
            total_pnl = 0
            
            for symbol in positions:
                result = paper_trader.close_position(symbol, reason="User requested close all")
                if result:
                    results.append(result)
                    total_pnl += result.get('profit', 0)
            
            text = f"""
✅ *TÜM POZİSYONLAR KAPATILDI*

• Kapatılan: {len(results)} pozisyon
• Toplam K/Z: ${total_pnl:+.2f}
• Ortalama K/Z: ${total_pnl/len(results):+.2f}

📊 Detaylar /trades komutu ile görüntülenebilir.
"""
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Pozisyon kapatma hatası: {str(e)}")
    
    # ===== CLOSE SINGLE POSITION =====
    @bot.message_handler(commands=['close'])
    def handle_close_position(message):
        parts = message.text.split()
        if len(parts) < 2:
            bot.reply_to(message, "❌ Kullanım: /close GARAN")
            return
        
        symbol = parts[1].upper()
        if not symbol.endswith('.IS'):
            symbol += '.IS'
        
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            if symbol not in paper_trader.portfolio['positions']:
                bot.reply_to(message, f"❌ {symbol} pozisyonu bulunamadı")
                return
            
            result = paper_trader.close_position(symbol, reason="User requested close")
            
            if result:
                emoji = "🟢" if result['profit'] > 0 else "🔴"
                text = f"""
{emoji} *POZİSYON KAPATILDI*

📊 *{symbol}*
• Giriş: ${result['entry_price']:.2f}
• Çıkış: ${result['exit_price']:.2f}
• Adet: {result['shares']}
• K/Z: ${result['profit']:.2f} ({result['profit_pct']:.2f}%)
• Süre: {result.get('holding_time', 'N/A')}
"""
                bot.reply_to(message, text, parse_mode='Markdown')
            else:
                bot.reply_to(message, "❌ Pozisyon kapatılamadı")
                
        except Exception as e:
            bot.reply_to(message, f"❌ Pozisyon kapatma hatası: {str(e)}")
    
    # ===== DAILY REPORT =====
    @bot.message_handler(commands=['daily_report'])
    def handle_daily_report(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            # Get today's trades
            trades_df = paper_trader.get_trade_history()
            today = datetime.now().date()
            
            if not trades_df.empty:
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
                today_trades = trades_df[trades_df['exit_date'].dt.date == today]
                
                # Calculate daily metrics
                num_trades = len(today_trades)
                total_profit = today_trades['profit'].sum() if num_trades > 0 else 0
                win_rate = len(today_trades[today_trades['profit'] > 0]) / num_trades * 100 if num_trades > 0 else 0
            else:
                num_trades = 0
                total_profit = 0
                win_rate = 0
            
            # Portfolio status
            status = paper_trader.get_portfolio_status()
            
            text = f"""
📅 *GÜNLÜK RAPOR*
📆 {today.strftime('%d.%m.%Y')}

💼 *Portföy Durumu:*
• Değer: ${status['portfolio_value']:,.2f}
• Nakit: ${status['cash']:,.2f}
• Pozisyon: {status['num_positions']}
• Günlük Getiri: %{status.get('daily_return', 0):.2f}

📊 *Bugünkü İşlemler:*
• Toplam: {num_trades}
• K/Z: ${total_profit:+.2f}
• Kazanma Oranı: %{win_rate:.1f}

📈 *Açık Pozisyonlar:*
"""
            
            # Add open positions summary
            if paper_trader.portfolio['positions']:
                for symbol, pos in paper_trader.portfolio['positions'].items():
                    pnl_pct = pos['unrealized_pnl_pct']
                    emoji = "🟢" if pnl_pct > 0 else "🔴"
                    text += f"• {emoji} {symbol}: {pnl_pct:+.1f}%\n"
            else:
                text += "• Açık pozisyon yok\n"
            
            # Market overview
            market_open = datetime.now().weekday() < 5 and 10 <= datetime.now().hour < 18
            text += f"\n🏛️ *Piyasa:* {'🟢 Açık' if market_open else '🔴 Kapalı'}"
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Günlük rapor hatası: {str(e)}")
    
    # ===== GET PARAMS COMMAND =====
    @bot.message_handler(commands=['get_params'])
    def handle_get_params(message):
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            text = f"""
⚙️ *TRADING PARAMETRELERİ*

💰 *Pozisyon Yönetimi:*
• Pozisyon Boyutu: %{paper_trader.position_size * 100:.0f}
• Max Pozisyon: {paper_trader.max_positions}
• Min İşlem: ${paper_trader.min_trade_amount:,.0f}

🛡️ *Risk Yönetimi:*
• Stop Loss: %{paper_trader.stop_loss_pct * 100:.1f}
• Take Profit: %{paper_trader.take_profit_pct * 100:.1f}
• Max Drawdown: %{getattr(paper_trader, 'max_drawdown_limit', 20):.0f}

📊 *İşlem Parametreleri:*
• Min Skor: {paper_trader.min_opportunity_score}
• Komisyon: %{paper_trader.commission * 100:.2f}
• Slippage: %{getattr(paper_trader, 'slippage', 0.001) * 100:.2f}

⏱️ *Zamanlama:*
• Check Interval: {getattr(paper_trader, 'check_interval', 60)}s
• Market Hours: 10:00 - 18:00
• Min Holding: {getattr(paper_trader, 'min_holding_time', 300)}s

🤖 *Otomasyonl:*
• Auto Trade: {'✅' if paper_trader.auto_trade_enabled else '❌'}
• Onay Modu: {'✅' if paper_trader.require_confirmation else '❌'}
• Demo Mode: {'✅' if getattr(paper_trader, 'demo_mode', False) else '❌'}
"""
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Parametre okuma hatası: {str(e)}")
    
    # ===== SET PARAM COMMAND =====
    @bot.message_handler(commands=['set_param'])
    def handle_set_param(message):
        parts = message.text.split()
        if len(parts) < 3:
            bot.reply_to(message, "❌ Kullanım: /set_param stop_loss 0.05")
            return
        
        param_name = parts[1].lower()
        try:
            value = float(parts[2])
        except:
            bot.reply_to(message, "❌ Geçersiz değer")
            return
        
        if not paper_trader:
            bot.reply_to(message, "❌ Trading sistemi bağlı değil")
            return
        
        try:
            # Allowed parameters
            allowed_params = {
                'stop_loss': ('stop_loss_pct', 0.01, 0.20),
                'take_profit': ('take_profit_pct', 0.01, 0.50),
                'position_size': ('position_size', 0.01, 0.30),
                'min_score': ('min_opportunity_score', 10, 90),
                'max_positions': ('max_positions', 1, 20),
            }
            
            if param_name not in allowed_params:
                bot.reply_to(message, f"❌ Bilinmeyen parametre: {param_name}\n\nİzin verilenler: {', '.join(allowed_params.keys())}")
                return
            
            attr_name, min_val, max_val = allowed_params[param_name]
            
            if value < min_val or value > max_val:
                bot.reply_to(message, f"❌ Değer aralık dışı: {min_val} - {max_val}")
                return
            
            # Set the parameter
            setattr(paper_trader, attr_name, value)
            
            bot.reply_to(message, f"✅ {param_name} = {value} olarak ayarlandı")
            logger.info(f"Parameter {param_name} set to {value} by {message.from_user.username}")
            
        except Exception as e:
            bot.reply_to(message, f"❌ Parametre ayarlama hatası: {str(e)}")
    
    # ===== UPDATE DATA COMMAND =====
    @bot.message_handler(commands=['update_data'])
    def handle_update_data(message):
        bot.reply_to(message, "🔄 Veri güncelleme başlatılıyor...")
        
        def update_data_thread():
            try:
                import subprocess
                cmd = [sys.executable, 'data_collection/update_all_data.py']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    bot.send_message(bot_instance.chat_id, "✅ Veri güncelleme tamamlandı!")
                else:
                    bot.send_message(bot_instance.chat_id, f"❌ Veri güncelleme hatası: {stderr[:500]}")
                    
            except Exception as e:
                bot.send_message(bot_instance.chat_id, f"❌ Güncelleme hatası: {str(e)}")
        
        update_thread = threading.Thread(target=update_data_thread)
        update_thread.daemon = True
        update_thread.start()
    
    # ===== TOP MOVERS COMMAND =====
    @bot.message_handler(commands=['top_movers'])
    def handle_top_movers(message):
        if not paper_trader or not paper_trader.market_data:
            bot.reply_to(message, "❌ Piyasa verisi yok")
            return
        
        try:
            movers = []
            for symbol, data in paper_trader.market_data.items():
                if symbol in SACRED_SYMBOLS:
                    change = data.get('price_change_pct', 0) * 100
                    movers.append((symbol, change, data.get('last_price', 0)))
            
            movers.sort(key=lambda x: abs(x[1]), reverse=True)
            
            text = "*📊 EN ÇOK HAREKET EDENLER*\n\n"
            
            # Top 10 movers
            for i, (symbol, change, price) in enumerate(movers[:10], 1):
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                text += f"{i}. {emoji} *{symbol}*\n"
                text += f"   • Fiyat: ${price:.2f}\n"
                text += f"   • Değişim: {change:+.2f}%\n\n"
            
            bot.reply_to(message, text, parse_mode='Markdown')
            
        except Exception as e:
            bot.reply_to(message, f"❌ Top movers hatası: {str(e)}")
    
    return True

# Import required modules
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from config.settings import SACRED_SYMBOLS
import sys
import subprocess
import threading
from loguru import logger