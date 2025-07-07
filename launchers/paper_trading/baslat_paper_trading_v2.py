#!/usr/bin/env python3
"""
Enhanced Paper Trading V2 - Gerçekçi Limit Order Simülasyonu ile
Türkçe kullanıcı arayüzü
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from paper_trading_module_v2 import EnhancedPaperTradingModule

# Renkli çıktı için ANSI kodları
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Başlık yazdır"""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}GELIŞMIŞ PAPER TRADING V2 - Gerçekçi Limit Order Simülasyonu{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}AlgoLab veri kaynağı + Telegram entegrasyonu{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

def print_commands():
    """Komutları yazdır"""
    print(f"{Colors.OKGREEN}Komutlar:{Colors.ENDC}")
    print("  durum   - Portfolio durumunu göster")
    print("  emirler - Aktif emirleri göster")
    print("  basla   - Otomatik ticareti başlat")
    print("  dur     - Otomatik ticareti durdur")
    print("  islem   - İşlem geçmişini göster")
    print("  dolum   - Emir dolum metriklerini göster")
    print("  perf    - Performans metriklerini göster")
    print("  kaydet  - Mevcut durumu kaydet")
    print("  telegram- Telegram bildirimlerini aç/kapa")
    print("  onay    - İşlem onaylarını aç/kapa")
    print("  cikis   - Programdan çık")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")

def format_currency_tr(value):
    """Para birimi formatla (Türkçe)"""
    return f"₺{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_percentage_tr(value):
    """Yüzde formatla (Türkçe)"""
    return f"%{value:.2f}".replace(".", ",")

async def main():
    """Ana program"""
    print_header()
    
    # Paper trading modülünü başlat
    print(f"{Colors.OKCYAN}Paper trading modülü başlatılıyor...{Colors.ENDC}")
    paper_trader = EnhancedPaperTradingModule(initial_capital=100000)
    
    # Önceki durumu yükle
    if paper_trader.load_state():
        print(f"{Colors.OKGREEN}✓ Önceki durum yüklendi{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}! Yeni portfolio oluşturuldu{Colors.ENDC}")
    
    # Bağlantıları başlat
    print(f"{Colors.OKCYAN}AlgoLab bağlantıları kuruluyor...{Colors.ENDC}")
    success = await paper_trader.initialize()
    
    if not success:
        print(f"{Colors.FAIL}✗ Başlatma hatası!{Colors.ENDC}")
        return
    
    print(f"{Colors.OKGREEN}✓ Sistem hazır!{Colors.ENDC}")
    
    # Telegram durumu
    if paper_trader.telegram_bot:
        print(f"{Colors.OKGREEN}✓ Telegram bot aktif{Colors.ENDC}")
    else:
        print(f"{Colors.WARNING}! Telegram bot aktif değil{Colors.ENDC}")
    
    print()
    print_commands()
    
    # Trading döngüsünü arka planda başlat
    trading_task = asyncio.create_task(paper_trader.trading_loop())
    
    # Komut döngüsü
    try:
        while True:
            try:
                cmd = input(f"\n{Colors.OKBLUE}Komut: {Colors.ENDC}").strip().lower()
                
                if cmd in ["cikis", "exit", "quit", "q"]:
                    break
                
                elif cmd in ["durum", "status"]:
                    status = paper_trader.get_portfolio_status()
                    print(f"\n{Colors.BOLD}Portfolio Durumu:{Colors.ENDC}")
                    print(f"Toplam Değer: {format_currency_tr(status['portfolio_value'])}")
                    print(f"Getiri: {format_currency_tr(status['total_return'])} ({format_percentage_tr(status['total_return_pct'])})")
                    print(f"Nakit: {format_currency_tr(status['cash'])}")
                    print(f"Pozisyonlar: {status['num_positions']}")
                    print(f"Aktif Emirler: {status['active_orders']}")
                    print(f"Toplam İşlem: {status['total_trades']}")
                    print(f"Kazanma Oranı: {format_percentage_tr(status['win_rate'])}")
                    
                    if status['positions']:
                        print(f"\n{Colors.BOLD}Açık Pozisyonlar:{Colors.ENDC}")
                        for pos in status['positions']:
                            color = Colors.OKGREEN if pos['profit_pct'] > 0 else Colors.FAIL
                            print(f"  {pos['symbol']}: {pos['shares']} adet @ {format_currency_tr(pos['entry_price'])} "
                                  f"→ {format_currency_tr(pos['current_price'])} "
                                  f"{color}({format_percentage_tr(pos['profit_pct'])}){Colors.ENDC} "
                                  f"[{pos['holding_days']} gün]")
                
                elif cmd in ["emirler", "orders"]:
                    active_orders = paper_trader.limit_order_manager.active_orders
                    if active_orders:
                        print(f"\n{Colors.BOLD}Aktif Emirler ({len(active_orders)} adet):{Colors.ENDC}")
                        for order_id, order in active_orders.items():
                            status_color = Colors.WARNING if order.status.value == "PARTIALLY_FILLED" else Colors.OKCYAN
                            print(f"  {order.symbol} - {order.side.value} {order.quantity} @ "
                                  f"{format_currency_tr(order.limit_price)} "
                                  f"{status_color}({order.status.value}){Colors.ENDC}")
                            if order.filled_quantity > 0:
                                print(f"    Dolum: {order.filled_quantity}/{order.quantity} "
                                      f"(Ort: {format_currency_tr(order.average_fill_price)})")
                    else:
                        print(f"{Colors.WARNING}Aktif emir yok{Colors.ENDC}")
                
                elif cmd in ["basla", "start"]:
                    paper_trader.auto_trade_enabled = True
                    print(f"{Colors.OKGREEN}✓ Otomatik ticaret başlatıldı{Colors.ENDC}")
                
                elif cmd in ["dur", "stop"]:
                    paper_trader.auto_trade_enabled = False
                    print(f"{Colors.WARNING}■ Otomatik ticaret durduruldu{Colors.ENDC}")
                
                elif cmd in ["islem", "trades"]:
                    trades_df = paper_trader.get_trade_history()
                    if not trades_df.empty:
                        print(f"\n{Colors.BOLD}Son İşlemler:{Colors.ENDC}")
                        for _, trade in trades_df.tail(5).iterrows():
                            color = Colors.OKGREEN if trade['profit'] > 0 else Colors.FAIL
                            print(f"  {trade['symbol']}: {format_currency_tr(trade['entry_price'])} → "
                                  f"{format_currency_tr(trade['exit_price'])} "
                                  f"{color}({format_percentage_tr(trade['profit_pct'])}){Colors.ENDC} "
                                  f"[{trade['reason']}]")
                    else:
                        print(f"{Colors.WARNING}Henüz işlem yok{Colors.ENDC}")
                
                elif cmd in ["dolum", "exec"]:
                    metrics = paper_trader.get_execution_metrics()
                    print(f"\n{Colors.BOLD}Emir Dolum Metrikleri:{Colors.ENDC}")
                    print(f"Verilen Emirler: {metrics.get('total_placed', 0)}")
                    print(f"Dolum Oranı: {format_percentage_tr(metrics.get('fill_rate', 0)*100)}")
                    print(f"Kısmi Dolum Oranı: {format_percentage_tr(metrics.get('partial_fill_rate', 0)*100)}")
                    print(f"Ort. Dolum Süresi: {metrics.get('avg_fill_time', 0):.1f} dk")
                    print(f"Ort. Kayma (Slippage): {format_percentage_tr(metrics.get('avg_slippage', 0))}")
                    print(f"Red Oranı: {format_percentage_tr(metrics.get('rejection_rate', 0)*100)}")
                
                elif cmd in ["perf", "performans"]:
                    metrics = paper_trader.get_performance_metrics()
                    if metrics:
                        print(f"\n{Colors.BOLD}Performans Metrikleri:{Colors.ENDC}")
                        print(f"Toplam İşlem: {metrics['total_trades']}")
                        print(f"Kazanma Oranı: {format_percentage_tr(metrics['win_rate'])}")
                        print(f"Net Kar: {format_currency_tr(metrics['net_profit'])}")
                        print(f"Ort. Kar: {format_currency_tr(metrics['avg_profit'])} ({format_percentage_tr(metrics['avg_profit_pct'])})")
                        print(f"Kar Faktörü: {metrics['profit_factor']:.2f}")
                        print(f"Sharpe Oranı: {metrics['sharpe_ratio']:.2f}")
                        print(f"Max Drawdown: {format_percentage_tr(metrics['max_drawdown'])}")
                        print(f"Toplam Komisyon: {format_currency_tr(metrics['total_commission'])}")
                        print(f"Dolum Oranı: {format_percentage_tr(metrics['fill_rate'])}")
                        print(f"Ort. Kayma: {metrics['avg_slippage_bps']:.1f} bps")
                    else:
                        print(f"{Colors.WARNING}Analiz için yeterli veri yok{Colors.ENDC}")
                
                elif cmd in ["kaydet", "save"]:
                    paper_trader.save_state()
                    print(f"{Colors.OKGREEN}✓ Durum kaydedildi{Colors.ENDC}")
                
                elif cmd == "telegram":
                    paper_trader.telegram_notifications = not paper_trader.telegram_notifications
                    status = "açık" if paper_trader.telegram_notifications else "kapalı"
                    print(f"{Colors.OKGREEN}Telegram bildirimleri {status}{Colors.ENDC}")
                
                elif cmd == "onay":
                    paper_trader.require_confirmation = not paper_trader.require_confirmation
                    status = "açık" if paper_trader.require_confirmation else "kapalı"
                    print(f"{Colors.OKGREEN}İşlem onayları {status}{Colors.ENDC}")
                
                elif cmd in ["yardim", "help", "?"]:
                    print_commands()
                
                else:
                    print(f"{Colors.WARNING}Bilinmeyen komut. 'yardim' yazın.{Colors.ENDC}")
                    
            except Exception as e:
                print(f"{Colors.FAIL}Hata: {e}{Colors.ENDC}")
                
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Program sonlandırılıyor...{Colors.ENDC}")
    
    finally:
        # Trading'i durdur
        print(f"{Colors.OKCYAN}Sistem kapatılıyor...{Colors.ENDC}")
        await paper_trader.stop()
        trading_task.cancel()
        
        # Final özet
        print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}KAPANIŞ ÖZETİ{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        
        status = paper_trader.get_portfolio_status()
        print(f"Final Portfolio Değeri: {format_currency_tr(status['portfolio_value'])}")
        print(f"Toplam Getiri: {format_currency_tr(status['total_return'])} ({format_percentage_tr(status['total_return_pct'])})")
        
        metrics = paper_trader.get_performance_metrics()
        if metrics:
            print(f"Toplam İşlem: {metrics['total_trades']}")
            print(f"Kazanma Oranı: {format_percentage_tr(metrics['win_rate'])}")
            print(f"Sharpe Oranı: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {format_percentage_tr(metrics['max_drawdown'])}")
            print(f"Dolum Oranı: {format_percentage_tr(metrics['fill_rate'])}")
            print(f"Ort. Kayma: {metrics['avg_slippage_bps']:.1f} bps")
        
        print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ Program başarıyla kapatıldı{Colors.ENDC}")


if __name__ == "__main__":
    # Windows'ta renkli çıktı için
    if sys.platform == "win32":
        import os
        os.system("color")
    
    asyncio.run(main())