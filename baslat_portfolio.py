#!/usr/bin/env python3
"""
Kolay Portfolio Trading Başlatıcı
"""
import os
import sys

print("""
╔══════════════════════════════════════════════╗
║        PORTFOLIO TRADING LAUNCHER            ║
╠══════════════════════════════════════════════╣
║                                              ║
║  1. Normal Portfolio Trading (5 dk kontrol)  ║
║  2. Hızlı Test (1 dk kontrol)               ║
║  3. Backtest (30-240 gün)                    ║
║  4. Mevcut Durumu Göster                     ║
║  5. Çıkış                                    ║
║                                              ║
╚══════════════════════════════════════════════╝
""")

choice = input("Seçiminiz (1-5): ")

if choice == '1':
    # Normal trading
    os.system("python portfolio_launcher.py")
elif choice == '2':
    # Fast test mode 
    print("\n⚡ Hızlı test modu (1 dakika kontrol)")
    os.system("python portfolio_launcher_debug.py")
elif choice == '3':
    # Backtest
    print("\n📊 Backtest modları:")
    print("1. Hızlı (30, 60 gün)")
    print("2. Normal (30, 60, 90, 120 gün)")
    print("3. Uzun (30, 60, 90, 120, 180, 240 gün)")
    
    bt_choice = input("\nBacktest seçimi (1-3): ")
    if bt_choice == '1':
        os.system('echo -e "1\\n100000" | python portfolio_backtest_runner.py')
    elif bt_choice == '2':
        os.system('echo -e "2\\n100000" | python portfolio_backtest_runner.py')
    elif bt_choice == '3':
        os.system('echo -e "3\\n100000" | python portfolio_backtest_runner.py')
elif choice == '4':
    # Show status
    os.system("python portfolio_status_checker.py")
elif choice == '5':
    print("\nGüle güle! 👋")
    sys.exit(0)
else:
    print("\nGeçersiz seçim!")