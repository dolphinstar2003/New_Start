#!/usr/bin/env python3
"""
Unified Trading System Launcher
Choose your trading mode
"""
import sys
import subprocess
from pathlib import Path
import os

# Set working directory to project root
os.chdir(Path(__file__).parent)

def print_menu():
    """Display main menu"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              BIST100 TRADING SYSTEM V2                     ║
    ╠════════════════════════════════════════════════════════════╣
    ║  1. Paper Trading V2 (Gerçekçi Simülasyon)                ║
    ║  2. Paper Trading V2 + Telegram Bot                       ║
    ║  3. Backtest - Ultra Aggressive (%8-10 aylık)             ║
    ║  4. Backtest - Realistic High Return (%5-6 aylık)         ║
    ║  5. Backtest - Conservative MTF (%3-4 aylık)              ║
    ║  6. Train ML Models                                        ║
    ║  7. System Status & Logs                                   ║
    ║  0. Exit                                                   ║
    ╚════════════════════════════════════════════════════════════╝
    """)

def main():
    """Main launcher logic"""
    while True:
        print_menu()
        choice = input("\nSeçiminiz (0-7): ").strip()
        
        if choice == "0":
            print("\n✓ Sistem kapatıldı")
            break
        elif choice == "1":
            subprocess.run([sys.executable, "launchers/paper_trading/baslat_paper_trading_v2.py"], cwd=os.getcwd())
        elif choice == "2":
            subprocess.run([sys.executable, "launchers/start_v2_with_telegram.py"], cwd=os.getcwd())
        elif choice == "3":
            subprocess.run([sys.executable, "strategies/proven/ultra_aggressive_backtest.py"], cwd=os.getcwd())
        elif choice == "4":
            subprocess.run([sys.executable, "strategies/proven/realistic_high_return_backtest.py"], cwd=os.getcwd())
        elif choice == "5":
            subprocess.run([sys.executable, "strategies/proven/enhanced_mtf_backtest.py"], cwd=os.getcwd())
        elif choice == "6":
            subprocess.run([sys.executable, "ml_models/train_models.py"], cwd=os.getcwd())
        elif choice == "7":
            try:
                subprocess.run(["tail", "-n", "50", "logs/trading.log"], cwd=os.getcwd())
            except:
                print("Log dosyası bulunamadı")
        else:
            print("\n❌ Geçersiz seçim!")
        
        if choice in ["1", "2", "3", "4", "5", "6", "7"]:
            input("\nDevam etmek için Enter'a basın...")

if __name__ == "__main__":
    main()
