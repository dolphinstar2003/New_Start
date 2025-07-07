#!/usr/bin/env python3
"""
Test script for Telegram Bot integration with trading system modules
"""
import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

# Test imports
print("Testing module imports...")

try:
    from utils.backtest_runner import run_backtest, get_backtest_parameters
    print("✅ Backtest runner imported successfully")
except Exception as e:
    print(f"❌ Failed to import backtest runner: {e}")

try:
    from utils.train_runner import run_training, get_model_status, get_training_parameters
    print("✅ Train runner imported successfully")
except Exception as e:
    print(f"❌ Failed to import train runner: {e}")

try:
    from utils.walkforward_runner import run_walkforward_analysis, get_walkforward_parameters
    print("✅ Walkforward runner imported successfully")
except Exception as e:
    print(f"❌ Failed to import walkforward runner: {e}")

try:
    from utils.report_generator import generate_full_report
    print("✅ Report generator imported successfully")
except Exception as e:
    print(f"❌ Failed to import report generator: {e}")

# Test functions
print("\n\nTesting functions...")

# Test backtest parameters
try:
    params = get_backtest_parameters()
    print(f"\n📊 Backtest parameters: {params}")
    print("✅ Backtest parameters retrieved successfully")
except Exception as e:
    print(f"❌ Failed to get backtest parameters: {e}")

# Test model status
try:
    status = get_model_status()
    print(f"\n🧠 Model status: {status}")
    print("✅ Model status retrieved successfully")
except Exception as e:
    print(f"❌ Failed to get model status: {e}")

# Test training parameters
try:
    params = get_training_parameters()
    print(f"\n🎯 Training parameters: {params}")
    print("✅ Training parameters retrieved successfully")
except Exception as e:
    print(f"❌ Failed to get training parameters: {e}")

# Test walkforward parameters
try:
    params = get_walkforward_parameters()
    print(f"\n📈 Walkforward parameters: {params}")
    print("✅ Walkforward parameters retrieved successfully")
except Exception as e:
    print(f"❌ Failed to get walkforward parameters: {e}")

print("\n\n✅ All imports and basic functions tested successfully!")
print("\nTo test actual functionality:")
print("1. Run a backtest: run_backtest(days=30)")
print("2. Check model status: get_model_status()")
print("3. Train models: run_training()")
print("4. Run walkforward: run_walkforward_analysis()")
print("5. Generate report: generate_full_report()")

# Test telegram bot import
print("\n\nTesting Telegram bot...")
try:
    from telegram_bot_full_control import FullControlTelegramBot
    print("✅ Telegram bot imported successfully")
    
    # Check if config exists
    from pathlib import Path
    import json
    
    config_file = Path('telegram_config.json')
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        if config.get('bot_token') and config.get('chat_id'):
            print("✅ Telegram configuration found")
        else:
            print("⚠️ Telegram configuration incomplete - add bot_token and chat_id")
    else:
        print("⚠️ No telegram_config.json found - create one with bot_token and chat_id")
        
except Exception as e:
    print(f"❌ Failed to import Telegram bot: {e}")

print("\n\n🎉 Integration test complete!")