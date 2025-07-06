#!/usr/bin/env python3
"""
Test real-world scenarios that were causing Telegram parsing errors
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project path
sys.path.append(str(Path(__file__).parent))

from utils.telegram_utils import (
    escape_markdown_v1, format_currency, format_percentage, 
    format_trade_message, format_portfolio_status, format_position_list,
    format_opportunities
)

def test_actual_error_scenarios():
    """Test scenarios that were actually causing parsing errors"""
    print("Testing actual error scenarios:")
    
    # Scenario 1: Complex trade reason with parentheses and percentages
    trade_msg = format_trade_message(
        action="SELL",
        symbol="TUPRS.IS",
        shares=50,
        price=486.50,
        reason="Stop-loss triggered (entry: 512.30, loss: -5.03%)",
        profit=-1290.00
    )
    print("Scenario 1 - Complex trade message:")
    print(trade_msg)
    print("\n" + "="*50 + "\n")
    
    # Scenario 2: Portfolio with special characters in positions
    mock_positions = [
        {
            'symbol': 'GARAN.IS',
            'shares': 100,
            'entry_price': 29.75,
            'current_price': 31.20,
            'value': 3120.00,
            'profit': 145.00,
            'profit_pct': 4.87,
            'holding_days': 5
        },
        {
            'symbol': 'AKBNK.IS',
            'shares': 75,
            'entry_price': 28.40,
            'current_price': 26.85,
            'value': 2013.75,
            'profit': -116.25,
            'profit_pct': -4.08,
            'holding_days': 12
        }
    ]
    
    positions_msg = format_position_list(mock_positions)
    print("Scenario 2 - Positions with gains/losses:")
    print(positions_msg)
    print("\n" + "="*50 + "\n")
    
    # Scenario 3: Opportunities with complex data
    mock_opportunities = [
        {
            'symbol': 'THYAO.IS',
            'score': 78.5,
            'price': 275.20,
            'in_position': False,
            'momentum_1h': 2.3,
            'momentum_day': -1.2,
            'volume_ratio': 1.85
        },
        {
            'symbol': 'BIMAS.IS',
            'score': 65.2,
            'price': 156.80,
            'in_position': True,
            'momentum_1h': -0.8,
            'momentum_day': 3.4,
            'volume_ratio': 0.92
        }
    ]
    
    opps_msg = format_opportunities(mock_opportunities, 5)
    print("Scenario 3 - Market opportunities:")
    print(opps_msg)
    print("\n" + "="*50 + "\n")
    
    # Scenario 4: Portfolio status with edge cases
    edge_case_status = {
        'portfolio_value': 98765.43,
        'total_return': -1234.57,
        'total_return_pct': -1.23,
        'cash': 25000.00,
        'num_positions': 0,
        'win_rate': 0.0,
        'total_trades': 0,
        'last_update': datetime.now()
    }
    
    status_msg = format_portfolio_status(edge_case_status)
    print("Scenario 4 - Edge case portfolio (no trades, negative return):")
    print(status_msg)
    print("\n" + "="*50 + "\n")

def test_characters_that_break_telegram():
    """Test specific character combinations that break Telegram parsing"""
    print("Testing character combinations that break Telegram:")
    
    problematic_strings = [
        "Price went from $25.30 to $28.75 (+13.6%)",
        "Symbol: [GARAN.IS] - Entry: $29.50 (Target: $32.00)",
        "Reason: RSI_oversold & volume_spike detected",
        "Analysis shows: buy_signal = True (confidence: 85%)",
        "Trade: BUY 100 AKBNK.IS @ $28.40 = $2,840.00",
        "P&L: +$145.50 (+5.13%) after 7 days",
        "Status: position_size > max_allowed (warning!)",
        "Time: 2024-01-15T14:30:25+03:00"
    ]
    
    for i, text in enumerate(problematic_strings, 1):
        escaped = escape_markdown_v1(text)
        print(f"Test {i}:")
        print(f"Original: {text}")
        print(f"Escaped:  {escaped}")
        print()

def test_byte_offset_error_simulation():
    """Simulate the specific byte offset error from the original issue"""
    print("Testing byte offset error simulation:")
    
    # This simulates a message that could cause "Can't find end of the entity starting at byte offset 398"
    long_message = format_trade_message(
        action="BUY",
        symbol="GUBRF.IS",
        shares=250,
        price=142.75,
        reason="Technical analysis indicates strong momentum with RSI(14)=45.2, MACD signal line cross, volume 2.3x average, price broke resistance at 140.50 with target 155.00 (+8.6% potential)"
    )
    
    print("Long message that could cause byte offset errors:")
    print(f"Message length: {len(long_message)} characters")
    print(long_message)
    print("\n" + "="*50 + "\n")

def test_notification_messages():
    """Test notification messages that are sent during trading"""
    print("Testing notification messages:")
    
    # Hourly update message
    hourly_msg = (
        f"📊 Hourly Update\n"
        f"Value: {format_currency(125750.45)}\n"
        f"Return: {format_percentage(15.75)}\n"
        f"Positions: 8"
    )
    print("Hourly update message:")
    print(hourly_msg)
    print()
    
    # Timeout message
    timeout_msg = f"⏰ Trade timeout: BUY 100 {escape_markdown_v1('GARAN.IS')}"
    print("Timeout message:")
    print(timeout_msg)
    print()
    
    # Status change messages
    telegram_msg = f"Telegram notifications {escape_markdown_v1('enabled')}"
    confirm_msg = f"Trade confirmations {escape_markdown_v1('disabled')}"
    print("Status change messages:")
    print(telegram_msg)
    print(confirm_msg)
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("REAL-WORLD TELEGRAM SCENARIO TESTS")
    print("=" * 60)
    print()
    
    test_actual_error_scenarios()
    test_characters_that_break_telegram()
    test_byte_offset_error_simulation()
    test_notification_messages()
    
    print("=" * 60)
    print("All real-world scenario tests completed!")
    print("These messages should now work without Telegram parsing errors.")
    print("=" * 60)