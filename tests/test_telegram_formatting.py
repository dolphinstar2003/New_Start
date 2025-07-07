#!/usr/bin/env python3
"""
Test script for Telegram message formatting and escaping
"""
import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))

from utils.telegram_utils import (
    escape_markdown_v1, format_currency, format_percentage, 
    format_trade_message, format_portfolio_status
)

def test_escape_function():
    """Test the escape_markdown_v1 function"""
    print("Testing escape_markdown_v1 function:")
    
    test_cases = [
        "Simple text",
        "Text with underscore_symbol",
        "Text with *asterisk*",
        "Price: $1,234.56",
        "Return: +15.32%",
        "GARAN.IS stock",
        "P&L: -$123.45 (-2.3%)",
        "Symbol: [AKBNK.IS]",
        "Reason: Stop-loss (triggered)",
        "Time: 2024-01-15 14:30:25",
        "Range: 100-200 TL"
    ]
    
    for test_case in test_cases:
        escaped = escape_markdown_v1(test_case)
        print(f"Original: {test_case}")
        print(f"Escaped:  {escaped}")
        print()

def test_currency_formatting():
    """Test currency formatting"""
    print("Testing format_currency function:")
    
    test_amounts = [1234.56, -567.89, 0, 1000000, 0.12]
    
    for amount in test_amounts:
        formatted = format_currency(amount)
        print(f"Amount: {amount} -> Formatted: {formatted}")
    print()

def test_percentage_formatting():
    """Test percentage formatting"""
    print("Testing format_percentage function:")
    
    test_percentages = [15.32, -2.45, 0, 100.0, 0.01]
    
    for pct in test_percentages:
        formatted = format_percentage(pct)
        formatted_no_plus = format_percentage(pct, show_plus=False)
        print(f"Percentage: {pct} -> With plus: {formatted}, Without plus: {formatted_no_plus}")
    print()

def test_trade_message():
    """Test trade message formatting"""
    print("Testing format_trade_message function:")
    
    # BUY message
    buy_msg = format_trade_message(
        action="BUY",
        symbol="GARAN.IS",
        shares=100,
        price=29.75,
        reason="Entry Signal (Score: 65.2)"
    )
    print("BUY Message:")
    print(buy_msg)
    print()
    
    # SELL message with profit
    sell_msg = format_trade_message(
        action="SELL",
        symbol="AKBNK.IS",
        shares=150,
        price=31.20,
        reason="Take Profit",
        profit=234.50
    )
    print("SELL Message:")
    print(sell_msg)
    print()

def test_portfolio_status():
    """Test portfolio status formatting"""
    print("Testing format_portfolio_status function:")
    
    mock_status = {
        'portfolio_value': 125750.45,
        'total_return': 25750.45,
        'total_return_pct': 25.75,
        'cash': 15420.30,
        'num_positions': 8,
        'win_rate': 68.5,
        'total_trades': 47
    }
    
    formatted_status = format_portfolio_status(mock_status)
    print("Portfolio Status:")
    print(formatted_status)
    print()

def test_problematic_characters():
    """Test characters that commonly cause Telegram parsing errors"""
    print("Testing problematic characters:")
    
    problematic_texts = [
        "Test_with_underscores_everywhere",
        "Multiple*asterisks*in*text",
        "Square [brackets] and (parentheses)",
        "Backticks `code` and ~strikethrough~",
        "Special chars: #hashtag +plus -minus =equals |pipe",
        "Dots and exclamation! Also {braces}",
        "URL-like: https://example.com/path?param=value",
        "Math: 2 + 2 = 4, 10 - 5 = 5"
    ]
    
    for text in problematic_texts:
        escaped = escape_markdown_v1(text)
        print(f"Original: {text}")
        print(f"Escaped:  {escaped}")
        print()

if __name__ == "__main__":
    print("=" * 60)
    print("TELEGRAM FORMATTING TEST SUITE")
    print("=" * 60)
    print()
    
    test_escape_function()
    test_currency_formatting()
    test_percentage_formatting()
    test_trade_message()
    test_portfolio_status()
    test_problematic_characters()
    
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)