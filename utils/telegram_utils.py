"""
Telegram utility functions for message formatting and escaping
"""
import re
from typing import Any, Dict, List


def escape_markdown(text: str) -> str:
    """
    Escape special characters in text for Telegram Markdown V2
    
    Telegram Markdown V2 special characters that need escaping:
    - _ (underscore)
    - * (asterisk) 
    - [ ] (square brackets)
    - ( ) (parentheses)
    - ~ (tilde)
    - ` (backtick)
    - > (greater than)
    - # (hash)
    - + (plus)
    - - (minus/hyphen)
    - = (equals)
    - | (pipe)
    - { } (curly braces)
    - . (dot)
    - ! (exclamation mark)
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Define escape mappings
    escape_chars = {
        '_': '\\_',
        '*': '\\*',
        '[': '\\[',
        ']': '\\]',
        '(': '\\(',
        ')': '\\)',
        '~': '\\~',
        '`': '\\`',
        '>': '\\>',
        '#': '\\#',
        '+': '\\+',
        '-': '\\-',
        '=': '\\=',
        '|': '\\|',
        '{': '\\{',
        '}': '\\}',
        '.': '\\.',
        '!': '\\!'
    }
    
    # Escape each character
    for char, escaped in escape_chars.items():
        text = text.replace(char, escaped)
    
    return text


def escape_markdown_v1(text: str) -> str:
    """
    Escape special characters for Telegram Markdown V1 (legacy)
    
    Only escapes: * _ [ ] ( ) ~ `
    """
    if not isinstance(text, str):
        text = str(text)
    
    # V1 escape mappings
    escape_chars = {
        '_': '\\_',
        '*': '\\*',
        '[': '\\[',
        ']': '\\]',
        '(': '\\(',
        ')': '\\)',
        '~': '\\~',
        '`': '\\`'
    }
    
    # Escape each character
    for char, escaped in escape_chars.items():
        text = text.replace(char, escaped)
    
    return text


def format_currency(amount: float, symbol: str = "$") -> str:
    """Format currency amount with proper escaping"""
    if amount >= 0:
        formatted = f"{symbol}{amount:,.2f}"
    else:
        formatted = f"-{symbol}{abs(amount):,.2f}"
    
    return escape_markdown_v1(formatted)


def format_percentage(value: float, show_plus: bool = True) -> str:
    """Format percentage with proper escaping"""
    if show_plus and value > 0:
        formatted = f"+{value:.2f}%"
    else:
        formatted = f"{value:.2f}%"
    
    return escape_markdown_v1(formatted)


def format_symbol(symbol: str) -> str:
    """Format trading symbol with proper escaping"""
    return escape_markdown_v1(symbol)


def format_trade_message(action: str, symbol: str, shares: int, price: float, 
                        reason: str, profit: float = None) -> str:
    """
    Format a trade message with proper escaping for all components
    """
    emoji = "🟢" if action == "BUY" else "🔴"
    
    # Escape all text components
    symbol_escaped = escape_markdown_v1(symbol)
    reason_escaped = escape_markdown_v1(reason)
    
    if action == "BUY":
        message = f"""
{emoji} *{action} SIGNAL*
━━━━━━━━━━━━━━━━━━━━

📊 *Symbol:* {symbol_escaped}
📈 *Shares:* {shares}
💰 *Price:* {format_currency(price)}
💵 *Total:* {format_currency(shares * price)}
📍 *Reason:* {reason_escaped}

⏰ *Time:* {escape_markdown_v1(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
"""
    else:  # SELL
        profit_emoji = "🔴" if profit and profit < 0 else "🟢"
        if profit is not None:
            profit_pct = (profit / (shares * price - profit)) * 100
            profit_text = f"{format_currency(profit)} ({format_percentage(profit_pct)})"
        else:
            profit_text = "N/A"
        
        message = f"""
{profit_emoji} *{action} SIGNAL*
━━━━━━━━━━━━━━━━━━━━

📊 *Symbol:* {symbol_escaped}
📈 *Shares:* {shares}
💰 *Price:* {format_currency(price)}
💵 *Total:* {format_currency(shares * price)}
💸 *Profit:* {profit_text}
📍 *Reason:* {reason_escaped}

⏰ *Time:* {escape_markdown_v1(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
"""
    
    return message.strip()


def format_portfolio_status(status: Dict[str, Any]) -> str:
    """Format portfolio status message with proper escaping"""
    
    message = f"""
📊 *Portfolio Status*
━━━━━━━━━━━━━━━━━━━━

💰 *Value:* {format_currency(status['portfolio_value'])}
📈 *Return:* {format_currency(status['total_return'])} ({format_percentage(status['total_return_pct'])})
💵 *Cash:* {format_currency(status['cash'])}
📍 *Positions:* {status['num_positions']}
🎯 *Win Rate:* {format_percentage(status['win_rate'], False)}
📊 *Total Trades:* {status['total_trades']}
"""
    
    if status.get('last_update'):
        last_update = escape_markdown_v1(status['last_update'].strftime('%Y-%m-%d %H:%M:%S'))
        message += f"⏰ *Last Update:* {last_update}"
    
    return message.strip()


def format_position_list(positions: List[Dict[str, Any]]) -> str:
    """Format positions list with proper escaping"""
    
    if not positions:
        return "📊 No active positions"
    
    message = "💼 *Current Positions*\n━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for pos in positions:
        emoji = "🟢" if pos['profit_pct'] > 0 else "🔴"
        symbol_escaped = escape_markdown_v1(pos['symbol'])
        
        message += f"{emoji} *{symbol_escaped}*\n"
        message += f"   Shares: {pos['shares']}\n"
        message += f"   Entry: {format_currency(pos['entry_price'])} → Current: {format_currency(pos['current_price'])}\n"
        message += f"   Value: {format_currency(pos['value'])}\n"
        message += f"   P&L: {format_currency(pos['profit'])} ({format_percentage(pos['profit_pct'])})\n"
        message += f"   Days: {pos['holding_days']}\n\n"
    
    return message.strip()


def format_trade_history(trades: List[Dict[str, Any]], limit: int = 10) -> str:
    """Format trade history with proper escaping"""
    
    if not trades:
        return "📊 No trades executed yet"
    
    recent_trades = trades[-limit:] if len(trades) > limit else trades
    
    message = "📈 *Recent Trades*\n━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for trade in recent_trades:
        emoji = "✅" if trade['profit'] > 0 else "❌"
        symbol_escaped = escape_markdown_v1(trade['symbol'])
        reason_escaped = escape_markdown_v1(trade['reason'])
        
        message += f"{emoji} *{symbol_escaped}*\n"
        message += f"   Entry: {format_currency(trade['entry_price'])} → Exit: {format_currency(trade['exit_price'])}\n"
        message += f"   P&L: {format_currency(trade['profit'])} ({format_percentage(trade['profit_pct'])})\n"
        message += f"   Reason: {reason_escaped}\n\n"
    
    return message.strip()


def format_performance_metrics(metrics: Dict[str, Any]) -> str:
    """Format performance metrics with proper escaping"""
    
    message = f"""
📈 *Performance Metrics*
━━━━━━━━━━━━━━━━━━━━

📊 *Returns:*
• Total Profit: {format_currency(metrics['total_profit'])}
• Avg Return: {format_percentage(metrics['avg_profit_pct'], False)}
• Best Trade: {format_percentage(metrics['max_win'])}
• Worst Trade: {format_percentage(metrics['max_loss'])}

🎯 *Win/Loss:*
• Total Trades: {metrics['total_trades']}
• Win Rate: {format_percentage(metrics['win_rate'], False)}
• Avg Win: {format_percentage(metrics['avg_win'], False)}
• Avg Loss: {format_percentage(metrics['avg_loss'], False)}

📉 *Risk Metrics:*
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Max Drawdown: {format_percentage(metrics['max_drawdown'], False)}
• Profit Factor: {metrics['profit_factor']:.2f}
"""
    
    return message.strip()


def format_opportunities(opportunities: List[Dict[str, Any]], limit: int = 10) -> str:
    """Format market opportunities with proper escaping"""
    
    if not opportunities:
        return "📊 No opportunities available"
    
    top_opps = opportunities[:limit]
    
    message = "🎯 *Top Market Opportunities*\n━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for i, opp in enumerate(top_opps, 1):
        emoji = "🟢" if opp['score'] >= 60 else "🟡" if opp['score'] >= 40 else "🔴"
        position_emoji = "📍" if opp['in_position'] else ""
        symbol_escaped = escape_markdown_v1(opp['symbol'])
        
        message += f"{i}. {emoji} *{symbol_escaped}* {position_emoji}\n"
        message += f"   Score: {opp['score']:.1f}\n"
        message += f"   Price: {format_currency(opp['price'])}\n"
        
        if 'momentum_1h' in opp:
            message += f"   Momentum: {format_percentage(opp['momentum_1h'])} (1h), {format_percentage(opp.get('momentum_day', 0))} (day)\n"
        
        message += "\n"
    
    return message.strip()


# Import datetime for the trade message formatting
from datetime import datetime