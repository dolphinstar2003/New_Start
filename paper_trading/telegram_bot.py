#!/usr/bin/env python3
"""
Telegram Bot Integration for Paper Trading
Provides notifications and control for paper trading system
"""
import os
import json
import asyncio
from datetime import datetime
from typing import Optional, Dict, List
import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from pathlib import Path
import pandas as pd
import logging
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.telegram_utils import (
    escape_markdown_v1, format_currency, format_percentage, format_symbol,
    format_trade_message, format_portfolio_status, format_position_list,
    format_trade_history, format_performance_metrics, format_opportunities
)

logger = logging.getLogger(__name__)

# Telegram bot configuration
TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),
    'admin_users': []
}

class PaperTradingBot:
    """Telegram bot for paper trading notifications and control"""
    
    def __init__(self, paper_trader=None):
        self.paper_trader = paper_trader
        self.bot = AsyncTeleBot(TELEGRAM_CONFIG['bot_token'])
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        self.admin_users = TELEGRAM_CONFIG['admin_users']
        self.is_running = False
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup message and callback handlers"""
        
        @self.bot.message_handler(commands=['start'])
        async def start_command(message):
            """Handle /start command"""
            if not self._is_authorized(message.from_user.id):
                await self.bot.send_message(message.chat.id, "❌ Unauthorized access")
                return
            
            welcome_text = """
🚀 *Paper Trading Bot*

Welcome! I'll help you manage your paper trading portfolios\\.

📊 *Portfolio Commands:*
/portfolios \\- Show all portfolios
/status \\[strategy\\] \\- Show portfolio status
/positions \\[strategy\\] \\- Show positions
/performance \\[strategy\\] \\- Show performance

💼 *Trading Commands:*
/signals \\- Show current signals
/execute \\[strategy\\] \\- Execute trading cycle
/report \\- Generate daily report

⚙️ *System Commands:*
/strategies \\- Show available strategies
/help \\- Show this help message

Example: `/status aggressive`
            """
            
            await self.bot.send_message(
                message.chat.id,
                welcome_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['portfolios'])
        async def portfolios_command(message):
            """Handle /portfolios command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            summary_df = self.paper_trader.portfolio_manager.get_summary()
            
            portfolios_text = "📊 *All Portfolios*\\n━━━━━━━━━━━━━━━━━━━━\\n\\n"
            
            for _, row in summary_df.iterrows():
                name = escape_markdown_v1(row['name'])
                value = format_currency(row['portfolio_value'])
                ret = format_percentage(row['total_return'])
                pos = row['num_positions']
                
                portfolios_text += f"*{name}:*\\n"
                portfolios_text += f"  💰 Value: {value}\\n"
                portfolios_text += f"  📈 Return: {ret}\\n"
                portfolios_text += f"  📍 Positions: {pos}\\n\\n"
            
            # Add best performer
            if not summary_df.empty:
                best_idx = summary_df['total_return'].idxmax()
                best_name = escape_markdown_v1(summary_df.loc[best_idx, 'name'])
                best_return = format_percentage(summary_df.loc[best_idx, 'total_return'])
                portfolios_text += f"🏆 *Best:* {best_name} \\({best_return}\\)"
            
            await self.bot.send_message(
                message.chat.id,
                portfolios_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['status'])
        async def status_command(message):
            """Handle /status [strategy] command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            # Extract strategy name
            parts = message.text.split()
            strategy = parts[1] if len(parts) > 1 else 'balanced'
            
            portfolio = self.paper_trader.portfolio_manager.get_portfolio(strategy)
            if not portfolio:
                await self.bot.send_message(
                    message.chat.id,
                    f"❌ Portfolio '{strategy}' not found\\. Available: aggressive, balanced, conservative"
                )
                return
            
            metrics = portfolio.get_performance_metrics()
            
            status_text = f"""
📊 *{escape_markdown_v1(strategy.upper())} Portfolio Status*
━━━━━━━━━━━━━━━━━━━━

💰 *Value:* {format_currency(metrics['portfolio_value'])}
📈 *Return:* {format_percentage(metrics['total_return'])}
💵 *Cash:* {format_currency(metrics['cash'])}
📍 *Positions:* {metrics['num_positions']}

📊 *Performance:*
  • Win Rate: {format_percentage(metrics['win_rate'], False)}
  • Sharpe: {metrics['sharpe_ratio']:.2f}
  • Max DD: {format_percentage(metrics['max_drawdown'], False)}
  • Total Trades: {metrics['total_trades']}
            """
            
            # Add positions if any
            if portfolio.positions:
                status_text += "\\n\\n*Current Positions:*\\n"
                for symbol, pos in portfolio.positions.items():
                    pnl_pct = pos.get_pnl_percentage()
                    status_text += f"  • {escape_markdown_v1(symbol)}: "
                    status_text += f"{pos.shares} @ {format_currency(pos.entry_price)} "
                    status_text += f"\\({format_percentage(pnl_pct)}\\)\\n"
            
            await self.bot.send_message(
                message.chat.id,
                status_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['positions'])
        async def positions_command(message):
            """Handle /positions [strategy] command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            # Extract strategy name
            parts = message.text.split()
            strategy = parts[1] if len(parts) > 1 else None
            
            positions_text = "📍 *Active Positions*\\n━━━━━━━━━━━━━━━━━━━━\\n\\n"
            has_positions = False
            
            for strat_name, config in self.paper_trader.strategies.items():
                if strategy and strat_name != strategy:
                    continue
                
                portfolio = self.paper_trader.portfolio_manager.get_portfolio(config['portfolio_name'])
                if portfolio and portfolio.positions:
                    has_positions = True
                    positions_text += f"*{escape_markdown_v1(strat_name.upper())}:*\\n"
                    
                    for symbol, pos in portfolio.positions.items():
                        current_price = pos.current_price
                        pnl = pos.get_pnl()
                        pnl_pct = pos.get_pnl_percentage()
                        
                        positions_text += f"  • {escape_markdown_v1(symbol)}: "
                        positions_text += f"{pos.shares} shares\\n"
                        positions_text += f"    Entry: {format_currency(pos.entry_price)}\\n"
                        positions_text += f"    Current: {format_currency(current_price)}\\n"
                        positions_text += f"    P&L: {format_currency(pnl)} "
                        positions_text += f"\\({format_percentage(pnl_pct)}\\)\\n\\n"
            
            if not has_positions:
                positions_text += "No active positions"
            
            await self.bot.send_message(
                message.chat.id,
                positions_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['signals'])
        async def signals_command(message):
            """Handle /signals command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            signals_text = "🔔 *Current Signals*\\n━━━━━━━━━━━━━━━━━━━━\\n\\n"
            
            for strategy in ['aggressive', 'balanced', 'conservative']:
                logger.info(f"Scanning signals for {strategy} strategy")
                signals = self.paper_trader.signal_generator.scan_all_symbols(strategy)
                
                if signals:
                    signals_text += f"*{escape_markdown_v1(strategy.upper())}:*\\n"
                    for symbol, signal_data in signals.items():
                        signal_type = "BUY" if signal_data['signal'] == 1 else "SELL"
                        sym_escaped = escape_markdown_v1(symbol)
                        price = format_currency(signal_data['price'])
                        signals_text += f"  • {signal_type} {sym_escaped} @ {price}\\n"
                    signals_text += "\\n"
            
            await self.bot.send_message(
                message.chat.id,
                signals_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['execute'])
        async def execute_command(message):
            """Handle /execute [strategy] command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            # Extract strategy name
            parts = message.text.split()
            strategy = parts[1] if len(parts) > 1 else None
            
            await self.bot.send_message(message.chat.id, "⏳ Executing trading cycle...")
            
            if strategy:
                if strategy in self.paper_trader.strategies:
                    self.paper_trader.execute_strategy(strategy)
                    await self.bot.send_message(
                        message.chat.id,
                        f"✅ Executed {strategy} strategy"
                    )
                else:
                    await self.bot.send_message(
                        message.chat.id,
                        f"❌ Unknown strategy: {strategy}"
                    )
            else:
                self.paper_trader.run_trading_cycle()
                await self.bot.send_message(message.chat.id, "✅ Trading cycle completed")
        
        @self.bot.message_handler(commands=['report'])
        async def report_command(message):
            """Handle /report command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            self.paper_trader.generate_daily_report()
            
            # Send summary
            summary_df = self.paper_trader.portfolio_manager.get_summary()
            
            report_text = f"""
📊 *Daily Report \\- {escape_markdown_v1(datetime.now().strftime('%Y-%m-%d'))}*
━━━━━━━━━━━━━━━━━━━━
"""
            
            for _, row in summary_df.iterrows():
                name = escape_markdown_v1(row['name'])
                value = format_currency(row['portfolio_value'])
                ret = format_percentage(row['total_return'])
                pnl = format_currency(row['total_pnl'])
                
                report_text += f"\\n*{name}:*\\n"
                report_text += f"  💰 Value: {value}\\n"
                report_text += f"  📈 Return: {ret}\\n"
                report_text += f"  💵 P&L: {pnl}\\n"
            
            await self.bot.send_message(
                message.chat.id,
                report_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['strategies'])
        async def strategies_command(message):
            """Handle /strategies command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            strategies_text = """
📋 *Available Strategies*
━━━━━━━━━━━━━━━━━━━━

*Aggressive:*
  • Signal: Supertrend only
  • Fast entries/exits
  • Higher risk/reward

*Balanced:*
  • Signals: Supertrend \\+ ADX \\+ MACD
  • 2 out of 3 confirmation
  • Moderate risk

*Conservative:*
  • Signals: 5 indicators
  • 3\\+ confirmations required
  • Lower risk
  • VixFix filter for volatility
            """
            
            await self.bot.send_message(
                message.chat.id,
                strategies_text,
                parse_mode='Markdown'
            )
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        if not self.admin_users:
            return True
        return user_id in self.admin_users
    
    async def send_trade_notification(self, strategy: str, action: str, symbol: str, 
                                     shares: int, price: float, reason: str = "Signal"):
        """Send trade execution notification"""
        emoji = "🟢" if action == "BUY" else "🔴"
        
        notification_text = f"""
{emoji} *Trade Executed \\- {escape_markdown_v1(strategy.upper())}*
━━━━━━━━━━━━━━━━━━━━

📊 *Action:* {action}
📈 *Symbol:* {escape_markdown_v1(symbol)}
💼 *Shares:* {shares}
💰 *Price:* {format_currency(price)}
💵 *Total:* {format_currency(shares * price)}
📍 *Reason:* {escape_markdown_v1(reason)}
        """
        
        await self.send_notification(notification_text, "trade")
    
    async def send_position_closed_notification(self, strategy: str, symbol: str, 
                                               entry_price: float, exit_price: float,
                                               shares: int, profit: float, reason: str):
        """Send position closed notification"""
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        emoji = "🟢" if profit > 0 else "🔴"
        
        notification_text = f"""
{emoji} *Position Closed \\- {escape_markdown_v1(strategy.upper())}*
━━━━━━━━━━━━━━━━━━━━

📈 *Symbol:* {escape_markdown_v1(symbol)}
💼 *Shares:* {shares}
💰 *Entry:* {format_currency(entry_price)}
💰 *Exit:* {format_currency(exit_price)}
📊 *P&L:* {format_currency(profit)} \\({format_percentage(pnl_pct)}\\)
📍 *Reason:* {escape_markdown_v1(reason)}
        """
        
        await self.send_notification(notification_text, "trade")
    
    async def send_daily_summary(self):
        """Send daily trading summary"""
        summary_df = self.paper_trader.portfolio_manager.get_summary()
        
        summary_text = f"""
📊 *Daily Summary \\- {escape_markdown_v1(datetime.now().strftime('%Y-%m-%d %H:%M'))}*
━━━━━━━━━━━━━━━━━━━━
"""
        
        total_value = 0
        total_initial = 0
        
        for _, row in summary_df.iterrows():
            name = escape_markdown_v1(row['name'])
            value = row['portfolio_value']
            ret = row['total_return']
            positions = row['num_positions']
            win_rate = row['win_rate']
            
            total_value += value
            total_initial += 50000  # Initial capital per strategy
            
            summary_text += f"""
*{name}:*
  💰 Value: {format_currency(value)}
  📈 Return: {format_percentage(ret)}
  📍 Positions: {positions}
  🎯 Win Rate: {format_percentage(win_rate, False)}
"""
        
        # Overall performance
        total_return_pct = ((total_value - total_initial) / total_initial) * 100
        summary_text += f"""
━━━━━━━━━━━━━━━━━━━━
*Total Portfolio:*
  💰 Value: {format_currency(total_value)}
  📈 Return: {format_percentage(total_return_pct)}
        """
        
        await self.send_notification(summary_text, "info")
    
    async def send_notification(self, message: str, notification_type: str = "info"):
        """Send notification to Telegram"""
        try:
            if self.chat_id:
                await self.bot.send_message(
                    self.chat_id,
                    message,
                    parse_mode='Markdown'
                )
            logger.info(f"Telegram notification sent: {notification_type}")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    async def start(self):
        """Start the Telegram bot"""
        self.is_running = True
        logger.info("Starting Paper Trading Telegram bot...")
        
        await self.send_notification(
            "*Paper Trading Bot Started* 🚀\\n\\nReady to manage your portfolios!",
            "success"
        )
        
        await self.bot.polling(non_stop=True)
    
    async def stop(self):
        """Stop the Telegram bot"""
        self.is_running = False
        
        await self.send_notification(
            "*Paper Trading Bot Stopping* 🛑",
            "warning"
        )
        
        await self.bot.close_session()
        logger.info("Paper Trading Telegram bot stopped")


# Standalone bot runner
async def run_bot(paper_trader):
    """Run Telegram bot with paper trader"""
    bot = PaperTradingBot(paper_trader)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()


if __name__ == "__main__":
    # Test bot connection
    print("Testing Paper Trading Telegram Bot...")
    
    # Load config
    config_file = Path(__file__).parent.parent / 'telegram_config.json'
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        TELEGRAM_CONFIG.update(config)
    
    async def test():
        bot = PaperTradingBot()
        await bot.send_notification(
            "*Paper Trading Bot Test* ✅\\n\\nIf you see this, the bot is configured correctly!",
            "success"
        )
        print("✅ Test message sent!")
    
    asyncio.run(test())