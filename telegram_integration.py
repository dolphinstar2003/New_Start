"""
Telegram Integration for Paper Trading Module
Provides notifications and control via Telegram bot
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
from loguru import logger

# Telegram bot configuration
TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),  # Set your bot token
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', ''),      # Set your chat ID
    'admin_users': [],  # List of authorized user IDs
}

class TelegramBot:
    """Telegram bot for paper trading notifications and control"""
    
    def __init__(self, paper_trader=None):
        self.paper_trader = paper_trader
        self.bot = AsyncTeleBot(TELEGRAM_CONFIG['bot_token'])
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        self.admin_users = TELEGRAM_CONFIG['admin_users']
        self.is_running = False
        
        # Command states
        self.pending_confirmations = {}
        
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
🚀 *Dynamic Portfolio Optimizer - Paper Trading Bot*

Welcome! I'm your trading assistant. Here are the available commands:

📊 *Portfolio Commands:*
/status - Show portfolio status
/positions - Show current positions
/trades - Show recent trades
/performance - Show performance metrics

💼 *Trading Commands:*
/start_trading - Enable auto trading
/stop_trading - Disable auto trading
/force_check - Force position check

📈 *Market Commands:*
/opportunities - Show top opportunities
/analyze [SYMBOL] - Analyze specific symbol

⚙️ *System Commands:*
/settings - Show current settings
/help - Show this help message

All trades require confirmation before execution.
            """
            
            await self.bot.send_message(
                message.chat.id,
                welcome_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['status'])
        async def status_command(message):
            """Handle /status command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            status = self.paper_trader.get_portfolio_status()
            
            status_text = f"""
📊 *Portfolio Status*
━━━━━━━━━━━━━━━━━━━━

💰 *Value:* ${status['portfolio_value']:,.2f}
📈 *Return:* ${status['total_return']:,.2f} ({status['total_return_pct']:+.2f}%)
💵 *Cash:* ${status['cash']:,.2f}
📍 *Positions:* {status['num_positions']}/{self.paper_trader.PORTFOLIO_PARAMS['max_positions']}
🎯 *Win Rate:* {status['win_rate']:.1f}%
📊 *Total Trades:* {status['total_trades']}

🔄 *Auto Trading:* {'✅ Enabled' if self.paper_trader.auto_trade_enabled else '❌ Disabled'}
⏰ *Last Update:* {status['last_update'].strftime('%Y-%m-%d %H:%M:%S') if status['last_update'] else 'N/A'}
            """
            
            # Add inline keyboard for quick actions
            keyboard = InlineKeyboardMarkup()
            if self.paper_trader.auto_trade_enabled:
                keyboard.add(InlineKeyboardButton("🛑 Stop Trading", callback_data="stop_trading"))
            else:
                keyboard.add(InlineKeyboardButton("▶️ Start Trading", callback_data="start_trading"))
            
            keyboard.add(
                InlineKeyboardButton("📊 Positions", callback_data="show_positions"),
                InlineKeyboardButton("📈 Performance", callback_data="show_performance")
            )
            
            await self.bot.send_message(
                message.chat.id,
                status_text,
                parse_mode='Markdown',
                reply_markup=keyboard
            )
        
        @self.bot.message_handler(commands=['positions'])
        async def positions_command(message):
            """Handle /positions command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            await self._send_positions_update(message.chat.id)
        
        @self.bot.message_handler(commands=['trades'])
        async def trades_command(message):
            """Handle /trades command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            trades_df = self.paper_trader.get_trade_history()
            
            if trades_df.empty:
                await self.bot.send_message(message.chat.id, "📊 No trades executed yet")
                return
            
            # Get last 10 trades
            recent_trades = trades_df.tail(10)
            
            trades_text = "📈 *Recent Trades*\n━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for _, trade in recent_trades.iterrows():
                emoji = "✅" if trade['profit'] > 0 else "❌"
                trades_text += f"{emoji} *{trade['symbol']}*\n"
                trades_text += f"   Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f}\n"
                trades_text += f"   P&L: ${trade['profit']:.2f} ({trade['profit_pct']:+.2f}%)\n"
                trades_text += f"   Reason: {trade['reason']}\n\n"
            
            await self.bot.send_message(
                message.chat.id,
                trades_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['performance'])
        async def performance_command(message):
            """Handle /performance command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            await self._send_performance_update(message.chat.id)
        
        @self.bot.message_handler(commands=['start_trading'])
        async def start_trading_command(message):
            """Handle /start_trading command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            # Request confirmation
            keyboard = InlineKeyboardMarkup()
            keyboard.add(
                InlineKeyboardButton("✅ Confirm", callback_data="confirm_start_trading"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_action")
            )
            
            await self.bot.send_message(
                message.chat.id,
                "⚠️ *Confirm Action*\n\nStart automatic trading?",
                parse_mode='Markdown',
                reply_markup=keyboard
            )
        
        @self.bot.message_handler(commands=['stop_trading'])
        async def stop_trading_command(message):
            """Handle /stop_trading command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            # Request confirmation
            keyboard = InlineKeyboardMarkup()
            keyboard.add(
                InlineKeyboardButton("✅ Confirm", callback_data="confirm_stop_trading"),
                InlineKeyboardButton("❌ Cancel", callback_data="cancel_action")
            )
            
            await self.bot.send_message(
                message.chat.id,
                "⚠️ *Confirm Action*\n\nStop automatic trading?",
                parse_mode='Markdown',
                reply_markup=keyboard
            )
        
        @self.bot.message_handler(commands=['opportunities'])
        async def opportunities_command(message):
            """Handle /opportunities command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            opportunities = self.paper_trader.evaluate_all_opportunities()
            
            if not opportunities:
                await self.bot.send_message(message.chat.id, "📊 No opportunities available")
                return
            
            # Get top 10 opportunities
            top_opps = opportunities[:10]
            
            opps_text = "🎯 *Top Market Opportunities*\n━━━━━━━━━━━━━━━━━━━━\n\n"
            
            for i, opp in enumerate(top_opps, 1):
                emoji = "🟢" if opp['score'] >= 60 else "🟡" if opp['score'] >= 40 else "🔴"
                position_emoji = "📍" if opp['in_position'] else ""
                
                opps_text += f"{i}. {emoji} *{opp['symbol']}* {position_emoji}\n"
                opps_text += f"   Score: {opp['score']:.1f}\n"
                opps_text += f"   Price: ${opp['price']:.2f}\n"
                opps_text += f"   Momentum: {opp['momentum_1h']:+.1f}% (1h), {opp['momentum_day']:+.1f}% (day)\n\n"
            
            await self.bot.send_message(
                message.chat.id,
                opps_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['analyze'])
        async def analyze_command(message):
            """Handle /analyze [SYMBOL] command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            # Extract symbol from command
            parts = message.text.split()
            if len(parts) < 2:
                await self.bot.send_message(
                    message.chat.id,
                    "Usage: /analyze SYMBOL\nExample: /analyze GARAN.IS"
                )
                return
            
            symbol = parts[1].upper()
            if not symbol.endswith('.IS'):
                symbol += '.IS'
            
            if symbol not in self.paper_trader.market_data:
                await self.bot.send_message(
                    message.chat.id,
                    f"❌ No data available for {symbol}"
                )
                return
            
            # Get symbol data
            market_data = self.paper_trader.market_data[symbol]
            score = self.paper_trader.calculate_opportunity_score(symbol, market_data)
            
            # Check if in position
            in_position = symbol in self.paper_trader.portfolio['positions']
            position_info = ""
            
            if in_position:
                pos = self.paper_trader.portfolio['positions'][symbol]
                current_price = market_data.get('last_price', pos['entry_price'])
                profit_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                
                position_info = f"""
📍 *Current Position:*
   Shares: {pos['shares']}
   Entry: ${pos['entry_price']:.2f}
   Current: ${current_price:.2f}
   P&L: {profit_pct:+.2f}%
"""
            
            analysis_text = f"""
📊 *Analysis: {symbol}*
━━━━━━━━━━━━━━━━━━━━

🎯 *Opportunity Score:* {score:.1f}/100
💰 *Last Price:* ${market_data.get('last_price', 0):.2f}
📈 *Price Change:*
   1h: {market_data.get('price_change_1h', 0)*100:+.2f}%
   Day: {market_data.get('price_change_day', 0)*100:+.2f}%
📊 *Volume Ratio:* {market_data.get('volume_ratio', 1):.2f}x
🔄 *Last Update:* {market_data.get('last_update', datetime.now()).strftime('%H:%M:%S')}
{position_info}
📌 *Recommendation:* {"BUY" if score >= 40 and not in_position else "HOLD" if in_position else "WAIT"}
            """
            
            await self.bot.send_message(
                message.chat.id,
                analysis_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['settings'])
        async def settings_command(message):
            """Handle /settings command"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Paper trader not initialized")
                return
            
            params = self.paper_trader.PORTFOLIO_PARAMS
            
            settings_text = f"""
⚙️ *Trading Settings*
━━━━━━━━━━━━━━━━━━━━

📊 *Position Management:*
• Base Position: {params['base_position_pct']*100:.0f}%
• Max Position: {params['max_position_pct']*100:.0f}%
• Min Position: {params['min_position_pct']*100:.0f}%
• Max Positions: {params['max_positions']}

🎯 *Entry/Exit:*
• Entry Threshold: {params['entry_threshold']}
• Exit Threshold: {params['exit_threshold']}
• Rotation Threshold: {params['rotation_threshold']}

🛡️ *Risk Management:*
• Stop Loss: {params['stop_loss']*100:.0f}%
• Take Profit: {params['take_profit']*100:.0f}%
• Trailing Start: {params['trailing_start']*100:.0f}%
• Trailing Distance: {params['trailing_distance']*100:.0f}%

🔄 *Rotation:*
• Enabled: {"✅" if params['enable_rotation'] else "❌"}
• Check Days: {params['rotation_check_days']}
• Min Holding: {params['min_holding_days']} days
            """
            
            await self.bot.send_message(
                message.chat.id,
                settings_text,
                parse_mode='Markdown'
            )
        
        @self.bot.callback_query_handler(func=lambda call: True)
        async def callback_handler(call):
            """Handle callback queries from inline keyboards"""
            if not self._is_authorized(call.from_user.id):
                await self.bot.answer_callback_query(call.id, "Unauthorized")
                return
            
            # Handle different callbacks
            if call.data == "start_trading":
                await self.bot.answer_callback_query(call.id)
                await start_trading_command(call.message)
                
            elif call.data == "stop_trading":
                await self.bot.answer_callback_query(call.id)
                await stop_trading_command(call.message)
                
            elif call.data == "confirm_start_trading":
                self.paper_trader.auto_trade_enabled = True
                await self.bot.answer_callback_query(call.id, "✅ Trading started")
                await self.bot.edit_message_text(
                    "✅ Automatic trading has been started",
                    call.message.chat.id,
                    call.message.message_id
                )
                # Send notification
                await self.send_notification("🚀 Automatic trading STARTED", "success")
                
            elif call.data == "confirm_stop_trading":
                self.paper_trader.auto_trade_enabled = False
                await self.bot.answer_callback_query(call.id, "✅ Trading stopped")
                await self.bot.edit_message_text(
                    "🛑 Automatic trading has been stopped",
                    call.message.chat.id,
                    call.message.message_id
                )
                # Send notification
                await self.send_notification("🛑 Automatic trading STOPPED", "warning")
                
            elif call.data == "cancel_action":
                await self.bot.answer_callback_query(call.id, "Cancelled")
                await self.bot.edit_message_text(
                    "❌ Action cancelled",
                    call.message.chat.id,
                    call.message.message_id
                )
                
            elif call.data == "show_positions":
                await self.bot.answer_callback_query(call.id)
                await self._send_positions_update(call.message.chat.id)
                
            elif call.data == "show_performance":
                await self.bot.answer_callback_query(call.id)
                await self._send_performance_update(call.message.chat.id)
            
            # Handle trade confirmations
            elif call.data.startswith("confirm_trade_"):
                trade_id = call.data.replace("confirm_trade_", "")
                if trade_id in self.pending_confirmations:
                    trade_info = self.pending_confirmations[trade_id]
                    
                    # Execute trade
                    success = await self.paper_trader.execute_trade(
                        trade_info['action'],
                        trade_info['symbol'],
                        trade_info['shares'],
                        trade_info['price'],
                        trade_info['reason']
                    )
                    
                    if success:
                        await self.bot.answer_callback_query(call.id, "✅ Trade executed")
                        await self.bot.edit_message_text(
                            f"✅ Trade executed: {trade_info['action']} {trade_info['shares']} {trade_info['symbol']} @ ${trade_info['price']:.2f}",
                            call.message.chat.id,
                            call.message.message_id
                        )
                    else:
                        await self.bot.answer_callback_query(call.id, "❌ Trade failed")
                        await self.bot.edit_message_text(
                            "❌ Trade execution failed",
                            call.message.chat.id,
                            call.message.message_id
                        )
                    
                    del self.pending_confirmations[trade_id]
                
            elif call.data.startswith("cancel_trade_"):
                trade_id = call.data.replace("cancel_trade_", "")
                if trade_id in self.pending_confirmations:
                    del self.pending_confirmations[trade_id]
                    await self.bot.answer_callback_query(call.id, "Trade cancelled")
                    await self.bot.edit_message_text(
                        "❌ Trade cancelled",
                        call.message.chat.id,
                        call.message.message_id
                    )
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        if not self.admin_users:
            return True  # If no admin users set, allow all
        return user_id in self.admin_users
    
    async def _send_positions_update(self, chat_id: int):
        """Send current positions update"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Paper trader not initialized")
            return
        
        status = self.paper_trader.get_portfolio_status()
        
        if not status['positions']:
            await self.bot.send_message(chat_id, "📊 No active positions")
            return
        
        positions_text = "💼 *Current Positions*\n━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for pos in status['positions']:
            emoji = "🟢" if pos['profit_pct'] > 0 else "🔴"
            positions_text += f"{emoji} *{pos['symbol']}*\n"
            positions_text += f"   Shares: {pos['shares']}\n"
            positions_text += f"   Entry: ${pos['entry_price']:.2f} → Current: ${pos['current_price']:.2f}\n"
            positions_text += f"   Value: ${pos['value']:,.2f}\n"
            positions_text += f"   P&L: ${pos['profit']:,.2f} ({pos['profit_pct']:+.2f}%)\n"
            positions_text += f"   Days: {pos['holding_days']}\n\n"
        
        await self.bot.send_message(
            chat_id,
            positions_text,
            parse_mode='Markdown'
        )
    
    async def _send_performance_update(self, chat_id: int):
        """Send performance metrics update"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Paper trader not initialized")
            return
        
        metrics = self.paper_trader.get_performance_metrics()
        
        if not metrics:
            await self.bot.send_message(chat_id, "📊 No performance data available")
            return
        
        perf_text = f"""
📈 *Performance Metrics*
━━━━━━━━━━━━━━━━━━━━

📊 *Returns:*
• Total Profit: ${metrics['total_profit']:,.2f}
• Avg Return: {metrics['avg_profit_pct']:.2f}%
• Best Trade: {metrics['max_win']:.2f}%
• Worst Trade: {metrics['max_loss']:.2f}%

🎯 *Win/Loss:*
• Total Trades: {metrics['total_trades']}
• Win Rate: {metrics['win_rate']:.1f}%
• Avg Win: {metrics['avg_win']:.2f}%
• Avg Loss: {metrics['avg_loss']:.2f}%

📉 *Risk Metrics:*
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Max Drawdown: {metrics['max_drawdown']:.2f}%
• Profit Factor: {metrics['profit_factor']:.2f}
        """
        
        await self.bot.send_message(
            chat_id,
            perf_text,
            parse_mode='Markdown'
        )
    
    async def send_trade_notification(self, action: str, symbol: str, shares: int, 
                                     price: float, reason: str, profit: Optional[float] = None):
        """Send trade execution notification"""
        if action == "BUY":
            emoji = "🟢"
            notification_text = f"""
{emoji} *BUY SIGNAL*
━━━━━━━━━━━━━━━━━━━━

📊 *Symbol:* {symbol}
📈 *Shares:* {shares}
💰 *Price:* ${price:.2f}
💵 *Total:* ${shares * price:,.2f}
📍 *Reason:* {reason}

⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        else:  # SELL
            emoji = "🔴" if profit and profit < 0 else "🟢"
            profit_text = f"${profit:,.2f} ({(profit/(shares*price-profit)*100):+.2f}%)" if profit else "N/A"
            
            notification_text = f"""
{emoji} *SELL SIGNAL*
━━━━━━━━━━━━━━━━━━━━

📊 *Symbol:* {symbol}
📈 *Shares:* {shares}
💰 *Price:* ${price:.2f}
💵 *Total:* ${shares * price:,.2f}
💸 *Profit:* {profit_text}
📍 *Reason:* {reason}

⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
        
        # Send notification
        await self.send_notification(notification_text, "trade")
    
    async def send_notification(self, message: str, notification_type: str = "info"):
        """Send notification to Telegram"""
        try:
            # Choose emoji based on type
            type_emojis = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "trade": "💼"
            }
            
            emoji = type_emojis.get(notification_type, "📢")
            
            # Format message
            formatted_message = f"{emoji} {message}"
            
            # Send to configured chat
            if self.chat_id:
                await self.bot.send_message(
                    self.chat_id,
                    formatted_message,
                    parse_mode='Markdown'
                )
            
            logger.info(f"Telegram notification sent: {notification_type}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    async def request_trade_confirmation(self, action: str, symbol: str, 
                                       shares: int, price: float, reason: str) -> str:
        """Request trade confirmation via Telegram"""
        import uuid
        trade_id = str(uuid.uuid4())[:8]
        
        # Store trade info
        self.pending_confirmations[trade_id] = {
            'action': action,
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        # Create confirmation message
        emoji = "🟢" if action == "BUY" else "🔴"
        
        confirmation_text = f"""
{emoji} *Trade Confirmation Required*
━━━━━━━━━━━━━━━━━━━━

📊 *Action:* {action}
📈 *Symbol:* {symbol}
💼 *Shares:* {shares}
💰 *Price:* ${price:.2f}
💵 *Total:* ${shares * price:,.2f}
📍 *Reason:* {reason}

⚠️ *Please confirm this trade:*
        """
        
        # Create inline keyboard
        keyboard = InlineKeyboardMarkup()
        keyboard.add(
            InlineKeyboardButton("✅ Execute", callback_data=f"confirm_trade_{trade_id}"),
            InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_trade_{trade_id}")
        )
        
        # Send confirmation request
        await self.bot.send_message(
            self.chat_id,
            confirmation_text,
            parse_mode='Markdown',
            reply_markup=keyboard
        )
        
        return trade_id
    
    async def send_daily_summary(self):
        """Send daily trading summary"""
        if not self.paper_trader:
            return
        
        status = self.paper_trader.get_portfolio_status()
        metrics = self.paper_trader.get_performance_metrics()
        trades_df = self.paper_trader.get_trade_history()
        
        # Get today's trades
        today = datetime.now().date()
        if not trades_df.empty:
            trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
            today_trades = trades_df[trades_df['exit_date'].dt.date == today]
            today_profit = today_trades['profit'].sum() if len(today_trades) > 0 else 0
            today_trades_count = len(today_trades)
        else:
            today_profit = 0
            today_trades_count = 0
        
        summary_text = f"""
📊 *Daily Trading Summary*
{datetime.now().strftime('%Y-%m-%d')}
━━━━━━━━━━━━━━━━━━━━

💰 *Portfolio Status:*
• Value: ${status['portfolio_value']:,.2f}
• Return: ${status['total_return']:,.2f} ({status['total_return_pct']:+.2f}%)
• Positions: {status['num_positions']}/{self.paper_trader.PORTFOLIO_PARAMS['max_positions']}

📈 *Today's Activity:*
• Trades: {today_trades_count}
• P&L: ${today_profit:,.2f}

🎯 *Overall Performance:*
• Total Trades: {status['total_trades']}
• Win Rate: {status['win_rate']:.1f}%
• Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
• Max DD: {metrics.get('max_drawdown', 0):.2f}%

Have a great trading day! 🚀
        """
        
        await self.send_notification(summary_text, "info")
    
    async def start(self):
        """Start the Telegram bot"""
        self.is_running = True
        logger.info("Starting Telegram bot...")
        
        # Send startup notification
        await self.send_notification(
            "*Telegram Bot Started*\n\nPaper trading bot is now online and ready! 🚀",
            "success"
        )
        
        # Start polling
        await self.bot.polling(non_stop=True)
    
    async def stop(self):
        """Stop the Telegram bot"""
        self.is_running = False
        
        # Send shutdown notification
        await self.send_notification(
            "*Telegram Bot Stopping*\n\nPaper trading bot is shutting down.",
            "warning"
        )
        
        await self.bot.close_session()
        logger.info("Telegram bot stopped")


# Configuration helper
def setup_telegram_config():
    """Interactive setup for Telegram configuration"""
    print("\n" + "="*50)
    print("TELEGRAM BOT CONFIGURATION")
    print("="*50)
    
    config_file = Path(__file__).parent / 'telegram_config.json'
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            existing_config = json.load(f)
        print("\nExisting configuration found.")
        use_existing = input("Use existing configuration? (y/n): ").lower()
        if use_existing == 'y':
            return existing_config
    
    print("\nPlease provide the following information:")
    print("(You can get these from @BotFather on Telegram)")
    
    config = {}
    
    # Bot token
    config['bot_token'] = input("\n1. Bot Token: ").strip()
    
    # Chat ID
    print("\n2. To get your Chat ID:")
    print("   - Send a message to your bot")
    print("   - Visit: https://api.telegram.org/bot{YOUR_BOT_TOKEN}/getUpdates")
    print("   - Look for 'chat':{'id':YOUR_CHAT_ID}")
    config['chat_id'] = input("\nChat ID: ").strip()
    
    # Admin users
    print("\n3. Admin User IDs (comma-separated, leave empty for all):")
    admin_input = input("Admin IDs: ").strip()
    if admin_input:
        config['admin_users'] = [int(uid.strip()) for uid in admin_input.split(',')]
    else:
        config['admin_users'] = []
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n✅ Configuration saved to:", config_file)
    print("\nIMPORTANT: Add these to your environment variables:")
    print(f"export TELEGRAM_BOT_TOKEN='{config['bot_token']}'")
    print(f"export TELEGRAM_CHAT_ID='{config['chat_id']}'")
    
    return config


if __name__ == "__main__":
    # Setup configuration
    config = setup_telegram_config()
    
    # Update global config
    TELEGRAM_CONFIG.update(config)
    
    print("\n" + "="*50)
    print("TESTING TELEGRAM CONNECTION")
    print("="*50)
    
    # Test bot
    async def test_bot():
        bot = TelegramBot()
        await bot.send_notification(
            "*Test Message*\n\nIf you see this, your Telegram bot is configured correctly! ✅",
            "success"
        )
        print("\n✅ Test message sent! Check your Telegram.")
    
    # Run test
    import asyncio
    asyncio.run(test_bot())