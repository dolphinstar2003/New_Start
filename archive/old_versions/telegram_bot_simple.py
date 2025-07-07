#!/usr/bin/env python3
"""
Simple Telegram Bot for Paper Trading Control
Uses synchronous telebot for better compatibility
"""
import os
import json
import threading
from pathlib import Path
from datetime import datetime
import telebot
from loguru import logger

# Configuration
config_file = Path(__file__).parent / 'telegram_config.json'
config = {}
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)

BOT_TOKEN = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
CHAT_ID = config.get('chat_id', os.getenv('TELEGRAM_CHAT_ID', ''))


class SimpleTelegramBot:
    """Simple synchronous Telegram bot"""
    
    def __init__(self, paper_trader=None):
        self.paper_trader = paper_trader
        self.bot = telebot.TeleBot(BOT_TOKEN)
        self.chat_id = CHAT_ID
        self.is_running = False
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup command handlers"""
        
        @self.bot.message_handler(commands=['help'])
        def handle_help(message):
            help_text = """
📊 *Trading Commands*
/status - Show portfolio status
/positions - List open positions
/trades - Show recent trades
/performance - Performance metrics

🎮 *Control Commands*
/start\\_trading - Enable auto trading
/stop\\_trading - Disable auto trading
/force\\_check - Check positions now

⚙️ *Settings*
/get\\_params - Show parameters
/help - Show this message
"""
            self.bot.reply_to(message, help_text, parse_mode='Markdown')
            logger.info(f"Help command from {message.from_user.username}")
        
        @self.bot.message_handler(commands=['status'])
        def handle_status(message):
            if self.paper_trader:
                status = self.paper_trader.get_portfolio_status()
                text = f"""
💼 *Portfolio Status*

Value: ${status['portfolio_value']:,.2f}
Cash: ${status['cash']:,.2f}
Positions: {status['num_positions']}/10
Return: {status['total_return_pct']:+.2f}%
Trades: {status['total_trades']}
"""
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
            logger.info(f"Status command from {message.from_user.username}")
        
        @self.bot.message_handler(commands=['positions'])
        def handle_positions(message):
            if self.paper_trader and self.paper_trader.portfolio['positions']:
                text = "*📈 Open Positions*\n\n"
                for symbol, pos in self.paper_trader.portfolio['positions'].items():
                    pnl = pos['unrealized_pnl']
                    pnl_pct = pos['unrealized_pnl_pct']
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    text += f"{emoji} *{symbol}*\n"
                    text += f"  Shares: {pos['shares']}\n"
                    text += f"  Avg Price: ${pos['average_price']:.2f}\n"
                    text += f"  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n\n"
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "📭 No open positions")
            logger.info(f"Positions command from {message.from_user.username}")
        
        @self.bot.message_handler(commands=['start_trading'])
        def handle_start_trading(message):
            if self.paper_trader:
                self.paper_trader.auto_trade_enabled = True
                self.bot.reply_to(message, "✅ Auto trading ENABLED")
                logger.info(f"Trading enabled by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['stop_trading'])
        def handle_stop_trading(message):
            if self.paper_trader:
                self.paper_trader.auto_trade_enabled = False
                self.bot.reply_to(message, "🛑 Auto trading DISABLED")
                logger.info(f"Trading disabled by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['force_check'])
        def handle_force_check(message):
            if self.paper_trader:
                self.bot.reply_to(message, "🔄 Checking positions...")
                # Note: This would need to be handled asynchronously in the main loop
                self.paper_trader._force_check_flag = True
                logger.info(f"Force check by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['get_params'])
        def handle_get_params(message):
            if self.paper_trader:
                text = "*⚙️ Trading Parameters*\n\n"
                text += f"Max Positions: {self.paper_trader.max_positions}\n"
                text += f"Position Size: {self.paper_trader.position_size_pct*100:.0f}%\n"
                text += f"Stop Loss: {self.paper_trader.stop_loss_pct*100:.0f}%\n"
                text += f"Take Profit: {self.paper_trader.take_profit_pct*100:.0f}%\n"
                text += f"Min Score: {self.paper_trader.min_score}\n"
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
            logger.info(f"Get params command from {message.from_user.username}")
        
        @self.bot.message_handler(commands=['performance'])
        def handle_performance(message):
            if self.paper_trader:
                metrics = self.paper_trader.get_performance_metrics()
                if metrics:
                    text = "*📊 Performance Metrics*\n\n"
                    text += f"Total Return: {metrics['total_return']:.2f}%\n"
                    text += f"Win Rate: {metrics['win_rate']:.1f}%\n"
                    text += f"Avg Win: ${metrics['avg_win']:.2f}\n"
                    text += f"Avg Loss: ${metrics['avg_loss']:.2f}\n"
                    text += f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                    text += f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                    text += f"Max Drawdown: {metrics['max_drawdown']:.2f}%\n"
                    self.bot.reply_to(message, text, parse_mode='Markdown')
                else:
                    self.bot.reply_to(message, "📊 Not enough data for metrics")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
            logger.info(f"Performance command from {message.from_user.username}")
        
        @self.bot.message_handler(commands=['trades'])
        def handle_trades(message):
            if self.paper_trader and self.paper_trader.trade_history:
                text = "*📜 Recent Trades*\n\n"
                recent_trades = self.paper_trader.trade_history[-5:]  # Last 5 trades
                for trade in reversed(recent_trades):
                    emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                    text += f"{emoji} {trade['action']} {trade['symbol']}\n"
                    text += f"  {trade['shares']} @ ${trade['price']:.2f}\n"
                    text += f"  {trade['timestamp'].strftime('%m/%d %H:%M')}\n\n"
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "📭 No trades yet")
            logger.info(f"Trades command from {message.from_user.username}")
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_unknown(message):
            self.bot.reply_to(message, "❓ Unknown command. Use /help for available commands.")
            logger.info(f"Unknown command '{message.text}' from {message.from_user.username}")
    
    def send_notification(self, message, notification_type="info"):
        """Send notification to Telegram"""
        try:
            # Escape markdown special characters
            message = message.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[').replace(']', '\\]')
            
            emojis = {
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "info": "ℹ️",
                "trade": "💰"
            }
            emoji = emojis.get(notification_type, "📢")
            
            self.bot.send_message(self.chat_id, f"{emoji} {message}", parse_mode='Markdown')
            logger.info(f"Notification sent: {notification_type}")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def start(self):
        """Start the bot in a separate thread"""
        self.is_running = True
        bot_thread = threading.Thread(target=self._polling_loop)
        bot_thread.daemon = True
        bot_thread.start()
        logger.info("Simple Telegram bot started")
        
        # Send start notification
        self.send_notification("🤖 *Trading Bot Started*\n\nUse /help for commands", "success")
    
    def _polling_loop(self):
        """Polling loop for the bot"""
        while self.is_running:
            try:
                self.bot.polling(non_stop=True, interval=1, timeout=30)
            except Exception as e:
                logger.error(f"Polling error: {e}")
                if self.is_running:
                    import time
                    time.sleep(5)
    
    def stop(self):
        """Stop the bot"""
        self.is_running = False
        self.bot.stop_polling()
        logger.info("Simple Telegram bot stopped")


if __name__ == "__main__":
    # Test the bot
    bot = SimpleTelegramBot()
    bot.start()
    
    print("Bot is running. Send /help in Telegram")
    print("Press Ctrl+C to stop")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.stop()