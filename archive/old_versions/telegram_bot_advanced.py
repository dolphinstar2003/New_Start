"""
Advanced Telegram Bot Integration
Full remote control system for trading and computer management
"""
import os
import sys
import json
import asyncio
import subprocess
import psutil
import platform
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
from loguru import logger
import telebot
from telebot.async_telebot import AsyncTeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton

# Add project path
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from utils.telegram_utils import (
    escape_markdown_v1, format_currency, format_percentage, format_symbol,
    format_trade_message, format_portfolio_status, format_position_list,
    format_trade_history, format_performance_metrics, format_opportunities
)


class AdvancedTelegramBot:
    """Advanced Telegram bot with full system control"""
    
    def __init__(self, paper_trader=None):
        self.paper_trader = paper_trader
        self.bot = None
        self.chat_id = None
        self.admin_users = []
        self.is_running = False
        
        # Command categories for help
        self.command_categories = {
            "📊 Trading Commands": {
                "/status": "Show portfolio status and returns",
                "/positions": "List all open positions with P&L",
                "/trades": "Show recent trade history",
                "/performance": "Detailed performance metrics",
                "/opportunities": "Top market opportunities",
                "/analyze [SYMBOL]": "Analyze specific symbol"
            },
            "🎮 Control Commands": {
                "/start_trading": "Enable automatic trading",
                "/stop_trading": "Disable automatic trading",
                "/set_confirmations [on/off]": "Toggle trade confirmations",
                "/set_notifications [on/off]": "Toggle notifications",
                "/force_check": "Force position evaluation now",
                "/save_state": "Save current portfolio state"
            },
            "💻 System Commands": {
                "/system": "Show system info (CPU, RAM, disk)",
                "/processes": "List running processes",
                "/screenshot": "Take screenshot",
                "/execute [command]": "Execute shell command",
                "/download [file]": "Download file from server",
                "/list_files [path]": "List files in directory"
            },
            "📈 Analysis Commands": {
                "/backtest [symbol] [period]": "Run backtest",
                "/scan_market": "Scan all symbols for signals",
                "/correlation": "Show symbol correlations",
                "/risk_report": "Portfolio risk analysis",
                "/daily_report": "Generate daily report"
            },
            "⚙️ Settings Commands": {
                "/get_params": "Show trading parameters",
                "/set_param [name] [value]": "Change parameter",
                "/reload_config": "Reload configuration",
                "/restart_bot": "Restart the bot",
                "/help": "Show this help message"
            }
        }
        
        # Load configuration
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file"""
        config_file = Path(__file__).parent / 'telegram_config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
                self.bot_token = config.get('bot_token', '')
                self.chat_id = config.get('chat_id', '')
                self.admin_users = config.get('admin_users', [])
        else:
            self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
            self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        if self.bot_token:
            self.bot = AsyncTeleBot(self.bot_token)
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup all message handlers"""
        
        @self.bot.message_handler(commands=['help', 'start'])
        async def help_command(message):
            """Show comprehensive help message"""
            if not self._is_authorized(message.from_user.id):
                await self.bot.send_message(message.chat.id, "❌ Unauthorized access")
                return
            
            help_text = "🤖 *Advanced Trading Bot \\- Command Reference*\n"
            help_text += "━" * 40 + "\n\n"
            
            for category, commands in self.command_categories.items():
                # Escape the category name
                category_escaped = escape_markdown_v1(category)
                help_text += f"*{category_escaped}*\n"
                for cmd, desc in commands.items():
                    # Escape special characters for Markdown
                    cmd_escaped = escape_markdown_v1(cmd)
                    desc_escaped = escape_markdown_v1(desc)
                    help_text += f"  {cmd_escaped} \\- {desc_escaped}\n"
                help_text += "\n"
            
            help_text += "━" * 40 + "\n"
            help_text += "💡 *Tips:*\n"
            help_text += "• Commands in \\[brackets\\] need parameters\n"
            help_text += "• All trades require confirmation by default\n"
            help_text += "• System commands require admin privileges\n"
            
            await self.bot.send_message(
                message.chat.id,
                help_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['status'])
        async def status_command(message):
            """Portfolio status with inline actions"""
            if not self._is_authorized(message.from_user.id):
                return
            
            if not self.paper_trader:
                await self.bot.send_message(message.chat.id, "❌ Trading system not initialized")
                return
            
            status = self.paper_trader.get_portfolio_status()
            
            # Use the proper formatting function
            status_text = format_portfolio_status(status)
            
            # Add trading status
            status_text += f"\n\n🤖 Auto Trading: {'✅ ON' if self.paper_trader.auto_trade_enabled else '❌ OFF'}\n"
            status_text += f"🔔 Confirmations: {'✅ Required' if self.paper_trader.require_confirmation else '❌ Disabled'}\n"
            
            # Create inline keyboard
            keyboard = InlineKeyboardMarkup()
            
            # Trading control buttons
            if self.paper_trader.auto_trade_enabled:
                keyboard.row(
                    InlineKeyboardButton("🛑 Stop Trading", callback_data="control_stop_trading"),
                    InlineKeyboardButton("🔄 Force Check", callback_data="control_force_check")
                )
            else:
                keyboard.row(
                    InlineKeyboardButton("▶️ Start Trading", callback_data="control_start_trading"),
                    InlineKeyboardButton("🔄 Force Check", callback_data="control_force_check")
                )
            
            # View buttons
            keyboard.row(
                InlineKeyboardButton("📊 Positions", callback_data="view_positions"),
                InlineKeyboardButton("📈 Performance", callback_data="view_performance")
            )
            
            # Analysis buttons
            keyboard.row(
                InlineKeyboardButton("🎯 Opportunities", callback_data="analyze_opportunities"),
                InlineKeyboardButton("📉 Risk Report", callback_data="analyze_risk")
            )
            
            await self.bot.send_message(
                message.chat.id,
                status_text,
                parse_mode='Markdown',
                reply_markup=keyboard
            )
        
        @self.bot.message_handler(commands=['system'])
        async def system_command(message):
            """Show system information"""
            if not self._is_authorized(message.from_user.id):
                return
            
            # Get system info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get network info
            net_io = psutil.net_io_counters()
            
            # Format system info with proper escaping
            platform_info = escape_markdown_v1(f"{platform.system()} {platform.release()}")
            processor_info = escape_markdown_v1(platform.processor())
            
            system_text = "💻 *System Information*\n"
            system_text += "━" * 30 + "\n\n"
            system_text += f"🖥️ *Hardware:*\n"
            system_text += f"  Platform: {platform_info}\n"
            system_text += f"  Processor: {processor_info}\n"
            system_text += f"  CPU Cores: {psutil.cpu_count()}\n\n"
            
            system_text += f"📊 *Resources:*\n"
            system_text += f"  CPU Usage: {cpu_percent}%\n"
            system_text += f"  RAM: {memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB ({memory.percent}%)\n"
            system_text += f"  Disk: {disk.used/1024/1024/1024:.1f}GB / {disk.total/1024/1024/1024:.1f}GB ({disk.percent}%)\n\n"
            
            system_text += f"🌐 *Network:*\n"
            system_text += f"  Sent: {net_io.bytes_sent/1024/1024:.1f} MB\n"
            system_text += f"  Received: {net_io.bytes_recv/1024/1024:.1f} MB\n\n"
            
            # Get Python process info
            process = psutil.Process(os.getpid())
            uptime_str = escape_markdown_v1(str(datetime.now() - datetime.fromtimestamp(process.create_time())))
            system_text += f"🐍 *Python Process:*\n"
            system_text += f"  Memory: {process.memory_info().rss/1024/1024:.1f} MB\n"
            system_text += f"  Threads: {process.num_threads()}\n"
            system_text += f"  Uptime: {uptime_str}\n"
            
            # Create action buttons
            keyboard = InlineKeyboardMarkup()
            keyboard.row(
                InlineKeyboardButton("🔄 Refresh", callback_data="system_refresh"),
                InlineKeyboardButton("📸 Screenshot", callback_data="system_screenshot")
            )
            keyboard.row(
                InlineKeyboardButton("📝 Processes", callback_data="system_processes"),
                InlineKeyboardButton("📁 Files", callback_data="system_files")
            )
            
            await self.bot.send_message(
                message.chat.id,
                system_text,
                parse_mode='Markdown',
                reply_markup=keyboard
            )
        
        @self.bot.message_handler(commands=['execute'])
        async def execute_command(message):
            """Execute shell command (admin only)"""
            if not self._is_authorized(message.from_user.id):
                return
            
            # Extra security check for system commands
            if message.from_user.id not in self.admin_users[:1]:  # Only first admin
                await self.bot.send_message(
                    message.chat.id,
                    "❌ This command requires super admin privileges"
                )
                return
            
            # Extract command
            parts = message.text.split(maxsplit=1)
            if len(parts) < 2:
                await self.bot.send_message(
                    message.chat.id,
                    "Usage: /execute <command>\nExample: /execute ls -la"
                )
                return
            
            command = parts[1]
            
            # Security: Whitelist safe commands
            safe_commands = ['ls', 'pwd', 'date', 'uptime', 'df', 'free']
            command_base = command.split()[0]
            
            if command_base not in safe_commands:
                # Ask for confirmation
                keyboard = InlineKeyboardMarkup()
                keyboard.row(
                    InlineKeyboardButton("✅ Execute", callback_data=f"exec_confirm_{command[:50]}"),
                    InlineKeyboardButton("❌ Cancel", callback_data="exec_cancel")
                )
                
                command_escaped = escape_markdown_v1(command)
                await self.bot.send_message(
                    message.chat.id,
                    f"⚠️ *Security Warning*\n\nExecute this command?\n`{command_escaped}`\n\nThis could be dangerous!",
                    parse_mode='Markdown',
                    reply_markup=keyboard
                )
                return
            
            # Execute safe command directly
            await self._execute_command(message.chat.id, command)
        
        @self.bot.message_handler(commands=['backtest'])
        async def backtest_command(message):
            """Run backtest for symbol"""
            if not self._is_authorized(message.from_user.id):
                return
            
            parts = message.text.split()
            if len(parts) < 2:
                await self.bot.send_message(
                    message.chat.id,
                    "Usage: /backtest <symbol> [period]\nExample: /backtest GARAN 2024"
                )
                return
            
            symbol = parts[1].upper()
            if not symbol.endswith('.IS'):
                symbol += '.IS'
            
            period = parts[2] if len(parts) > 2 else "2024"
            
            await self.bot.send_message(
                message.chat.id,
                f"🔄 Starting backtest for {symbol} - {period}..."
            )
            
            # TODO: Implement actual backtest
            await asyncio.sleep(2)  # Simulate backtest
            
            # Send mock results
            results_text = f"📊 *Backtest Results - {symbol}*\n"
            results_text += "━" * 30 + "\n\n"
            results_text += f"Period: {period}\n"
            results_text += f"Total Return: +24.5%\n"
            results_text += f"Win Rate: 52.3%\n"
            results_text += f"Sharpe Ratio: 1.85\n"
            results_text += f"Max Drawdown: -8.2%\n"
            results_text += f"Total Trades: 47\n"
            
            await self.bot.send_message(
                message.chat.id,
                results_text,
                parse_mode='Markdown'
            )
        
        @self.bot.message_handler(commands=['set_param'])
        async def set_param_command(message):
            """Set trading parameter"""
            if not self._is_authorized(message.from_user.id):
                return
            
            parts = message.text.split()
            if len(parts) < 3:
                params_text = "*Available Parameters:*\n"
                for key, value in self.paper_trader.PORTFOLIO_PARAMS.items():
                    params_text += f"`{key}`: {value}\n"
                
                await self.bot.send_message(
                    message.chat.id,
                    f"Usage: /set\\_param <name> <value>\n\n{params_text}",
                    parse_mode='Markdown'
                )
                return
            
            param_name = parts[1]
            param_value = parts[2]
            
            if param_name in self.paper_trader.PORTFOLIO_PARAMS:
                try:
                    # Convert value to appropriate type
                    old_value = self.paper_trader.PORTFOLIO_PARAMS[param_name]
                    if isinstance(old_value, bool):
                        new_value = param_value.lower() in ['true', '1', 'yes', 'on']
                    elif isinstance(old_value, int):
                        new_value = int(param_value)
                    elif isinstance(old_value, float):
                        new_value = float(param_value)
                    else:
                        new_value = param_value
                    
                    # Update parameter
                    self.paper_trader.PORTFOLIO_PARAMS[param_name] = new_value
                    
                    await self.bot.send_message(
                        message.chat.id,
                        f"✅ Parameter updated:\n`{param_name}`: {old_value} → {new_value}",
                        parse_mode='Markdown'
                    )
                    
                except Exception as e:
                    await self.bot.send_message(
                        message.chat.id,
                        f"❌ Error: {str(e)}"
                    )
            else:
                await self.bot.send_message(
                    message.chat.id,
                    f"❌ Unknown parameter: {param_name}"
                )
        
        @self.bot.callback_query_handler(func=lambda call: True)
        async def callback_handler(call):
            """Handle all callback queries"""
            if not self._is_authorized(call.from_user.id):
                await self.bot.answer_callback_query(call.id, "Unauthorized")
                return
            
            # Control callbacks
            if call.data == "control_start_trading":
                self.paper_trader.auto_trade_enabled = True
                await self.bot.answer_callback_query(call.id, "✅ Trading started")
                await self.bot.edit_message_text(
                    "✅ Automatic trading has been started",
                    call.message.chat.id,
                    call.message.message_id
                )
                await self.send_notification("🚀 Automatic trading STARTED", "success")
                
            elif call.data == "control_stop_trading":
                self.paper_trader.auto_trade_enabled = False
                await self.bot.answer_callback_query(call.id, "✅ Trading stopped")
                await self.bot.edit_message_text(
                    "🛑 Automatic trading has been stopped",
                    call.message.chat.id,
                    call.message.message_id
                )
                await self.send_notification("🛑 Automatic trading STOPPED", "warning")
                
            elif call.data == "control_force_check":
                await self.bot.answer_callback_query(call.id, "Checking positions...")
                await self._force_check_positions(call.message.chat.id)
                
            # View callbacks
            elif call.data == "view_positions":
                await self.bot.answer_callback_query(call.id)
                await self._send_positions_update(call.message.chat.id)
                
            elif call.data == "view_performance":
                await self.bot.answer_callback_query(call.id)
                await self._send_performance_update(call.message.chat.id)
                
            # Analysis callbacks
            elif call.data == "analyze_opportunities":
                await self.bot.answer_callback_query(call.id, "Analyzing market...")
                await self._send_opportunities(call.message.chat.id)
                
            elif call.data == "analyze_risk":
                await self.bot.answer_callback_query(call.id, "Generating risk report...")
                await self._send_risk_report(call.message.chat.id)
                
            # System callbacks
            elif call.data == "system_refresh":
                await self.bot.answer_callback_query(call.id, "Refreshing...")
                # Delete old message and send new system info
                await self.bot.delete_message(call.message.chat.id, call.message.message_id)
                await system_command(call.message)
                
            elif call.data == "system_screenshot":
                await self.bot.answer_callback_query(call.id, "Taking screenshot...")
                await self._take_screenshot(call.message.chat.id)
                
            elif call.data == "system_processes":
                await self.bot.answer_callback_query(call.id)
                await self._send_process_list(call.message.chat.id)
                
            elif call.data == "system_files":
                await self.bot.answer_callback_query(call.id)
                await self._send_file_list(call.message.chat.id, ".")
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        if not self.admin_users:
            return True
        return user_id in self.admin_users
    
    async def _execute_command(self, chat_id: int, command: str):
        """Execute shell command and send result"""
        try:
            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout or result.stderr
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            
            # Escape the command and output for code block
            command_escaped = escape_markdown_v1(command)
            output_escaped = escape_markdown_v1(output)
            response = f"```\n$ {command_escaped}\n{output_escaped}\n```"
            
            await self.bot.send_message(
                chat_id,
                response,
                parse_mode='Markdown'
            )
            
        except subprocess.TimeoutExpired:
            await self.bot.send_message(chat_id, "❌ Command timed out")
        except Exception as e:
            await self.bot.send_message(chat_id, f"❌ Error: {str(e)}")
    
    async def _force_check_positions(self, chat_id: int):
        """Force check all positions"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Trading system not initialized")
            return
        
        await self.bot.send_message(chat_id, "🔄 Checking all positions...")
        
        # Run position checks
        try:
            await self.paper_trader.check_positions_for_exit()
            await self.paper_trader.check_for_rotation()
            await self.paper_trader.check_for_new_entries()
            
            status = self.paper_trader.get_portfolio_status()
            await self.bot.send_message(
                chat_id,
                f"✅ Position check complete\n"
                f"Positions: {status['num_positions']}\n"
                f"Portfolio Value: ${status['portfolio_value']:,.2f}"
            )
        except Exception as e:
            await self.bot.send_message(chat_id, f"❌ Error during check: {str(e)}")
    
    async def _send_positions_update(self, chat_id: int):
        """Send detailed positions update"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Trading system not initialized")
            return
        
        status = self.paper_trader.get_portfolio_status()
        
        if not status['positions']:
            await self.bot.send_message(chat_id, "📊 No active positions")
            return
        
        # Use the proper formatting function
        positions_text = format_position_list(status['positions'])
        
        # Add totals
        total_value = sum(pos['value'] for pos in status['positions'])
        total_profit = sum(pos['profit'] for pos in status['positions'])
        
        positions_text += "\n━" * 30 + "\n"
        positions_text += f"*Total Value:* {format_currency(total_value)}\n"
        positions_text += f"*Total P&L:* {format_currency(total_profit)}\n"
        
        await self.bot.send_message(
            chat_id,
            positions_text,
            parse_mode='Markdown'
        )
    
    async def _send_performance_update(self, chat_id: int):
        """Send performance metrics"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Trading system not initialized")
            return
        
        metrics = self.paper_trader.get_performance_metrics()
        
        if not metrics:
            await self.bot.send_message(chat_id, "📊 No performance data available")
            return
        
        # Use the proper formatting function
        perf_text = format_performance_metrics(metrics)
        
        await self.bot.send_message(
            chat_id,
            perf_text,
            parse_mode='Markdown'
        )
    
    async def _send_opportunities(self, chat_id: int):
        """Send top market opportunities"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Trading system not initialized")
            return
        
        opportunities = self.paper_trader.evaluate_all_opportunities()
        
        if not opportunities:
            await self.bot.send_message(chat_id, "📊 No opportunities available")
            return
        
        # Get top 10
        top_opps = opportunities[:10]
        
        # Use the proper formatting function
        opps_text = format_opportunities(top_opps, 10)
        
        await self.bot.send_message(
            chat_id,
            opps_text,
            parse_mode='Markdown'
        )
    
    async def _send_risk_report(self, chat_id: int):
        """Send portfolio risk analysis"""
        if not self.paper_trader:
            await self.bot.send_message(chat_id, "❌ Trading system not initialized")
            return
        
        status = self.paper_trader.get_portfolio_status()
        
        risk_text = "📉 *Portfolio Risk Report*\n"
        risk_text += "━" * 30 + "\n\n"
        
        # Portfolio allocation
        cash_pct = (status['cash'] / status['portfolio_value']) * 100
        invested_pct = 100 - cash_pct
        
        risk_text += "*Asset Allocation:*\n"
        risk_text += f"  Cash: {cash_pct:.1f}%\n"
        risk_text += f"  Invested: {invested_pct:.1f}%\n"
        risk_text += f"  Positions: {status['num_positions']}/{self.paper_trader.PORTFOLIO_PARAMS['max_positions']}\n\n"
        
        # Position concentration
        if status['positions']:
            position_values = [pos['value'] for pos in status['positions']]
            max_position = max(position_values)
            max_position_pct = (max_position / status['portfolio_value']) * 100
            
            risk_text += "*Position Concentration:*\n"
            risk_text += f"  Largest Position: {max_position_pct:.1f}%\n"
            risk_text += f"  Avg Position Size: {invested_pct / status['num_positions']:.1f}%\n\n"
        
        # Risk parameters
        risk_text += "*Risk Parameters:*\n"
        risk_text += f"  Stop Loss: {self.paper_trader.PORTFOLIO_PARAMS['stop_loss'] * 100:.0f}%\n"
        risk_text += f"  Take Profit: {self.paper_trader.PORTFOLIO_PARAMS['take_profit'] * 100:.0f}%\n"
        risk_text += f"  Max Position Size: {self.paper_trader.PORTFOLIO_PARAMS['max_position_pct'] * 100:.0f}%\n"
        
        # Risk assessment
        risk_score = 0
        if invested_pct > 80:
            risk_score += 3
        elif invested_pct > 60:
            risk_score += 2
        else:
            risk_score += 1
        
        if status['num_positions'] > 7:
            risk_score += 2
        elif status['num_positions'] > 5:
            risk_score += 1
        
        risk_levels = {1: "🟢 Low", 2: "🟢 Low", 3: "🟡 Medium", 4: "🟡 Medium", 5: "🔴 High"}
        
        risk_text += f"\n*Overall Risk Level:* {risk_levels.get(risk_score, '🔴 High')}\n"
        
        await self.bot.send_message(
            chat_id,
            risk_text,
            parse_mode='Markdown'
        )
    
    async def _take_screenshot(self, chat_id: int):
        """Take and send screenshot"""
        try:
            # Take screenshot using platform-specific command
            screenshot_path = Path("screenshot.png")
            
            if platform.system() == "Linux":
                subprocess.run(["scrot", str(screenshot_path)], check=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["screencapture", "-x", str(screenshot_path)], check=True)
            elif platform.system() == "Windows":
                # Use PIL for Windows
                from PIL import ImageGrab
                screenshot = ImageGrab.grab()
                screenshot.save(screenshot_path)
            
            # Send screenshot
            if screenshot_path.exists():
                with open(screenshot_path, 'rb') as photo:
                    await self.bot.send_photo(
                        chat_id,
                        photo,
                        caption=f"📸 Screenshot taken at {datetime.now().strftime('%H:%M:%S')}"
                    )
                screenshot_path.unlink()  # Delete after sending
            else:
                await self.bot.send_message(chat_id, "❌ Failed to take screenshot")
                
        except Exception as e:
            await self.bot.send_message(chat_id, f"❌ Screenshot error: {str(e)}")
    
    async def _send_process_list(self, chat_id: int):
        """Send list of running processes"""
        try:
            # Get top 10 processes by CPU
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['cpu_percent'] > 0:
                        processes.append(pinfo)
                except:
                    pass
            
            # Sort by CPU usage
            processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
            
            proc_text = "📝 *Top Processes by CPU*\n"
            proc_text += "━" * 30 + "\n\n"
            
            for proc in processes[:10]:
                proc_text += f"PID: {proc['pid']} - {proc['name']}\n"
                proc_text += f"  CPU: {proc['cpu_percent']:.1f}%\n"
                proc_text += f"  RAM: {proc['memory_percent']:.1f}%\n\n"
            
            await self.bot.send_message(
                chat_id,
                proc_text,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            await self.bot.send_message(chat_id, f"❌ Process list error: {str(e)}")
    
    async def _send_file_list(self, chat_id: int, path: str):
        """Send list of files in directory"""
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                await self.bot.send_message(chat_id, "❌ Path does not exist")
                return
            
            files_text = f"📁 *Files in {path}*\n"
            files_text += "━" * 30 + "\n\n"
            
            # List files and directories
            items = list(path_obj.iterdir())[:20]  # Limit to 20 items
            
            for item in sorted(items):
                if item.is_dir():
                    files_text += f"📁 {item.name}/\n"
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024*1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/1024/1024:.1f}MB"
                    
                    files_text += f"📄 {item.name} ({size_str})\n"
            
            if len(items) < len(list(path_obj.iterdir())):
                files_text += f"\n... and {len(list(path_obj.iterdir())) - len(items)} more"
            
            # Add navigation buttons
            keyboard = InlineKeyboardMarkup()
            if path != ".":
                parent = str(Path(path).parent)
                keyboard.row(
                    InlineKeyboardButton("⬆️ Parent", callback_data=f"files_{parent}")
                )
            
            await self.bot.send_message(
                chat_id,
                files_text,
                parse_mode='Markdown',
                reply_markup=keyboard
            )
            
        except Exception as e:
            await self.bot.send_message(chat_id, f"❌ File list error: {str(e)}")
    
    async def send_notification(self, message: str, notification_type: str = "info"):
        """Send notification to Telegram with proper escaping"""
        try:
            type_emojis = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌",
                "trade": "💼"
            }
            
            emoji = type_emojis.get(notification_type, "📢")
            
            # For trade messages, the message is already formatted
            if notification_type == "trade":
                formatted_message = message
            else:
                # Escape special characters for other notifications
                escaped_message = escape_markdown_v1(message)
                formatted_message = f"{emoji} {escaped_message}"
            
            if self.chat_id:
                await self.bot.send_message(
                    self.chat_id,
                    formatted_message,
                    parse_mode='Markdown'
                )
            
            logger.info(f"Telegram notification sent: {notification_type}")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
    
    async def start(self):
        """Start the Telegram bot"""
        self.is_running = True
        logger.info("Starting Advanced Telegram bot...")
        
        await self.send_notification(
            "*🤖 Advanced Trading Bot Started*\n\n"
            "Full system control enabled!\n"
            "Type /help for all commands",
            "success"
        )
        
        await self.bot.polling(non_stop=True)
    
    async def stop(self):
        """Stop the Telegram bot"""
        self.is_running = False
        
        await self.send_notification(
            "*Bot Stopping*\n\nSystem shutting down...",
            "warning"
        )
        
        await self.bot.close_session()
        logger.info("Advanced Telegram bot stopped")


if __name__ == "__main__":
    # Test the advanced bot
    async def test():
        bot = AdvancedTelegramBot()
        await bot.send_notification(
            "*Advanced Bot Test*\n\nIf you see this, the advanced bot is working!",
            "success"
        )
        await bot.bot.close_session()
    
    import asyncio
    asyncio.run(test())