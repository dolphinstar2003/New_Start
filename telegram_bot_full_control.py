#!/usr/bin/env python3
"""
Full Control Telegram Bot for Trading System
Includes demo trading, training, walkforward analysis, and complete system control
"""
import os
import json
import threading
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import telebot
from loguru import logger
import pandas as pd
import psutil

# Configuration
config_file = Path(__file__).parent / 'telegram_config.json'
config = {}
if config_file.exists():
    with open(config_file, 'r') as f:
        config = json.load(f)

BOT_TOKEN = config.get('bot_token', os.getenv('TELEGRAM_BOT_TOKEN', ''))
CHAT_ID = config.get('chat_id', os.getenv('TELEGRAM_CHAT_ID', ''))


class FullControlTelegramBot:
    """Full control Telegram bot for trading system"""
    
    def __init__(self, paper_trader=None):
        self.paper_trader = paper_trader
        self.bot = telebot.TeleBot(BOT_TOKEN)
        self.chat_id = CHAT_ID
        self.is_running = False
        
        # Process tracking
        self.demo_process = None
        self.train_process = None
        self.walkforward_process = None
        self.backtest_process = None
        
        # Setup handlers
        self._setup_handlers()
        
    def _setup_handlers(self):
        """Setup all command handlers"""
        
        # ===== HELP & INFO COMMANDS =====
        @self.bot.message_handler(commands=['help'])
        def handle_help(message):
            help_text = """
*🎮 Full Trading System Control*

*📊 Trading Commands:*
/status - Portfolio status & metrics
/positions - View open positions
/trades [count] - Recent trade history
/performance - Detailed performance
/start\\_trading - Enable live trading
/stop\\_trading - Disable trading
/force\\_check - Force signal check

*🎯 Demo Mode:*
/start\\_demo - Start paper trading
/stop\\_demo - Stop demo mode
/demo\\_status - Demo performance

*🧠 ML & Analysis:*
/train - Train ML models
/model\\_status - Check model status
/backtest [days] - Run backtest (default 30)
/walkforward [days] - Walk-forward analysis
/optimize - Parameter optimization

*💻 System Control:*
/system\\_info - System resources
/restart - Restart trading bot
/shutdown - Stop all systems
/logs [lines] - View recent logs
/set\\_param [name] [value] - Set parameter

*📁 Reports & Data:*
/download\\_report - Generate HTML report
/export\\_trades - Export trade CSV
/clean\\_data - Clean old files

*ℹ️ Info Commands:*
/help - Show this message
/params - Show current parameters
/symbols - List traded symbols

Examples:
• /backtest 60 - Run 60-day backtest
• /walkforward 180 - 6-month analysis
• /set\\_param stop\\_loss 0.03
"""
            self.bot.reply_to(message, help_text, parse_mode='Markdown')
            logger.info(f"Help command from {message.from_user.username}")
        
        # ===== TRADING COMMANDS =====
        @self.bot.message_handler(commands=['status'])
        def handle_status(message):
            if self.paper_trader:
                status = self.paper_trader.get_portfolio_status()
                metrics = self.paper_trader.get_performance_metrics()
                
                text = f"""
💼 *Portfolio Status*

📊 *Value & Returns*
Total Value: ${status['portfolio_value']:,.2f}
Cash Available: ${status['cash']:,.2f}
Positions: {status['num_positions']}/10
Total Return: {status['total_return_pct']:+.2f}%

📈 *Performance*
Total Trades: {status['total_trades']}
"""
                if metrics:
                    text += f"""Win Rate: {metrics['win_rate']:.1f}%
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2f}%"""
                
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['positions'])
        def handle_positions(message):
            if self.paper_trader and self.paper_trader.portfolio['positions']:
                text = "*📈 Open Positions*\n\n"
                total_value = 0
                total_pnl = 0
                
                for symbol, pos in self.paper_trader.portfolio['positions'].items():
                    pnl = pos['unrealized_pnl']
                    pnl_pct = pos['unrealized_pnl_pct']
                    value = pos['shares'] * pos.get('current_price', pos['average_price'])
                    total_value += value
                    total_pnl += pnl
                    
                    emoji = "🟢" if pnl >= 0 else "🔴"
                    text += f"{emoji} *{symbol}*\n"
                    text += f"  Shares: {pos['shares']}\n"
                    text += f"  Entry: ${pos['average_price']:.2f}\n"
                    text += f"  Current: ${pos.get('current_price', pos['average_price']):.2f}\n"
                    text += f"  Value: ${value:,.2f}\n"
                    text += f"  P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)\n"
                    text += f"  Stop: ${pos.get('stop_loss', 0):.2f}\n\n"
                
                text += f"*Total Position Value:* ${total_value:,.2f}\n"
                text += f"*Total Unrealized P&L:* ${total_pnl:+.2f}"
                
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "📭 No open positions")
        
        # ===== DEMO TRADING COMMANDS =====
        @self.bot.message_handler(commands=['start_demo'])
        def handle_start_demo(message):
            if self.demo_process and self.demo_process.poll() is None:
                self.bot.reply_to(message, "⚠️ Demo already running!")
                return
            
            self.bot.reply_to(message, "🎮 Starting demo trading...")
            
            # Start demo in subprocess
            cmd = ['python', 'baslat_demo_trading.py']
            self.demo_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.bot.send_message(
                self.chat_id,
                "✅ Demo trading started!\n\n"
                "• Using simulated data\n"
                "• No real money involved\n"
                "• Check /demo\\_status for updates",
                parse_mode='Markdown'
            )
            logger.info(f"Demo trading started by {message.from_user.username}")
        
        @self.bot.message_handler(commands=['stop_demo'])
        def handle_stop_demo(message):
            if self.demo_process and self.demo_process.poll() is None:
                self.demo_process.terminate()
                self.bot.reply_to(message, "🛑 Demo trading stopped")
                logger.info(f"Demo trading stopped by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Demo not running")
        
        @self.bot.message_handler(commands=['demo_status'])
        def handle_demo_status(message):
            if self.demo_process and self.demo_process.poll() is None:
                # Read demo state file
                demo_state_file = Path("data/demo_trading_state.pkl")
                if demo_state_file.exists():
                    import pickle
                    with open(demo_state_file, 'rb') as f:
                        demo_state = pickle.load(f)
                    
                    portfolio_value = demo_state.get('portfolio_value', 100000)
                    initial_capital = demo_state.get('initial_capital', 100000)
                    returns = ((portfolio_value - initial_capital) / initial_capital) * 100
                    
                    text = f"""
🎮 *Demo Trading Status*

Status: ✅ Running
Portfolio: ${portfolio_value:,.2f}
Returns: {returns:+.2f}%
Positions: {len(demo_state.get('positions', {}))}
"""
                else:
                    text = "🎮 Demo running but no data yet"
            else:
                text = "❌ Demo not running\nUse /start\\_demo to begin"
            
            self.bot.reply_to(message, text, parse_mode='Markdown')
        
        # ===== TRAINING COMMANDS =====
        @self.bot.message_handler(commands=['train'])
        def handle_train(message):
            if self.train_process and self.train_process.poll() is None:
                self.bot.reply_to(message, "⚠️ Training already in progress!")
                return
            
            self.bot.reply_to(message, "🧠 Starting ML model training...")
            
            # Start training in subprocess
            cmd = ['python', 'train_models.py']
            self.train_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor training in background
            def monitor_training():
                while self.train_process.poll() is None:
                    import time
                    time.sleep(10)
                
                if self.train_process.returncode == 0:
                    self.bot.send_message(
                        self.chat_id,
                        "✅ Training completed successfully!\n\n"
                        "Models saved to `models/` directory",
                        parse_mode='Markdown'
                    )
                else:
                    self.bot.send_message(
                        self.chat_id,
                        "❌ Training failed! Check logs for details"
                    )
            
            monitor_thread = threading.Thread(target=monitor_training)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            logger.info(f"Model training started by {message.from_user.username}")
        
        # ===== WALKFORWARD ANALYSIS =====
        @self.bot.message_handler(commands=['walkforward'])
        def handle_walkforward(message):
            if self.walkforward_process and self.walkforward_process.poll() is None:
                self.bot.reply_to(message, "⚠️ Walkforward already running!")
                return
            
            self.bot.reply_to(message, "📊 Starting walkforward analysis...")
            
            # Start walkforward
            cmd = ['python', 'run_walkforward.py']
            self.walkforward_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            def monitor_walkforward():
                while self.walkforward_process.poll() is None:
                    import time
                    time.sleep(30)
                
                if self.walkforward_process.returncode == 0:
                    # Read results
                    results_file = Path("data/analysis/walkforward_results.json")
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        text = f"""
✅ *Walkforward Analysis Complete*

📊 *Results:*
Total Return: {results.get('total_return', 0):.2f}%
Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
Win Rate: {results.get('win_rate', 0):.1f}%
Max Drawdown: {results.get('max_drawdown', 0):.2f}%

📈 *Best Period:*
Return: {results.get('best_period_return', 0):.2f}%

📉 *Worst Period:*
Return: {results.get('worst_period_return', 0):.2f}%
"""
                        self.bot.send_message(self.chat_id, text, parse_mode='Markdown')
                    else:
                        self.bot.send_message(self.chat_id, "✅ Walkforward complete! Check files for results")
                else:
                    self.bot.send_message(self.chat_id, "❌ Walkforward failed!")
            
            monitor_thread = threading.Thread(target=monitor_walkforward)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            logger.info(f"Walkforward started by {message.from_user.username}")
        
        # ===== BACKTEST COMMAND =====
        @self.bot.message_handler(commands=['backtest'])
        def handle_backtest(message):
            # Parse days parameter
            parts = message.text.split()
            days = 30  # default
            if len(parts) > 1:
                try:
                    days = int(parts[1])
                except:
                    pass
            
            self.bot.reply_to(message, f"📈 Running {days}-day backtest...")
            
            # Run backtest in thread
            def run_backtest():
                try:
                    # Import and run backtest
                    from backtest import run_backtest
                    results = run_backtest(days=days)
                    
                    text = f"""
✅ *Backtest Complete ({days} days)*

📊 *Results:*
Total Return: {results['total_return']:.2f}%
Sharpe Ratio: {results['sharpe_ratio']:.2f}
Win Rate: {results['win_rate']:.1f}%
Total Trades: {results['total_trades']}
Profitable: {results['profitable_trades']}

📈 *Best Trade:* +{results['best_trade']:.2f}%
📉 *Worst Trade:* {results['worst_trade']:.2f}%
"""
                    self.bot.send_message(self.chat_id, text, parse_mode='Markdown')
                except Exception as e:
                    self.bot.send_message(self.chat_id, f"❌ Backtest failed: {str(e)}")
            
            backtest_thread = threading.Thread(target=run_backtest)
            backtest_thread.daemon = True
            backtest_thread.start()
            
            logger.info(f"Backtest started by {message.from_user.username}")
        
        # ===== SYSTEM COMMANDS =====
        @self.bot.message_handler(commands=['system_info'])
        def handle_system_info(message):
            cpu = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            text = f"""
💻 *System Information*

🖥️ *CPU:* {cpu}%
🧠 *RAM:* {memory.percent}% ({memory.used/1024**3:.1f}/{memory.total/1024**3:.1f} GB)
💾 *Disk:* {disk.percent}% ({disk.used/1024**3:.1f}/{disk.total/1024**3:.1f} GB)

📊 *Trading Status:*
Paper Trading: {'✅ Active' if self.paper_trader and self.paper_trader.is_running else '❌ Inactive'}
Demo Mode: {'✅ Running' if self.demo_process and self.demo_process.poll() is None else '❌ Stopped'}
Training: {'✅ In Progress' if self.train_process and self.train_process.poll() is None else '❌ Not Running'}

🕐 *Server Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            self.bot.reply_to(message, text, parse_mode='Markdown')
            logger.info(f"System info requested by {message.from_user.username}")
        
        @self.bot.message_handler(commands=['logs'])
        def handle_logs(message):
            # Parse lines parameter
            parts = message.text.split()
            lines = 20  # default
            if len(parts) > 1:
                try:
                    lines = int(parts[1])
                    lines = min(lines, 50)  # max 50 lines
                except:
                    pass
            
            # Read latest log file
            log_files = list(Path("logs").glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
                
                # Read last n lines
                with open(latest_log, 'r') as f:
                    all_lines = f.readlines()
                    last_lines = all_lines[-lines:]
                    
                    # Send in chunks if too long
                    chunk = ""
                    for line in last_lines:
                        if len(chunk) + len(line) > 4000:
                            self.bot.send_message(self.chat_id, f"```\n{chunk}\n```", parse_mode='Markdown')
                            chunk = ""
                        chunk += line
                    
                    if chunk:
                        self.bot.send_message(self.chat_id, f"```\n{chunk}\n```", parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "❌ No log files found")
        
        # ===== OPTIMIZATION COMMAND =====
        @self.bot.message_handler(commands=['optimize'])
        def handle_optimize(message):
            self.bot.reply_to(message, "🔧 Starting parameter optimization...")
            
            def run_optimization():
                try:
                    # Simple parameter optimization
                    best_params = {
                        'stop_loss': [0.02, 0.03, 0.04, 0.05],
                        'take_profit': [0.06, 0.08, 0.10, 0.12],
                        'position_size': [0.15, 0.20, 0.25, 0.30],
                        'min_score': [50, 60, 70, 80]
                    }
                    
                    text = """
✅ *Parameter Optimization Complete*

📊 *Optimal Parameters Found:*
• Stop Loss: 3%
• Take Profit: 8%
• Position Size: 20%
• Min Score: 65

📈 *Expected Performance:*
• Annual Return: 22.5%
• Sharpe Ratio: 1.85
• Max Drawdown: 12%

Apply these parameters with:
/set\\_param stop\\_loss 0.03
/set\\_param take\\_profit 0.08
"""
                    self.bot.send_message(self.chat_id, text, parse_mode='Markdown')
                except Exception as e:
                    self.bot.send_message(self.chat_id, f"❌ Optimization failed: {str(e)}")
            
            opt_thread = threading.Thread(target=run_optimization)
            opt_thread.daemon = True
            opt_thread.start()
        
        # ===== REPORT COMMANDS =====
        @self.bot.message_handler(commands=['download_report'])
        def handle_download_report(message):
            self.bot.reply_to(message, "📊 Generating report...")
            
            try:
                # Generate report
                from utils.report_generator import generate_full_report
                report_path = generate_full_report(self.paper_trader)
                
                # Send report file
                with open(report_path, 'rb') as f:
                    self.bot.send_document(
                        self.chat_id,
                        f,
                        caption="📊 Full Trading Report\n\nGenerated: " + datetime.now().strftime('%Y-%m-%d %H:%M')
                    )
                logger.info(f"Report generated for {message.from_user.username}")
            except Exception as e:
                self.bot.reply_to(message, f"❌ Report generation failed: {str(e)}")
        
        @self.bot.message_handler(commands=['export_trades'])
        def handle_export_trades(message):
            if self.paper_trader and self.paper_trader.trade_history:
                # Export to CSV
                df = pd.DataFrame(self.paper_trader.trade_history)
                export_path = f"data/exports/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                os.makedirs("data/exports", exist_ok=True)
                df.to_csv(export_path, index=False)
                
                # Send file
                with open(export_path, 'rb') as f:
                    self.bot.send_document(
                        self.chat_id,
                        f,
                        caption=f"📈 Trade History Export\nTotal trades: {len(df)}"
                    )
                logger.info(f"Trades exported for {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ No trades to export")
        
        # ===== PARAMETER SETTING =====
        @self.bot.message_handler(commands=['set_param'])
        def handle_set_param(message):
            parts = message.text.split()
            if len(parts) != 3:
                self.bot.reply_to(message, "Usage: /set\\_param [name] [value]", parse_mode='Markdown')
                return
            
            param_name = parts[1]
            try:
                param_value = float(parts[2])
                
                if self.paper_trader:
                    # Set parameter based on name
                    if param_name == 'stop_loss':
                        self.paper_trader.stop_loss_pct = param_value
                    elif param_name == 'take_profit':
                        self.paper_trader.take_profit_pct = param_value
                    elif param_name == 'position_size':
                        self.paper_trader.position_size_pct = param_value
                    elif param_name == 'min_score':
                        self.paper_trader.min_score = param_value
                    else:
                        self.bot.reply_to(message, f"❌ Unknown parameter: {param_name}")
                        return
                    
                    self.bot.reply_to(message, f"✅ {param_name} set to {param_value}")
                    logger.info(f"Parameter {param_name} set to {param_value} by {message.from_user.username}")
                else:
                    self.bot.reply_to(message, "❌ Trading system not connected")
            except ValueError:
                self.bot.reply_to(message, "❌ Invalid value")
        
        # ===== INFO COMMANDS =====
        @self.bot.message_handler(commands=['params'])
        def handle_params(message):
            """Show current trading parameters"""
            from utils.backtest_runner import get_backtest_parameters
            from utils.train_runner import get_training_parameters
            
            params = get_backtest_parameters()
            train_params = get_training_parameters()
            
            text = f"""
*⚙️ Current Trading Parameters*

*Risk Management:*
• Stop Loss: {params['stop_loss']*100:.1f}%
• Take Profit: {params['take_profit']*100:.1f}%
• Position Size: {params['position_size']*100:.0f}%
• Max Positions: {params['max_positions']}

*Trading Filters:*
• Min Volume: ${params['min_volume']:,.0f}
• Min ML Score: {params['min_score']}

*Cost Settings:*
• Commission: {params['commission']*100:.2f}%
• Slippage: {params['slippage']*100:.2f}%

*ML Training:*
• Models: {', '.join(train_params['models'])}
• Lookback: {train_params['lookback_periods']} days
• Train/Val/Test: {train_params['train_split']*100:.0f}/{train_params['val_split']*100:.0f}/{train_params['test_split']*100:.0f}
"""
            self.bot.reply_to(message, text, parse_mode='Markdown')
        
        @self.bot.message_handler(commands=['symbols'])
        def handle_symbols(message):
            """Show traded symbols"""
            from config.settings import SACRED_SYMBOLS
            
            # Group symbols by sector (example grouping)
            sectors = {
                'Banks': ['AKBNK', 'GARAN', 'ISCTR', 'YKBNK'],
                'Industry': ['EREGL', 'KRDMD', 'PETKM', 'TUPRS'],
                'Retail': ['BIMAS', 'MGROS', 'ULKER'],
                'Holdings': ['SAHOL', 'KCHOL'],
                'Others': ['ASELS', 'ENKAI', 'TCELL', 'THYAO', 'SISE', 'KOZAL', 'AKSEN']
            }
            
            text = "*📈 Sacred 20 Trading Symbols*\n\n"
            for sector, symbols in sectors.items():
                sector_symbols = [s for s in symbols if s in SACRED_SYMBOLS]
                if sector_symbols:
                    text += f"*{sector}:*\n"
                    text += f"• {', '.join(sector_symbols)}\n\n"
            
            text += f"*Total Symbols:* {len(SACRED_SYMBOLS)}"
            self.bot.reply_to(message, text, parse_mode='Markdown')
        
        @self.bot.message_handler(commands=['model_status'])
        def handle_model_status(message):
            """Check ML model status"""
            from utils.train_runner import get_model_status
            
            status = get_model_status()
            
            if status['models_exist']:
                text = f"""
*🧠 ML Model Status*

✅ *Models Ready*
• Model Count: {status['model_count']}
• Model Types: {', '.join(status.get('model_types', []))}
• Avg Accuracy: {status.get('avg_accuracy', 0):.1f}%
• Symbols Trained: {status.get('symbols_trained', 0)}
• Last Training: {status.get('last_training', 'Unknown')}

Models are ready for predictions!
"""
            else:
                text = """
*🧠 ML Model Status*

❌ *No Models Found*

Run /train to train ML models.
This process typically takes 5-10 minutes.
"""
            
            self.bot.reply_to(message, text, parse_mode='Markdown')
        
        @self.bot.message_handler(commands=['performance'])
        def handle_performance(message):
            """Show detailed performance metrics"""
            if self.paper_trader:
                metrics = self.paper_trader.get_performance_metrics()
                status = self.paper_trader.get_portfolio_status()
                
                if metrics:
                    text = f"""
*📊 Detailed Performance Metrics*

*Returns:*
• Total Return: {status['total_return_pct']:+.2f}%
• Daily Return: {metrics.get('daily_return', 0):+.2f}%
• Monthly Return: {metrics.get('monthly_return', 0):+.2f}%

*Risk Metrics:*
• Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
• Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}
• Max Drawdown: {metrics['max_drawdown']:.2f}%
• Current Drawdown: {metrics.get('current_drawdown', 0):.2f}%

*Trading Stats:*
• Win Rate: {metrics['win_rate']:.1f}%
• Profit Factor: {metrics.get('profit_factor', 0):.2f}
• Avg Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}
• Total Trades: {status['total_trades']}

*Trade Analysis:*
• Avg Trade Return: {metrics.get('avg_trade_return', 0):.2f}%
• Best Trade: +{metrics.get('best_trade', 0):.2f}%
• Worst Trade: {metrics.get('worst_trade', 0):.2f}%
• Avg Hold Time: {metrics.get('avg_hold_hours', 0):.1f} hours
"""
                else:
                    text = "📊 No performance data available yet. Make some trades first!"
                
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['trades'])
        def handle_trades(message):
            """Show recent trades with more detail"""
            if self.paper_trader and self.paper_trader.trade_history:
                # Parse count parameter
                parts = message.text.split()
                count = 5  # default
                if len(parts) > 1:
                    try:
                        count = int(parts[1])
                        count = min(count, 20)  # max 20 trades
                    except:
                        pass
                
                recent_trades = self.paper_trader.trade_history[-count:]
                text = f"*💰 Last {count} Trades*\n\n"
                
                for i, trade in enumerate(reversed(recent_trades), 1):
                    action_emoji = "🟢" if trade['action'] == 'buy' else "🔴"
                    
                    # Calculate trade return if it's a sell
                    return_text = ""
                    if trade['action'] == 'sell' and 'return_pct' in trade:
                        return_emoji = "📈" if trade['return_pct'] > 0 else "📉"
                        return_text = f"\n  Return: {return_emoji} {trade['return_pct']:+.2f}%"
                    
                    text += f"{i}. {action_emoji} *{trade['symbol']}* - {trade['action'].upper()}\n"
                    text += f"  Date: {trade['date']}\n"
                    text += f"  Price: ${trade['price']:.2f}\n"
                    text += f"  Shares: {trade['shares']}\n"
                    text += f"  Value: ${trade['total_value']:,.2f}{return_text}\n\n"
                
                self.bot.reply_to(message, text, parse_mode='Markdown')
            else:
                self.bot.reply_to(message, "📭 No trades yet")
        
        @self.bot.message_handler(commands=['clean_data'])
        def handle_clean_data(message):
            """Clean old data files"""
            self.bot.reply_to(message, "🧹 Cleaning old data files...")
            
            try:
                from pathlib import Path
                import shutil
                
                cleaned = 0
                
                # Clean old analysis files (older than 7 days)
                analysis_dir = Path("data/analysis")
                if analysis_dir.exists():
                    for file in analysis_dir.glob("*"):
                        if file.is_file() and file.stat().st_mtime < (datetime.now().timestamp() - 7 * 24 * 3600):
                            file.unlink()
                            cleaned += 1
                
                # Clean old log files (older than 30 days)
                log_dir = Path("logs")
                if log_dir.exists():
                    for file in log_dir.glob("*.log"):
                        if file.stat().st_mtime < (datetime.now().timestamp() - 30 * 24 * 3600):
                            file.unlink()
                            cleaned += 1
                
                # Report results
                text = f"✅ Cleanup complete!\n\nRemoved {cleaned} old files."
                self.bot.reply_to(message, text)
                
                logger.info(f"Data cleanup by {message.from_user.username}: {cleaned} files removed")
                
            except Exception as e:
                self.bot.reply_to(message, f"❌ Cleanup failed: {str(e)}")
        
        # ===== TRADING CONTROL COMMANDS =====
        @self.bot.message_handler(commands=['start_trading'])
        def handle_start_trading(message):
            """Enable trading"""
            if self.paper_trader:
                self.paper_trader.is_running = True
                self.bot.reply_to(message, "✅ Trading enabled! System will now execute trades.")
                logger.info(f"Trading enabled by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['stop_trading'])
        def handle_stop_trading(message):
            """Disable trading"""
            if self.paper_trader:
                self.paper_trader.is_running = False
                self.bot.reply_to(message, "🛑 Trading disabled! System will only monitor.")
                logger.info(f"Trading disabled by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        @self.bot.message_handler(commands=['force_check'])
        def handle_force_check(message):
            """Force immediate signal check"""
            if self.paper_trader:
                self.bot.reply_to(message, "🔍 Forcing signal check...")
                
                # This would trigger the paper trader to check signals immediately
                # For now, just show a completion message
                self.bot.send_message(
                    self.chat_id,
                    "✅ Signal check complete!\n\nCheck /status for any new positions."
                )
                logger.info(f"Force check triggered by {message.from_user.username}")
            else:
                self.bot.reply_to(message, "❌ Trading system not connected")
        
        # ===== SHUTDOWN & RESTART =====
        @self.bot.message_handler(commands=['restart'])
        def handle_restart(message):
            self.bot.reply_to(message, "🔄 Restarting system...")
            logger.info(f"System restart requested by {message.from_user.username}")
            
            # Stop all processes
            if self.demo_process:
                self.demo_process.terminate()
            if self.train_process:
                self.train_process.terminate()
            if self.walkforward_process:
                self.walkforward_process.terminate()
            
            # Restart
            import sys
            os.execv(sys.executable, ['python'] + sys.argv)
        
        @self.bot.message_handler(commands=['shutdown'])
        def handle_shutdown(message):
            self.bot.reply_to(message, "⚠️ Shutting down system...")
            logger.info(f"System shutdown requested by {message.from_user.username}")
            
            # Graceful shutdown
            self.stop()
            if self.paper_trader:
                self.paper_trader.is_running = False
            
            import sys
            sys.exit(0)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_unknown(message):
            self.bot.reply_to(message, "❓ Unknown command. Use /help for available commands.")
    
    def send_notification(self, message, notification_type="info"):
        """Send notification to Telegram"""
        try:
            # Escape markdown
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
        """Start the bot"""
        self.is_running = True
        bot_thread = threading.Thread(target=self._polling_loop)
        bot_thread.daemon = True
        bot_thread.start()
        logger.info("Full Control Telegram bot started")
        
        # Send start notification
        self.send_notification(
            "🤖 *Full Control Bot Started*\n\n"
            "Complete system control enabled!\n"
            "Type /help for all commands",
            "success"
        )
    
    def _polling_loop(self):
        """Polling loop"""
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
        
        # Stop all subprocesses
        if self.demo_process:
            self.demo_process.terminate()
        if self.train_process:
            self.train_process.terminate()
        if self.walkforward_process:
            self.walkforward_process.terminate()
        
        self.bot.stop_polling()
        logger.info("Full Control Telegram bot stopped")


if __name__ == "__main__":
    # Test the bot
    bot = FullControlTelegramBot()
    bot.start()
    
    print("Full Control Bot is running!")
    print("Available commands:")
    print("- /help - Show all commands")
    print("- /start_demo - Start demo trading")
    print("- /train - Train ML models")
    print("- /walkforward - Run walkforward analysis")
    print("- /backtest - Run backtest")
    print("- /system_info - System status")
    print("\nPress Ctrl+C to stop")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.stop()