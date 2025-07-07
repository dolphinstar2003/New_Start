# Backtest module initialization
from .realistic_backtest import run_realistic_backtest

# Keep the old function name for compatibility
async def run_backtest(days=30):
    """Run indicator-based backtest"""
    return await run_realistic_backtest(days)