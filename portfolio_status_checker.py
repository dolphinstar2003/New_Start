#!/usr/bin/env python3
"""
Check current portfolio status and positions
"""
import pickle
from pathlib import Path
from datetime import datetime

# Check if there's a saved state
state_file = Path("data/portfolio_trading_state.pkl")

if state_file.exists():
    with open(state_file, 'rb') as f:
        state = pickle.load(f)
    
    print("\n" + "="*60)
    print("PORTFOLIO TRADING STATE")
    print("="*60)
    
    if 'positions' in state:
        print(f"\nOpen Positions: {len(state['positions'])}")
        for symbol, pos in state['positions'].items():
            print(f"  {symbol}: Entry=${pos.get('entry_price', 0):.2f}, Size=${pos.get('size', 0):.2f}")
    
    if 'trades_log' in state:
        print(f"\nTotal Trades: {len(state['trades_log'])}")
        recent = state['trades_log'][-5:] if len(state['trades_log']) > 5 else state['trades_log']
        print("\nRecent Trades:")
        for trade in recent:
            print(f"  {trade.get('timestamp', 'N/A')} - {trade.get('action')} {trade.get('symbol')} @ ${trade.get('price', 0):.2f}")
    
    print("\nLast Update:", state.get('last_update', 'Unknown'))
else:
    print("No saved portfolio state found")

# Also check paper trader state
paper_state_file = Path("data/paper_trading_state.pkl")
if paper_state_file.exists():
    print("\n✅ Paper trading state file exists")
else:
    print("\n❌ No paper trading state file")