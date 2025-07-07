#!/usr/bin/env python3
import sys
sys.path.append('.')
from portfolio_walkforward_backtest import PortfolioManager
from datetime import datetime, timedelta
import pandas as pd

# Test signal generation
pm = PortfolioManager()

# Test on ASELS
symbol = 'ASELS'
data = pm.calc.load_raw_data(symbol, '1d')
print(f'Data available for {symbol}: {len(data)} rows')
print(f'Date range: {data.index[0]} to {data.index[-1]}')

# Test recent data
recent_data = data.tail(30)
print(f'\nTesting on recent 30 days...')

# Test aggressive strategy
signals = pm.generate_signals(symbol, recent_data, 'aggressive')
print(f'Aggressive signals: {signals.sum()} total, {(signals != 0).sum()} non-zero')
print(f'Recent aggressive signals: {signals.tail(10).tolist()}')

# Test balanced strategy  
signals = pm.generate_signals(symbol, recent_data, 'balanced')
print(f'Balanced signals: {signals.sum()} total, {(signals != 0).sum()} non-zero')
print(f'Recent balanced signals: {signals.tail(10).tolist()}')

# Test conservative strategy
signals = pm.generate_signals(symbol, recent_data, 'conservative')  
print(f'Conservative signals: {signals.sum()} total, {(signals != 0).sum()} non-zero')
print(f'Recent conservative signals: {signals.tail(10).tolist()}')