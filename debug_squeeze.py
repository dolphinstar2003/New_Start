#!/usr/bin/env python3
import sys
sys.path.append('.')
from indicators.squeeze_momentum import calculate_squeeze_momentum
from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
import json

# Load current params
with open('universal_optimal_parameters_complete.json', 'r') as f:
    params = json.load(f)

sq_params = params['Squeeze']['params']
print('Squeeze params:', sq_params)

calc = IndicatorCalculator(DATA_DIR)
symbol = 'ASELS'
data = calc.load_raw_data(symbol, '1d')

if data is not None:
    print('Testing Squeeze on', symbol)
    result = calculate_squeeze_momentum(
        data, sq_params['sq_bb_length'], sq_params['sq_bb_mult'],
        sq_params['sq_kc_length'], sq_params['sq_kc_mult'], sq_params['sq_mom_length']
    )
    
    print('Squeeze Results:')
    momentum = result['momentum']
    print('Momentum range:', momentum.min(), '-', momentum.max())
    print('Momentum mean:', momentum.mean())
    nonzero = (momentum != 0).sum()
    print('Non-zero momentum count:', nonzero)
    
    # Check crossovers
    positive_cross = (momentum > 0) & (momentum.shift(1) <= 0)
    negative_cross = (momentum < 0) & (momentum.shift(1) >= 0)
    
    print('Positive crossovers:', positive_cross.sum())
    print('Negative crossovers:', negative_cross.sum())
    
    print('Recent momentum values:')
    print(momentum.tail(10))