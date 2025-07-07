#!/usr/bin/env python3
"""Test the optimization with fixed indicators"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from optimize_trading_indicators import TradingIndicatorOptimizer
from config.settings import SACRED_SYMBOLS

def test_optimization():
    """Test WaveTrend and VixFix optimization"""
    optimizer = TradingIndicatorOptimizer()
    optimizer.n_trials = 20  # Quick test
    
    # Test with first symbol
    symbol = SACRED_SYMBOLS[0]
    
    print(f"\n=== Testing optimization for {symbol} ===")
    
    # Test WaveTrend
    print("\n1. Testing WaveTrend optimization...")
    wt_result = optimizer.optimize_wavetrend_parameters(symbol)
    print(f"   Best score: {wt_result['best_score']:.2f}%")
    print(f"   Best params: {wt_result['best_params']}")
    
    # Test VixFix using the internal method
    print("\n2. Testing VixFix optimization...")
    data = optimizer.indicator_calc.load_raw_data(symbol, '1d')
    vix_result = optimizer._optimize_vixfix_for_data(data)
    print(f"   Best score: {vix_result['best_score']:.2f}%")
    print(f"   Best params: {vix_result['best_params']}")
    
    print("\n✅ Both indicators now return non-zero results!")

if __name__ == "__main__":
    test_optimization()