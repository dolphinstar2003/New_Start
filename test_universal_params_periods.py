#!/usr/bin/env python3
"""
Test Universal Parameters on Different Time Periods
Test 30, 90, 180, 360 day periods to find most robust parameters
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom
from indicators.adx_di import calculate_adx_di
from indicators.supertrend import calculate_supertrend


class PeriodTester:
    """Test universal parameters on different time periods"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        self.timeframe = '1d'
        
        # Time periods to test (in days)
        self.test_periods = [30, 90, 180, 360]
        
        # Universal parameters from optimization
        self.universal_params = {
            'MACD': {
                'fast': 8,
                'slow': 21,
                'signal': 5
            },
            'ADX': {
                'period': 7,
                'threshold': 15.03,
                'exit_threshold': 11.91
            }
        }
        
        # Load all data
        self.all_data = {}
        print("Loading data for all symbols...")
        for symbol in SACRED_SYMBOLS:
            data = self.calc.load_raw_data(symbol, self.timeframe)
            if data is not None:
                self.all_data[symbol] = data
        print(f"Loaded data for {len(self.all_data)} symbols\n")
    
    def calculate_strategy_return(self, data, signals):
        """Calculate strategy return from signals"""
        returns = data['close'].pct_change().fillna(0)
        position = 0
        strategy_returns = []
        
        for i in range(len(signals)):
            if signals[i] == 1:  # Buy
                position = 1
            elif signals[i] == -1:  # Sell
                position = 0
            
            if i+1 < len(returns):
                strategy_returns.append(position * returns.iloc[i+1])
        
        if strategy_returns:
            total_return = (1 + pd.Series(strategy_returns)).prod() - 1
            return total_return * 100
        return 0
    
    def test_macd_periods(self):
        """Test MACD on different time periods"""
        print("="*80)
        print("📊 TESTING MACD ON DIFFERENT TIME PERIODS")
        print("="*80)
        print(f"Parameters: Fast={self.universal_params['MACD']['fast']}, "
              f"Slow={self.universal_params['MACD']['slow']}, "
              f"Signal={self.universal_params['MACD']['signal']}")
        print("-"*80)
        
        results_by_period = {}
        
        for period_days in self.test_periods:
            print(f"\n📅 Testing {period_days}-day period:")
            period_results = []
            
            for symbol, full_data in self.all_data.items():
                # Get last N days of data
                data = full_data.tail(period_days + 50)  # Extra for indicator warmup
                
                try:
                    # Calculate MACD
                    result = calculate_macd_custom(
                        data, 
                        self.universal_params['MACD']['fast'],
                        self.universal_params['MACD']['slow'],
                        self.universal_params['MACD']['signal']
                    )
                    
                    if result is None:
                        continue
                    
                    macd_line = result['macd']
                    macd_signal = result['signal']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(macd_line)):
                        if pd.isna(macd_line.iloc[i]) or pd.isna(macd_signal.iloc[i]):
                            signals.append(0)
                        elif macd_line.iloc[i] > macd_signal.iloc[i] and macd_line.iloc[i-1] <= macd_signal.iloc[i-1]:
                            signals.append(1)
                        elif macd_line.iloc[i] < macd_signal.iloc[i] and macd_line.iloc[i-1] >= macd_signal.iloc[i-1]:
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                    
                    period_results.append({
                        'symbol': symbol,
                        'return': ret,
                        'buy_hold': buy_hold,
                        'outperform': ret - buy_hold
                    })
                    
                except Exception as e:
                    continue
            
            # Calculate statistics
            returns = [r['return'] for r in period_results]
            positive_count = sum(1 for r in returns if r > 0)
            outperform_count = sum(1 for r in period_results if r['outperform'] > 0)
            
            avg_return = np.mean(returns) if returns else 0
            median_return = np.median(returns) if returns else 0
            
            results_by_period[period_days] = {
                'results': period_results,
                'stats': {
                    'positive_count': positive_count,
                    'outperform_count': outperform_count,
                    'avg_return': avg_return,
                    'median_return': median_return
                }
            }
            
            # Show top performers
            sorted_results = sorted(period_results, key=lambda x: x['return'], reverse=True)
            print(f"\n  Top 5 performers:")
            for i, r in enumerate(sorted_results[:5]):
                print(f"    {i+1}. {r['symbol']}: {r['return']:.1f}% (B&H: {r['buy_hold']:.1f}%)")
            
            print(f"\n  Statistics:")
            print(f"    Positive returns: {positive_count}/{len(period_results)}")
            print(f"    Beat buy&hold: {outperform_count}/{len(period_results)}")
            print(f"    Average return: {avg_return:.1f}%")
            print(f"    Median return: {median_return:.1f}%")
        
        return results_by_period
    
    def test_adx_periods(self):
        """Test ADX on different time periods"""
        print("\n" + "="*80)
        print("📊 TESTING ADX ON DIFFERENT TIME PERIODS")
        print("="*80)
        print(f"Parameters: Period={self.universal_params['ADX']['period']}, "
              f"Threshold={self.universal_params['ADX']['threshold']:.2f}, "
              f"Exit={self.universal_params['ADX']['exit_threshold']:.2f}")
        print("-"*80)
        
        results_by_period = {}
        
        for period_days in self.test_periods:
            print(f"\n📅 Testing {period_days}-day period:")
            period_results = []
            
            for symbol, full_data in self.all_data.items():
                # Get last N days of data
                data = full_data.tail(period_days + 50)
                
                try:
                    # Calculate ADX
                    result = calculate_adx_di(
                        data,
                        length=self.universal_params['ADX']['period'],
                        threshold=self.universal_params['ADX']['threshold']
                    )
                    
                    if result is None:
                        continue
                    
                    adx = result['adx']
                    plus_di = result['plus_di']
                    minus_di = result['minus_di']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(adx)):
                        if pd.isna(adx.iloc[i]) or pd.isna(plus_di.iloc[i]) or pd.isna(minus_di.iloc[i]):
                            signals.append(0)
                        elif (plus_di.iloc[i] > minus_di.iloc[i] and plus_di.iloc[i-1] <= minus_di.iloc[i-1] 
                              and adx.iloc[i] > self.universal_params['ADX']['threshold']):
                            signals.append(1)
                        elif (minus_di.iloc[i] > plus_di.iloc[i] and minus_di.iloc[i-1] <= plus_di.iloc[i-1]) or \
                             (adx.iloc[i] < self.universal_params['ADX']['exit_threshold']):
                            signals.append(-1)
                        else:
                            signals.append(0)
                    
                    # Calculate return
                    ret = self.calculate_strategy_return(data, signals)
                    buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                    
                    period_results.append({
                        'symbol': symbol,
                        'return': ret,
                        'buy_hold': buy_hold,
                        'outperform': ret - buy_hold
                    })
                    
                except Exception as e:
                    continue
            
            # Calculate statistics
            returns = [r['return'] for r in period_results]
            positive_count = sum(1 for r in returns if r > 0)
            outperform_count = sum(1 for r in period_results if r['outperform'] > 0)
            
            avg_return = np.mean(returns) if returns else 0
            median_return = np.median(returns) if returns else 0
            
            results_by_period[period_days] = {
                'results': period_results,
                'stats': {
                    'positive_count': positive_count,
                    'outperform_count': outperform_count,
                    'avg_return': avg_return,
                    'median_return': median_return
                }
            }
            
            # Show top performers
            sorted_results = sorted(period_results, key=lambda x: x['return'], reverse=True)
            print(f"\n  Top 5 performers:")
            for i, r in enumerate(sorted_results[:5]):
                print(f"    {i+1}. {r['symbol']}: {r['return']:.1f}% (B&H: {r['buy_hold']:.1f}%)")
            
            print(f"\n  Statistics:")
            print(f"    Positive returns: {positive_count}/{len(period_results)}")
            print(f"    Beat buy&hold: {outperform_count}/{len(period_results)}")
            print(f"    Average return: {avg_return:.1f}%")
            print(f"    Median return: {median_return:.1f}%")
        
        return results_by_period
    
    def summarize_results(self, macd_results, adx_results):
        """Summarize results across all periods"""
        print("\n" + "="*80)
        print("📈 SUMMARY: PERFORMANCE ACROSS TIME PERIODS")
        print("="*80)
        
        # MACD Summary
        print("\n🔵 MACD Performance Summary:")
        print(f"{'Period':<10} {'Avg Return':<12} {'Win Rate':<12} {'Beat B&H':<12}")
        print("-"*46)
        
        for period in self.test_periods:
            stats = macd_results[period]['stats']
            total = len(macd_results[period]['results'])
            win_rate = (stats['positive_count'] / total * 100) if total > 0 else 0
            beat_rate = (stats['outperform_count'] / total * 100) if total > 0 else 0
            
            print(f"{period}d{' '*6} {stats['avg_return']:>8.1f}%    "
                  f"{win_rate:>8.1f}%    {beat_rate:>8.1f}%")
        
        # ADX Summary
        print("\n🟢 ADX Performance Summary:")
        print(f"{'Period':<10} {'Avg Return':<12} {'Win Rate':<12} {'Beat B&H':<12}")
        print("-"*46)
        
        for period in self.test_periods:
            stats = adx_results[period]['stats']
            total = len(adx_results[period]['results'])
            win_rate = (stats['positive_count'] / total * 100) if total > 0 else 0
            beat_rate = (stats['outperform_count'] / total * 100) if total > 0 else 0
            
            print(f"{period}d{' '*6} {stats['avg_return']:>8.1f}%    "
                  f"{win_rate:>8.1f}%    {beat_rate:>8.1f}%")
        
        # Save results
        all_results = {
            'MACD': macd_results,
            'ADX': adx_results,
            'test_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('universal_params_period_test.json', 'w') as f:
            # Convert numpy types
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                else:
                    return obj
            
            json.dump(convert_types(all_results), f, indent=2)
        
        print("\n✅ Results saved to: universal_params_period_test.json")


def main():
    """Main function"""
    tester = PeriodTester()
    
    # Test MACD
    macd_results = tester.test_macd_periods()
    
    # Test ADX
    adx_results = tester.test_adx_periods()
    
    # Summarize
    tester.summarize_results(macd_results, adx_results)


if __name__ == "__main__":
    main()