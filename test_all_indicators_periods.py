#!/usr/bin/env python3
"""
Test ALL Universal Parameters on Different Time Periods
Test all 6 indicators on 30, 90, 180, 360 day periods
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
from indicators.wavetrend import calculate_wavetrend
from indicators.squeeze_momentum import calculate_squeeze_momentum
from indicators.vixfix import calculate_vixfix


class AllIndicatorsPeriodTester:
    """Test all indicators on different time periods"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        self.timeframe = '1d'
        
        # Time periods to test (in days)
        self.test_periods = [30, 90, 180, 360]
        
        # Load universal parameters
        with open('universal_optimal_parameters_complete.json', 'r') as f:
            self.universal_params = json.load(f)
        
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
    
    def test_supertrend_periods(self):
        """Test Supertrend on different time periods"""
        print("="*80)
        print("🌟 TESTING SUPERTREND ON DIFFERENT TIME PERIODS")
        print("="*80)
        params = self.universal_params['Supertrend']['params']
        print(f"Parameters: Period={params['st_period']}, Multiplier={params['st_multiplier']:.2f}")
        print("-"*80)
        
        results_by_period = {}
        
        for period_days in self.test_periods:
            print(f"\n📅 Testing {period_days}-day period:")
            period_results = []
            
            for symbol, full_data in self.all_data.items():
                # Get last N days of data
                data = full_data.tail(period_days + 50)  # Extra for indicator warmup
                
                try:
                    # Calculate Supertrend
                    result = calculate_supertrend(
                        data, 
                        params['st_period'],
                        params['st_multiplier']
                    )
                    
                    if result is None:
                        continue
                    
                    trend = result['trend']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(trend)):
                        if pd.isna(trend.iloc[i]):
                            signals.append(0)
                        elif trend.iloc[i] == 1 and trend.iloc[i-1] == -1:
                            signals.append(1)
                        elif trend.iloc[i] == -1 and trend.iloc[i-1] == 1:
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
    
    def test_wavetrend_periods(self):
        """Test WaveTrend on different time periods"""
        print("\n" + "="*80)
        print("🌊 TESTING WAVETREND ON DIFFERENT TIME PERIODS")
        print("="*80)
        params = self.universal_params['WaveTrend']['params']
        print(f"Parameters: N1={params['wt_n1']}, N2={params['wt_n2']}, OB={params['wt_overbought']}, OS={params['wt_oversold']}")
        print("-"*80)
        
        results_by_period = {}
        
        for period_days in self.test_periods:
            print(f"\n📅 Testing {period_days}-day period:")
            period_results = []
            
            for symbol, full_data in self.all_data.items():
                data = full_data.tail(period_days + 50)
                
                try:
                    # Calculate WaveTrend
                    result = calculate_wavetrend(
                        data,
                        params['wt_n1'],
                        params['wt_n2']
                    )
                    
                    if result is None:
                        continue
                    
                    wt1 = result['wt1']
                    wt2 = result['wt2']
                    
                    # Generate signals
                    signals = []
                    for i in range(1, len(wt1)):
                        if pd.isna(wt1.iloc[i]) or pd.isna(wt2.iloc[i]):
                            signals.append(0)
                        elif (wt1.iloc[i] > wt2.iloc[i] and wt1.iloc[i-1] <= wt2.iloc[i-1] 
                              and wt1.iloc[i] < params['wt_oversold']):
                            signals.append(1)
                        elif (wt1.iloc[i] < wt2.iloc[i] and wt1.iloc[i-1] >= wt2.iloc[i-1] 
                              and wt1.iloc[i] > params['wt_overbought']):
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
            
            # Statistics
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
            
            # Show results
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
    
    def test_squeeze_periods(self):
        """Test Squeeze Momentum on different time periods"""
        print("\n" + "="*80)
        print("🔥 TESTING SQUEEZE MOMENTUM ON DIFFERENT TIME PERIODS")
        print("="*80)
        params = self.universal_params['Squeeze']['params']
        print(f"Parameters: BB={params['sq_bb_length']}/{params['sq_bb_mult']}, KC={params['sq_kc_length']}/{params['sq_kc_mult']}, Mom={params['sq_mom_length']}")
        print("-"*80)
        
        results_by_period = {}
        
        for period_days in self.test_periods:
            print(f"\n📅 Testing {period_days}-day period:")
            period_results = []
            
            for symbol, full_data in self.all_data.items():
                data = full_data.tail(period_days + 50)
                
                try:
                    # Calculate Squeeze
                    result = calculate_squeeze_momentum(
                        data,
                        params['sq_bb_length'],
                        params['sq_bb_mult'],
                        params['sq_kc_length'],
                        params['sq_kc_mult'],
                        params['sq_mom_length']
                    )
                    
                    if result is None:
                        continue
                    
                    momentum = result['momentum']
                    
                    # Simple momentum crossover
                    signals = []
                    for i in range(1, len(momentum)):
                        if pd.isna(momentum.iloc[i]):
                            signals.append(0)
                        elif momentum.iloc[i] > 0 and momentum.iloc[i-1] <= 0:
                            signals.append(1)
                        elif momentum.iloc[i] < 0 and momentum.iloc[i-1] >= 0:
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
            
            # Statistics
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
            
            # Show results
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
    
    def summarize_all_results(self, all_results):
        """Summarize results for all indicators"""
        print("\n" + "="*80)
        print("📈 SUMMARY: ALL INDICATORS PERFORMANCE ACROSS TIME PERIODS")
        print("="*80)
        
        # Summary table
        print("\n📊 Average Returns by Period:")
        print(f"{'Indicator':<15} {'30d':<12} {'90d':<12} {'180d':<12} {'360d':<12}")
        print("-" * 63)
        
        indicators = ['MACD', 'ADX', 'Supertrend', 'WaveTrend', 'Squeeze']
        
        for ind in indicators:
            if ind in all_results:
                row = f"{ind:<15}"
                for period in self.test_periods:
                    if period in all_results[ind]:
                        avg_ret = all_results[ind][period]['stats']['avg_return']
                        row += f"{avg_ret:>10.1f}% "
                    else:
                        row += f"{'N/A':>10} "
                print(row)
        
        # Win rates
        print("\n📊 Win Rates by Period (% positive returns):")
        print(f"{'Indicator':<15} {'30d':<12} {'90d':<12} {'180d':<12} {'360d':<12}")
        print("-" * 63)
        
        for ind in indicators:
            if ind in all_results:
                row = f"{ind:<15}"
                for period in self.test_periods:
                    if period in all_results[ind]:
                        total = len(all_results[ind][period]['results'])
                        positive = all_results[ind][period]['stats']['positive_count']
                        win_rate = (positive / total * 100) if total > 0 else 0
                        row += f"{win_rate:>10.1f}% "
                    else:
                        row += f"{'N/A':>10} "
                print(row)
        
        # Beat B&H rates
        print("\n📊 Beat Buy&Hold Rates by Period:")
        print(f"{'Indicator':<15} {'30d':<12} {'90d':<12} {'180d':<12} {'360d':<12}")
        print("-" * 63)
        
        for ind in indicators:
            if ind in all_results:
                row = f"{ind:<15}"
                for period in self.test_periods:
                    if period in all_results[ind]:
                        total = len(all_results[ind][period]['results'])
                        beat = all_results[ind][period]['stats']['outperform_count']
                        beat_rate = (beat / total * 100) if total > 0 else 0
                        row += f"{beat_rate:>10.1f}% "
                    else:
                        row += f"{'N/A':>10} "
                print(row)
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        complete_results = {
            'test_date': timestamp,
            'periods': self.test_periods,
            'results': all_results
        }
        
        with open('all_indicators_period_test_results.json', 'w') as f:
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
            
            json.dump(convert_types(complete_results), f, indent=2)
        
        print("\n✅ Complete results saved to: all_indicators_period_test_results.json")


def main():
    """Main function"""
    tester = AllIndicatorsPeriodTester()
    
    # Load existing results for MACD and ADX
    all_results = {}
    
    # We already have MACD and ADX results from previous test
    try:
        with open('universal_params_period_test.json', 'r') as f:
            existing = json.load(f)
            all_results['MACD'] = existing.get('MACD', {})
            all_results['ADX'] = existing.get('ADX', {})
    except:
        pass
    
    # Test remaining indicators
    print("🚀 Testing remaining indicators on different time periods...")
    print("This will test: Supertrend, WaveTrend, Squeeze Momentum\n")
    
    # Test each indicator
    all_results['Supertrend'] = tester.test_supertrend_periods()
    all_results['WaveTrend'] = tester.test_wavetrend_periods()
    all_results['Squeeze'] = tester.test_squeeze_periods()
    
    # Summarize all results
    tester.summarize_all_results(all_results)


if __name__ == "__main__":
    main()