#!/usr/bin/env python3
"""
Create Ensemble Strategy with VixFix Filter
Combine top indicators with volatility filtering
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
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


class EnsembleStrategy:
    """Ensemble strategy combining multiple indicators with VixFix filter"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        
        # Load universal parameters
        with open('universal_optimal_parameters_complete.json', 'r') as f:
            self.params = json.load(f)
        
        # Strategy configurations
        self.strategies = {
            'aggressive': {
                'name': 'Aggressive (Supertrend Only)',
                'description': 'Uses only Supertrend for maximum returns',
                'indicators': ['supertrend'],
                'vixfix_filter': False
            },
            'balanced': {
                'name': 'Balanced (Top 3)',
                'description': 'Combines Supertrend + ADX + MACD',
                'indicators': ['supertrend', 'adx', 'macd'],
                'vixfix_filter': True
            },
            'conservative': {
                'name': 'Conservative (All Confirmations)',
                'description': 'Requires 3 out of 5 indicators + VixFix filter',
                'indicators': ['supertrend', 'adx', 'macd', 'wavetrend', 'squeeze'],
                'vixfix_filter': True
            },
            'vix_enhanced': {
                'name': 'VixFix Enhanced',
                'description': 'Uses VixFix to time entries with top indicators',
                'indicators': ['supertrend', 'adx'],
                'vixfix_filter': True,
                'vix_timing': True
            }
        }
    
    def calculate_indicators(self, data, symbol):
        """Calculate all indicators for given data"""
        indicators = {}
        
        # Supertrend
        st_params = self.params['Supertrend']['params']
        st_result = calculate_supertrend(data, st_params['st_period'], st_params['st_multiplier'])
        if st_result is not None:
            indicators['supertrend'] = st_result['trend']
        
        # ADX
        adx_params = self.params['ADX']['params']
        adx_result = calculate_adx_di(data, adx_params['adx_period'], adx_params['adx_threshold'])
        if adx_result is not None:
            indicators['adx'] = adx_result['adx']
            indicators['plus_di'] = adx_result['plus_di']
            indicators['minus_di'] = adx_result['minus_di']
        
        # MACD
        macd_params = self.params['MACD']['params']
        macd_result = calculate_macd_custom(data, macd_params['macd_fast'], macd_params['macd_slow'], macd_params['macd_signal'])
        if macd_result is not None:
            indicators['macd'] = macd_result['macd']
            indicators['macd_signal'] = macd_result['signal']
        
        # WaveTrend
        wt_params = self.params['WaveTrend']['params']
        wt_result = calculate_wavetrend(data, wt_params['wt_n1'], wt_params['wt_n2'])
        if wt_result is not None:
            indicators['wt1'] = wt_result['wt1']
            indicators['wt2'] = wt_result['wt2']
        
        # Squeeze
        sq_params = self.params['Squeeze']['params']
        sq_result = calculate_squeeze_momentum(
            data, sq_params['sq_bb_length'], sq_params['sq_bb_mult'],
            sq_params['sq_kc_length'], sq_params['sq_kc_mult'], sq_params['sq_mom_length']
        )
        if sq_result is not None:
            indicators['squeeze_momentum'] = sq_result['momentum']
        
        # VixFix
        vf_params = self.params['VixFix']['params']
        vf_result = calculate_vixfix(
            data, vf_params['vf_lookback'], vf_params['vf_bb_length'], vf_params['vf_bb_mult']
        )
        if vf_result is not None:
            indicators['vixfix'] = vf_result['vixfix']
            indicators['vix_bb_upper'] = vf_result['bb_upper']
            indicators['vix_high_volatility'] = vf_result['high_volatility']
        
        return indicators
    
    def generate_signals(self, data, indicators, strategy_type='balanced'):
        """Generate buy/sell signals based on strategy type"""
        strategy = self.strategies[strategy_type]
        signals = []
        
        for i in range(1, len(data)):
            buy_votes = 0
            sell_votes = 0
            total_indicators = 0
            
            # Check each indicator
            if 'supertrend' in strategy['indicators'] and 'supertrend' in indicators:
                if not pd.isna(indicators['supertrend'].iloc[i]):
                    total_indicators += 1
                    if indicators['supertrend'].iloc[i] == 1 and indicators['supertrend'].iloc[i-1] == -1:
                        buy_votes += 1
                    elif indicators['supertrend'].iloc[i] == -1 and indicators['supertrend'].iloc[i-1] == 1:
                        sell_votes += 1
            
            if 'adx' in strategy['indicators'] and all(k in indicators for k in ['adx', 'plus_di', 'minus_di']):
                adx_params = self.params['ADX']['params']
                if not pd.isna(indicators['adx'].iloc[i]):
                    total_indicators += 1
                    if (indicators['plus_di'].iloc[i] > indicators['minus_di'].iloc[i] and 
                        indicators['plus_di'].iloc[i-1] <= indicators['minus_di'].iloc[i-1] and
                        indicators['adx'].iloc[i] > adx_params['adx_threshold']):
                        buy_votes += 1
                    elif (indicators['minus_di'].iloc[i] > indicators['plus_di'].iloc[i] and 
                          indicators['minus_di'].iloc[i-1] <= indicators['plus_di'].iloc[i-1]) or \
                         (indicators['adx'].iloc[i] < adx_params['adx_exit_threshold']):
                        sell_votes += 1
            
            if 'macd' in strategy['indicators'] and all(k in indicators for k in ['macd', 'macd_signal']):
                if not pd.isna(indicators['macd'].iloc[i]):
                    total_indicators += 1
                    if indicators['macd'].iloc[i] > indicators['macd_signal'].iloc[i] and \
                       indicators['macd'].iloc[i-1] <= indicators['macd_signal'].iloc[i-1]:
                        buy_votes += 1
                    elif indicators['macd'].iloc[i] < indicators['macd_signal'].iloc[i] and \
                         indicators['macd'].iloc[i-1] >= indicators['macd_signal'].iloc[i-1]:
                        sell_votes += 1
            
            if 'wavetrend' in strategy['indicators'] and all(k in indicators for k in ['wt1', 'wt2']):
                wt_params = self.params['WaveTrend']['params']
                if not pd.isna(indicators['wt1'].iloc[i]):
                    total_indicators += 1
                    if (indicators['wt1'].iloc[i] > indicators['wt2'].iloc[i] and 
                        indicators['wt1'].iloc[i-1] <= indicators['wt2'].iloc[i-1] and
                        indicators['wt1'].iloc[i] < wt_params['wt_oversold']):
                        buy_votes += 1
                    elif (indicators['wt1'].iloc[i] < indicators['wt2'].iloc[i] and 
                          indicators['wt1'].iloc[i-1] >= indicators['wt2'].iloc[i-1] and
                          indicators['wt1'].iloc[i] > wt_params['wt_overbought']):
                        sell_votes += 1
            
            if 'squeeze' in strategy['indicators'] and 'squeeze_momentum' in indicators:
                if not pd.isna(indicators['squeeze_momentum'].iloc[i]):
                    total_indicators += 1
                    if indicators['squeeze_momentum'].iloc[i] > 0 and indicators['squeeze_momentum'].iloc[i-1] <= 0:
                        buy_votes += 1
                    elif indicators['squeeze_momentum'].iloc[i] < 0 and indicators['squeeze_momentum'].iloc[i-1] >= 0:
                        sell_votes += 1
            
            # Apply VixFix filter
            vix_filter_pass = True
            if strategy.get('vixfix_filter', False) and 'vix_high_volatility' in indicators:
                # In high volatility, be more conservative
                if indicators['vix_high_volatility'].iloc[i]:
                    # Require more confirmations in high volatility
                    if strategy_type == 'balanced':
                        required_votes = 2  # Need 2 out of 3
                    elif strategy_type == 'conservative':
                        required_votes = 3  # Need 3 out of 5
                    else:
                        required_votes = 1
                else:
                    # Normal volatility - standard requirements
                    if strategy_type == 'balanced':
                        required_votes = 2
                    elif strategy_type == 'conservative':
                        required_votes = 2
                    else:
                        required_votes = 1
            else:
                # No VixFix filter
                if strategy_type == 'aggressive':
                    required_votes = 1
                elif strategy_type == 'balanced':
                    required_votes = 2
                else:
                    required_votes = 2
            
            # Special handling for vix_enhanced strategy
            if strategy.get('vix_timing', False) and 'vixfix' in indicators:
                # Buy when VixFix spikes (fear) and primary indicators confirm
                if indicators['vixfix'].iloc[i] > indicators['vix_bb_upper'].iloc[i] and buy_votes > 0:
                    buy_votes += 1  # Extra vote for VixFix spike
            
            # Generate signal
            if buy_votes >= required_votes:
                signals.append(1)
            elif sell_votes >= 1:  # Any sell signal exits
                signals.append(-1)
            else:
                signals.append(0)
        
        return signals
    
    def backtest_strategy(self, symbol, strategy_type='balanced'):
        """Backtest a strategy on a single symbol"""
        data = self.calc.load_raw_data(symbol, '1d')
        if data is None:
            return None
        
        # Calculate indicators
        indicators = self.calculate_indicators(data, symbol)
        
        # Generate signals
        signals = self.generate_signals(data, indicators, strategy_type)
        
        # Calculate returns
        returns = data['close'].pct_change().fillna(0)
        position = 0
        strategy_returns = []
        trades = []
        
        for i in range(len(signals)):
            if signals[i] == 1:  # Buy
                if position == 0:
                    position = 1
                    trades.append({'type': 'buy', 'date': data.index[i], 'price': data['close'].iloc[i]})
            elif signals[i] == -1:  # Sell
                if position == 1:
                    position = 0
                    trades.append({'type': 'sell', 'date': data.index[i], 'price': data['close'].iloc[i]})
            
            if i+1 < len(returns):
                strategy_returns.append(position * returns.iloc[i+1])
        
        # Calculate metrics
        if strategy_returns:
            total_return = (1 + pd.Series(strategy_returns)).prod() - 1
            buy_hold = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1)
            
            # Calculate Sharpe ratio
            if len(strategy_returns) > 1:
                sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)
            else:
                sharpe = 0
            
            # Calculate max drawdown
            cumulative = (1 + pd.Series(strategy_returns)).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            return {
                'symbol': symbol,
                'strategy': strategy_type,
                'total_return': total_return * 100,
                'buy_hold_return': buy_hold * 100,
                'outperformance': (total_return - buy_hold) * 100,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown * 100,
                'num_trades': len([t for t in trades if t['type'] == 'buy']),
                'win_rate': sum(1 for r in strategy_returns if r > 0) / len(strategy_returns) * 100 if strategy_returns else 0
            }
        
        return None
    
    def compare_all_strategies(self):
        """Compare all strategies across all symbols"""
        print("🔬 ENSEMBLE STRATEGY COMPARISON")
        print("="*80)
        
        results = {strategy: [] for strategy in self.strategies.keys()}
        
        # Test each strategy on each symbol
        for symbol in SACRED_SYMBOLS:
            print(f"\nTesting {symbol}...", end='', flush=True)
            
            for strategy_type in self.strategies.keys():
                result = self.backtest_strategy(symbol, strategy_type)
                if result:
                    results[strategy_type].append(result)
            
            print(" ✓")
        
        # Display results
        print("\n" + "="*80)
        print("📊 STRATEGY PERFORMANCE SUMMARY")
        print("="*80)
        
        for strategy_type, strategy_results in results.items():
            if not strategy_results:
                continue
            
            print(f"\n🎯 {self.strategies[strategy_type]['name']}")
            print(f"   {self.strategies[strategy_type]['description']}")
            print("-"*60)
            
            # Calculate averages
            avg_return = np.mean([r['total_return'] for r in strategy_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in strategy_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in strategy_results])
            avg_trades = np.mean([r['num_trades'] for r in strategy_results])
            win_count = sum(1 for r in strategy_results if r['total_return'] > r['buy_hold_return'])
            
            print(f"   Average Return: {avg_return:.1f}%")
            print(f"   Average Sharpe: {avg_sharpe:.2f}")
            print(f"   Average Max Drawdown: {avg_drawdown:.1f}%")
            print(f"   Average Trades: {avg_trades:.0f}")
            print(f"   Beat Buy&Hold: {win_count}/{len(strategy_results)} ({win_count/len(strategy_results)*100:.0f}%)")
            
            # Top performers
            top_3 = sorted(strategy_results, key=lambda x: x['total_return'], reverse=True)[:3]
            print(f"\n   Top 3 Performers:")
            for i, r in enumerate(top_3):
                print(f"   {i+1}. {r['symbol']}: {r['total_return']:.1f}% (B&H: {r['buy_hold_return']:.1f}%)")
        
        # Save detailed results
        with open('ensemble_strategy_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n✅ Detailed results saved to: ensemble_strategy_results.json")


def main():
    """Main function"""
    ensemble = EnsembleStrategy()
    ensemble.compare_all_strategies()


if __name__ == "__main__":
    main()