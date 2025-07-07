#!/usr/bin/env python3
"""
Signal Generator for Paper Trading
Generates buy/sell signals based on ensemble strategies
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
from indicators.macd_custom import calculate_macd_custom
from indicators.adx_di import calculate_adx_di
from indicators.supertrend import calculate_supertrend
from indicators.wavetrend import calculate_wavetrend
from indicators.squeeze_momentum import calculate_squeeze_momentum
from indicators.vixfix import calculate_vixfix
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Generates trading signals for different strategies"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        
        # Load universal parameters
        params_file = Path(__file__).parent.parent / 'universal_optimal_parameters_complete.json'
        with open(params_file, 'r') as f:
            self.params = json.load(f)
        
        # Signal history
        self.signal_history = []
        self.last_signal_date = {}
        
    def calculate_all_indicators(self, symbol: str, lookback_days: int = 100) -> dict:
        """Calculate all indicators for a symbol"""
        # Load data
        data = self.calc.load_raw_data(symbol, '1d')
        if data is None or len(data) < lookback_days:
            return {}
        
        # Use recent data
        data = data.tail(lookback_days)
        
        indicators = {}
        
        try:
            # Supertrend
            st_params = self.params['Supertrend']['params']
            st_result = calculate_supertrend(data, st_params['st_period'], st_params['st_multiplier'])
            if st_result is not None and not st_result.empty:
                indicators['supertrend'] = st_result['trend']
                indicators['supertrend_upper'] = st_result['up_band']
                indicators['supertrend_lower'] = st_result['dn_band']
            
            # ADX
            adx_params = self.params['ADX']['params']
            adx_result = calculate_adx_di(data, adx_params['adx_period'], adx_params['adx_threshold'])
            if adx_result is not None and not adx_result.empty:
                indicators['adx'] = adx_result['adx']
                indicators['plus_di'] = adx_result['plus_di']
                indicators['minus_di'] = adx_result['minus_di']
            
            # MACD
            macd_params = self.params['MACD']['params']
            macd_result = calculate_macd_custom(data, macd_params['macd_fast'], macd_params['macd_slow'], macd_params['macd_signal'])
            if macd_result is not None and not macd_result.empty:
                indicators['macd'] = macd_result['macd']
                indicators['macd_signal'] = macd_result['signal']
                indicators['macd_histogram'] = macd_result['histogram']
            
            # WaveTrend
            wt_params = self.params['WaveTrend']['params']
            wt_result = calculate_wavetrend(data, wt_params['wt_n1'], wt_params['wt_n2'])
            if wt_result is not None and not wt_result.empty:
                indicators['wt1'] = wt_result['wt1']
                indicators['wt2'] = wt_result['wt2']
            
            # Squeeze
            sq_params = self.params['Squeeze']['params']
            sq_result = calculate_squeeze_momentum(
                data, sq_params['sq_bb_length'], sq_params['sq_bb_mult'],
                sq_params['sq_kc_length'], sq_params['sq_kc_mult'], sq_params['sq_mom_length']
            )
            if sq_result is not None and not sq_result.empty:
                indicators['squeeze_momentum'] = sq_result['momentum']
                indicators['squeeze_on'] = sq_result['squeeze_on']
            
            # VixFix
            vf_params = self.params['VixFix']['params']
            vf_result = calculate_vixfix(
                data, vf_params['vf_lookback'], vf_params['vf_bb_length'], vf_params['vf_bb_mult']
            )
            if vf_result is not None and not vf_result.empty:
                indicators['vixfix'] = vf_result['vixfix']
                indicators['vix_bb_upper'] = vf_result['bb_upper']
                indicators['vix_high_volatility'] = vf_result['high_volatility']
            
            # Add price data
            indicators['close'] = data['close']
            indicators['volume'] = data['volume']
            indicators['high'] = data['high']
            indicators['low'] = data['low']
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            
        return indicators
    
    def generate_aggressive_signals(self, indicators: dict) -> int:
        """Generate signals for aggressive strategy (Supertrend only)"""
        if 'supertrend' not in indicators:
            return 0
        
        # Get last two values
        current_trend = indicators['supertrend'].iloc[-1]
        prev_trend = indicators['supertrend'].iloc[-2]
        
        # Buy signal: trend changes from -1 to 1
        if current_trend == 1 and prev_trend == -1:
            return 1
        
        # Sell signal: trend changes from 1 to -1
        elif current_trend == -1 and prev_trend == 1:
            return -1
        
        return 0
    
    def generate_balanced_signals(self, indicators: dict) -> int:
        """Generate signals for balanced strategy (Supertrend + ADX + MACD)"""
        required = ['supertrend', 'adx', 'macd', 'macd_signal', 'plus_di', 'minus_di']
        if not all(ind in indicators for ind in required):
            return 0
        
        buy_votes = 0
        sell_votes = 0
        
        # Supertrend signal
        current_trend = indicators['supertrend'].iloc[-1]
        prev_trend = indicators['supertrend'].iloc[-2]
        
        if current_trend == 1 and prev_trend == -1:
            buy_votes += 1
        elif current_trend == -1 and prev_trend == 1:
            sell_votes += 1
        
        # ADX signal
        adx_params = self.params['ADX']['params']
        current_adx = indicators['adx'].iloc[-1]
        current_plus_di = indicators['plus_di'].iloc[-1]
        current_minus_di = indicators['minus_di'].iloc[-1]
        prev_plus_di = indicators['plus_di'].iloc[-2]
        prev_minus_di = indicators['minus_di'].iloc[-2]
        
        if (current_plus_di > current_minus_di and prev_plus_di <= prev_minus_di and 
            current_adx > adx_params['adx_threshold']):
            buy_votes += 1
        elif (current_minus_di > current_plus_di and prev_minus_di <= prev_plus_di) or \
             (current_adx < adx_params['adx_exit_threshold']):
            sell_votes += 1
        
        # MACD signal
        current_macd = indicators['macd'].iloc[-1]
        current_signal = indicators['macd_signal'].iloc[-1]
        prev_macd = indicators['macd'].iloc[-2]
        prev_signal = indicators['macd_signal'].iloc[-2]
        
        if current_macd > current_signal and prev_macd <= prev_signal:
            buy_votes += 1
        elif current_macd < current_signal and prev_macd >= prev_signal:
            sell_votes += 1
        
        # Need 2 out of 3 for buy, any sell signal exits
        if buy_votes >= 2:
            return 1
        elif sell_votes >= 1:
            return -1
        
        return 0
    
    def generate_conservative_signals(self, indicators: dict) -> int:
        """Generate signals for conservative strategy (MACD + ADX focus)"""
        buy_votes = 0
        sell_votes = 0
        total_indicators = 0
        
        # Focus on MACD and ADX (best performing from optimization)
        # 1. MACD (3,585% avg return)
        if all(k in indicators for k in ['macd', 'macd_signal']):
            current_macd = indicators['macd'].iloc[-1]
            current_signal = indicators['macd_signal'].iloc[-1]
            prev_macd = indicators['macd'].iloc[-2]
            prev_signal = indicators['macd_signal'].iloc[-2]
            
            if current_macd > current_signal and prev_macd <= prev_signal:
                buy_votes += 1
            elif current_macd < current_signal and prev_macd >= prev_signal:
                sell_votes += 1
            total_indicators += 1
        
        # 2. ADX (3,603% avg return)
        if all(k in indicators for k in ['adx', 'plus_di', 'minus_di']):
            adx_params = self.params['ADX']['params']
            current_adx = indicators['adx'].iloc[-1]
            current_plus_di = indicators['plus_di'].iloc[-1]
            current_minus_di = indicators['minus_di'].iloc[-1]
            prev_plus_di = indicators['plus_di'].iloc[-2]
            prev_minus_di = indicators['minus_di'].iloc[-2]
            
            if (current_plus_di > current_minus_di and prev_plus_di <= prev_minus_di and 
                current_adx > adx_params['adx_threshold']):
                buy_votes += 1
            elif (current_minus_di > current_plus_di and prev_minus_di <= prev_plus_di) or \
                 (current_adx < adx_params['adx_exit_threshold']):
                sell_votes += 1
            total_indicators += 1
        
        # 3. Supertrend as confirmation (41,347% avg return)
        if 'supertrend' in indicators:
            current_trend = indicators['supertrend'].iloc[-1]
            prev_trend = indicators['supertrend'].iloc[-2]
            
            if current_trend == 1 and prev_trend == -1:
                buy_votes += 0.5  # Half vote as confirmation
            elif current_trend == -1 and prev_trend == 1:
                sell_votes += 0.5
            total_indicators += 0.5
        
        # Conservative strategy: Need both MACD and ADX to agree
        # Plus optional Supertrend confirmation
        if buy_votes >= 1.5:  # Need at least MACD + ADX or one + Supertrend
            return 1
        elif sell_votes >= 1:  # Any sell signal exits
            return -1
        
        return 0
    
    def generate_vixfix_enhanced_signals(self, indicators: dict) -> int:
        """Generate signals for VixFix enhanced strategy"""
        if 'supertrend' not in indicators or 'vixfix' not in indicators:
            return 0
        
        # Base signal from Supertrend
        current_trend = indicators['supertrend'].iloc[-1]
        prev_trend = indicators['supertrend'].iloc[-2]
        
        base_buy = current_trend == 1 and prev_trend == -1
        base_sell = current_trend == -1 and prev_trend == 1
        
        # VixFix enhancement
        current_vix = indicators['vixfix'].iloc[-1]
        vix_bb_upper = indicators['vix_bb_upper'].iloc[-1]
        
        # Buy when:
        # 1. Supertrend buy signal AND
        # 2. VixFix is elevated (fear in market)
        if base_buy:
            if current_vix > vix_bb_upper * 0.8:  # VixFix above 80% of upper band
                return 1  # Strong buy
            elif current_vix > indicators['vixfix'].mean():
                return 1  # Normal buy
        
        # Sell signal remains the same
        if base_sell:
            return -1
        
        # Additional VixFix extreme buy
        if current_vix > vix_bb_upper and 'adx' in indicators:
            # Extreme fear + trending market
            if indicators['adx'].iloc[-1] > 25:
                return 1
        
        return 0
    
    def scan_all_symbols(self, strategy: str = 'balanced') -> dict:
        """Scan all symbols for signals"""
        signals = {}
        
        for symbol in SACRED_SYMBOLS:
            try:
                # Calculate indicators
                indicators = self.calculate_all_indicators(symbol)
                if not indicators:
                    continue
                
                # Generate signal based on strategy
                signal = 0
                if strategy == 'aggressive':
                    signal = self.generate_aggressive_signals(indicators)
                elif strategy == 'balanced':
                    signal = self.generate_balanced_signals(indicators)
                elif strategy == 'conservative':
                    signal = self.generate_conservative_signals(indicators)
                elif strategy == 'vixfix_enhanced':
                    signal = self.generate_vixfix_enhanced_signals(indicators)
                
                if signal != 0:
                    # Check if we had a recent signal for this symbol
                    last_signal = self.last_signal_date.get(symbol)
                    if last_signal and (datetime.now() - last_signal).days < 2:
                        continue  # Skip if signal too recent
                    
                    signals[symbol] = {
                        'signal': signal,
                        'price': float(indicators['close'].iloc[-1]),
                        'volume': float(indicators['volume'].iloc[-1]),
                        'indicators': {
                            'supertrend': float(indicators.get('supertrend', pd.Series([0])).iloc[-1]),
                            'adx': float(indicators.get('adx', pd.Series([0])).iloc[-1]),
                            'macd': float(indicators.get('macd', pd.Series([0])).iloc[-1])
                        }
                    }
                    
                    # Update last signal date
                    self.last_signal_date[symbol] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                continue
        
        # Record signals
        if signals:
            self.signal_history.append({
                'date': datetime.now().isoformat(),
                'strategy': strategy,
                'signals': signals
            })
        
        return signals
    
    def get_exit_signals(self, positions: dict, strategy: str = 'balanced') -> dict:
        """Check for exit signals for existing positions"""
        exit_signals = {}
        
        for symbol in positions.keys():
            try:
                indicators = self.calculate_all_indicators(symbol)
                if not indicators:
                    continue
                
                # Generate signal
                signal = 0
                if strategy == 'aggressive':
                    signal = self.generate_aggressive_signals(indicators)
                elif strategy == 'balanced':
                    signal = self.generate_balanced_signals(indicators)
                elif strategy == 'conservative':
                    signal = self.generate_conservative_signals(indicators)
                elif strategy == 'vixfix_enhanced':
                    signal = self.generate_vixfix_enhanced_signals(indicators)
                
                # Exit on sell signal
                if signal == -1:
                    exit_signals[symbol] = {
                        'price': float(indicators['close'].iloc[-1]),
                        'reason': 'signal'
                    }
                    
            except Exception as e:
                logger.error(f"Error checking exit for {symbol}: {e}")
                continue
        
        return exit_signals
    
    def save_signal_history(self, filepath: str = "paper_trading/data/signal_history.json"):
        """Save signal history to file"""
        # Keep only last 1000 signals
        recent_history = self.signal_history[-1000:]
        
        with open(filepath, 'w') as f:
            json.dump(recent_history, f, indent=2)


if __name__ == "__main__":
    # Test signal generator
    sg = SignalGenerator()
    
    print("Testing signal generation for all strategies...\n")
    
    strategies = ['aggressive', 'balanced', 'conservative', 'vixfix_enhanced']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy Signals:")
        print("-" * 50)
        
        signals = sg.scan_all_symbols(strategy)
        
        if signals:
            for symbol, signal_data in signals.items():
                signal_type = "BUY" if signal_data['signal'] == 1 else "SELL"
                print(f"{symbol}: {signal_type} @ {signal_data['price']:.2f}")
        else:
            print("No signals generated")
    
    # Save history
    sg.save_signal_history()