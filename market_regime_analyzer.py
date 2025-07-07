#!/usr/bin/env python3
"""
Market Regime Analyzer
Detects market regimes using VIX/volatility, trend analysis, and manual periods
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from indicators.calculator import IndicatorCalculator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegimeAnalyzer:
    """Analyzes market regimes using multiple methods"""
    
    def __init__(self):
        self.calc = IndicatorCalculator(DATA_DIR)
        self.regimes = {}
        self.vix_data = None
        self.bist_data = None
        
    def load_market_data(self):
        """Load VIX and BIST100 data"""
        logger.info("📊 Loading market data for regime analysis...")
        
        # Load VIX data
        try:
            vix = yf.download("^VIX", start="2023-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            self.vix_data = vix['Close'].dropna()
            logger.info(f"✅ VIX data loaded: {len(self.vix_data)} points")
        except Exception as e:
            logger.warning(f"⚠️  Could not load VIX data: {e}")
            # Create synthetic VIX using BIST volatility
            self.create_synthetic_vix()
        
        # Load BIST100 data
        try:
            bist = yf.download("XU100.IS", start="2023-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            self.bist_data = bist['Close'].dropna()
            logger.info(f"✅ BIST100 data loaded: {len(self.bist_data)} points")
        except Exception as e:
            logger.warning(f"⚠️  Could not load BIST100 data: {e}")
            # Use average of sacred symbols
            self.create_synthetic_bist()
    
    def create_synthetic_vix(self):
        """Create synthetic VIX using BIST volatility"""
        logger.info("🔄 Creating synthetic VIX from BIST volatility...")
        
        # Calculate rolling volatility of BIST or sacred symbols
        if self.bist_data is not None:
            returns = self.bist_data.pct_change().dropna()
        else:
            # Use sacred symbols average
            symbol_data = []
            for symbol in SACRED_SYMBOLS[:5]:  # Use top 5 for speed
                data = self.calc.load_raw_data(symbol, '1d')
                if data is not None:
                    symbol_data.append(data['close'])
            
            if symbol_data:
                avg_price = pd.concat(symbol_data, axis=1).mean(axis=1)
                returns = avg_price.pct_change().dropna()
            else:
                logger.error("❌ No data available for synthetic VIX")
                return
        
        # Calculate 20-day rolling volatility (annualized)
        vol = returns.rolling(20).std() * np.sqrt(252) * 100
        self.vix_data = vol.dropna()
        logger.info(f"✅ Synthetic VIX created: {len(self.vix_data)} points")
    
    def create_synthetic_bist(self):
        """Create synthetic BIST using sacred symbols"""
        logger.info("🔄 Creating synthetic BIST from sacred symbols...")
        
        symbol_data = []
        for symbol in SACRED_SYMBOLS:
            data = self.calc.load_raw_data(symbol, '1d')
            if data is not None:
                # Normalize to 100 at start
                normalized = (data['close'] / data['close'].iloc[0]) * 100
                symbol_data.append(normalized)
        
        if symbol_data:
            # Equal weight average
            self.bist_data = pd.concat(symbol_data, axis=1).mean(axis=1)
            logger.info(f"✅ Synthetic BIST created: {len(self.bist_data)} points")
        else:
            logger.error("❌ No data available for synthetic BIST")
    
    def analyze_vix_regimes(self) -> Dict[str, List[Tuple[str, str]]]:
        """Analyze market regimes based on VIX/volatility"""
        logger.info("🎯 Analyzing VIX-based market regimes...")
        
        if self.vix_data is None:
            logger.error("❌ No VIX data available")
            return {}
        
        # Calculate VIX percentiles
        vix_25 = self.vix_data.quantile(0.25)
        vix_75 = self.vix_data.quantile(0.75)
        
        logger.info(f"📈 VIX thresholds: Low<{vix_25:.1f}, {vix_25:.1f}<Medium<{vix_75:.1f}, High>{vix_75:.1f}")
        
        # Classify regimes
        regimes = []
        current_regime = None
        regime_start = None
        
        for date, vix_val in self.vix_data.items():
            if vix_val <= vix_25:
                new_regime = 'low_volatility'  # Bull market
            elif vix_val >= vix_75:
                new_regime = 'high_volatility'  # Bear market
            else:
                new_regime = 'medium_volatility'  # Sideways market
            
            if new_regime != current_regime:
                # End previous regime
                if current_regime is not None and regime_start is not None:
                    regimes.append({
                        'regime': current_regime,
                        'start': regime_start.strftime('%Y-%m-%d'),
                        'end': date.strftime('%Y-%m-%d'),
                        'duration_days': (date - regime_start).days
                    })
                
                # Start new regime
                current_regime = new_regime
                regime_start = date
        
        # Close last regime
        if current_regime is not None and regime_start is not None:
            regimes.append({
                'regime': current_regime,
                'start': regime_start.strftime('%Y-%m-%d'),
                'end': self.vix_data.index[-1].strftime('%Y-%m-%d'),
                'duration_days': (self.vix_data.index[-1] - regime_start).days
            })
        
        # Group by regime type
        vix_regimes = {
            'bull_market': [],  # Low volatility periods
            'bear_market': [],  # High volatility periods
            'sideways_market': []  # Medium volatility periods
        }
        
        for regime in regimes:
            if regime['regime'] == 'low_volatility':
                vix_regimes['bull_market'].append((regime['start'], regime['end']))
            elif regime['regime'] == 'high_volatility':
                vix_regimes['bear_market'].append((regime['start'], regime['end']))
            else:
                vix_regimes['sideways_market'].append((regime['start'], regime['end']))
        
        logger.info(f"🐂 Bull periods: {len(vix_regimes['bull_market'])}")
        logger.info(f"🐻 Bear periods: {len(vix_regimes['bear_market'])}")
        logger.info(f"➡️  Sideways periods: {len(vix_regimes['sideways_market'])}")
        
        self.regimes['vix_based'] = vix_regimes
        return vix_regimes
    
    def analyze_trend_regimes(self) -> Dict[str, List[Tuple[str, str]]]:
        """Analyze market regimes based on trend analysis"""
        logger.info("📈 Analyzing trend-based market regimes...")
        
        if self.bist_data is None:
            logger.error("❌ No BIST data available")
            return {}
        
        # Calculate moving averages
        ma_50 = self.bist_data.rolling(50).mean()
        ma_200 = self.bist_data.rolling(200).mean()
        
        # Calculate trend strength
        price = self.bist_data
        trend_up = (price > ma_50) & (ma_50 > ma_200)
        trend_down = (price < ma_50) & (ma_50 < ma_200)
        trend_sideways = ~(trend_up | trend_down)
        
        # Find regime changes
        regimes = []
        
        # Process trend_up
        in_regime = False
        start_date = None
        
        for regime_type, trend_series in [
            ('bull_market', trend_up),
            ('bear_market', trend_down), 
            ('sideways_market', trend_sideways)
        ]:
            regime_periods = []
            in_regime = False
            start_date = None
            
            for date, is_in_regime in trend_series.items():
                if is_in_regime and not in_regime:
                    # Start of regime
                    in_regime = True
                    start_date = date
                elif not is_in_regime and in_regime:
                    # End of regime
                    if start_date is not None:
                        # Only include periods longer than 5 days
                        if (date - start_date).days >= 5:
                            regime_periods.append((start_date.strftime('%Y-%m-%d'), 
                                                 date.strftime('%Y-%m-%d')))
                    in_regime = False
                    start_date = None
            
            # Close last regime if still active
            if in_regime and start_date is not None:
                regime_periods.append((start_date.strftime('%Y-%m-%d'), 
                                     trend_series.index[-1].strftime('%Y-%m-%d')))
            
            regimes.append((regime_type, regime_periods))
        
        trend_regimes = dict(regimes)
        
        logger.info(f"🐂 Trend Bull periods: {len(trend_regimes['bull_market'])}")
        logger.info(f"🐻 Trend Bear periods: {len(trend_regimes['bear_market'])}")
        logger.info(f"➡️  Trend Sideways periods: {len(trend_regimes['sideways_market'])}")
        
        self.regimes['trend_based'] = trend_regimes
        return trend_regimes
    
    def define_manual_regimes(self) -> Dict[str, List[Tuple[str, str]]]:
        """Define manual market regimes based on known events"""
        logger.info("✋ Defining manual market regimes...")
        
        manual_regimes = {
            'bull_market': [
                ('2023-01-01', '2023-05-15'),  # Early 2023 rally
                ('2023-10-15', '2024-01-15'),  # Late 2023 rally
                ('2024-11-01', '2024-12-31'),  # Late 2024 rally
            ],
            'bear_market': [
                ('2023-05-16', '2023-10-14'),  # Mid 2023 correction
                ('2024-01-16', '2024-03-15'),  # Early 2024 correction
                ('2024-08-01', '2024-10-31'),  # Late 2024 correction
            ],
            'sideways_market': [
                ('2024-03-16', '2024-07-31'),  # Mid 2024 consolidation
            ]
        }
        
        for regime_type, periods in manual_regimes.items():
            logger.info(f"{regime_type}: {len(periods)} periods")
            for start, end in periods:
                logger.info(f"  {start} to {end}")
        
        self.regimes['manual'] = manual_regimes
        return manual_regimes
    
    def get_current_regime(self, method: str = 'vix_based') -> str:
        """Determine current market regime"""
        logger.info(f"🔍 Determining current regime using {method} method...")
        
        if method == 'vix_based' and self.vix_data is not None:
            current_vix = self.vix_data.iloc[-1]
            vix_25 = self.vix_data.quantile(0.25)
            vix_75 = self.vix_data.quantile(0.75)
            
            if current_vix <= vix_25:
                current_regime = 'bull_market'
            elif current_vix >= vix_75:
                current_regime = 'bear_market'
            else:
                current_regime = 'sideways_market'
                
            logger.info(f"📊 Current VIX: {current_vix:.1f}, Regime: {current_regime}")
            
        elif method == 'trend_based' and self.bist_data is not None:
            price = self.bist_data.iloc[-1]
            ma_50 = self.bist_data.rolling(50).mean().iloc[-1]
            ma_200 = self.bist_data.rolling(200).mean().iloc[-1]
            
            if price > ma_50 and ma_50 > ma_200:
                current_regime = 'bull_market'
            elif price < ma_50 and ma_50 < ma_200:
                current_regime = 'bear_market'
            else:
                current_regime = 'sideways_market'
                
            logger.info(f"📈 Price: {price:.1f}, MA50: {ma_50:.1f}, MA200: {ma_200:.1f}, Regime: {current_regime}")
            
        else:
            # Default to manual - assume current period
            current_date = datetime.now().strftime('%Y-%m-%d')
            current_regime = 'sideways_market'  # Conservative default
            
            for regime_type, periods in self.regimes.get('manual', {}).items():
                for start, end in periods:
                    if start <= current_date <= end:
                        current_regime = regime_type
                        break
            
            logger.info(f"📅 Manual regime for {current_date}: {current_regime}")
        
        return current_regime
    
    def plot_regimes(self):
        """Plot market regimes analysis"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: VIX with regimes
        if self.vix_data is not None:
            ax1 = axes[0]
            ax1.plot(self.vix_data.index, self.vix_data.values, label='VIX/Volatility', color='black')
            ax1.axhline(self.vix_data.quantile(0.25), color='green', linestyle='--', alpha=0.7, label='25th percentile')
            ax1.axhline(self.vix_data.quantile(0.75), color='red', linestyle='--', alpha=0.7, label='75th percentile')
            ax1.set_title('VIX-Based Market Regimes')
            ax1.set_ylabel('VIX/Volatility')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: BIST with moving averages
        if self.bist_data is not None:
            ax2 = axes[1]
            ax2.plot(self.bist_data.index, self.bist_data.values, label='BIST100', color='blue')
            
            ma_50 = self.bist_data.rolling(50).mean()
            ma_200 = self.bist_data.rolling(200).mean()
            
            ax2.plot(ma_50.index, ma_50.values, label='MA50', color='orange', alpha=0.8)
            ax2.plot(ma_200.index, ma_200.values, label='MA200', color='red', alpha=0.8)
            ax2.set_title('Trend-Based Market Regimes')
            ax2.set_ylabel('BIST100 Index')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Regime summary
        ax3 = axes[2]
        regime_colors = {'bull_market': 'green', 'bear_market': 'red', 'sideways_market': 'gray'}
        
        y_pos = 0
        for method, regimes in self.regimes.items():
            for regime_type, periods in regimes.items():
                for start, end in periods:
                    start_date = pd.to_datetime(start)
                    end_date = pd.to_datetime(end)
                    ax3.barh(y_pos, (end_date - start_date).days, 
                            left=start_date, height=0.8,
                            color=regime_colors[regime_type], alpha=0.7,
                            label=f'{method}_{regime_type}' if periods.index((start, end)) == 0 else "")
            y_pos += 1
        
        ax3.set_title('Market Regime Timeline')
        ax3.set_xlabel('Date')
        ax3.set_yticks(range(len(self.regimes)))
        ax3.set_yticklabels(list(self.regimes.keys()))
        
        plt.tight_layout()
        plt.savefig('market_regimes_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_regimes(self):
        """Save regime analysis to JSON"""
        output = {
            'analysis_date': datetime.now().isoformat(),
            'current_regimes': {
                'vix_based': self.get_current_regime('vix_based'),
                'trend_based': self.get_current_regime('trend_based'),
                'manual': self.get_current_regime('manual')
            },
            'regimes': self.regimes,
            'vix_stats': {
                'current': float(self.vix_data.iloc[-1]) if self.vix_data is not None else None,
                'mean': float(self.vix_data.mean()) if self.vix_data is not None else None,
                'q25': float(self.vix_data.quantile(0.25)) if self.vix_data is not None else None,
                'q75': float(self.vix_data.quantile(0.75)) if self.vix_data is not None else None
            }
        }
        
        with open('market_regimes_analysis.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        logger.info("💾 Market regime analysis saved to: market_regimes_analysis.json")
    
    def run_full_analysis(self):
        """Run complete market regime analysis"""
        logger.info("🚀 Starting comprehensive market regime analysis...")
        
        # Load data
        self.load_market_data()
        
        # Analyze regimes using all methods
        self.analyze_vix_regimes()
        self.analyze_trend_regimes()
        self.define_manual_regimes()
        
        # Generate outputs
        self.plot_regimes()
        self.save_regimes()
        
        # Summary
        logger.info("\n📊 MARKET REGIME ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        current_regimes = {
            'VIX-based': self.get_current_regime('vix_based'),
            'Trend-based': self.get_current_regime('trend_based'),
            'Manual': self.get_current_regime('manual')
        }
        
        logger.info("🔍 CURRENT MARKET REGIME:")
        for method, regime in current_regimes.items():
            logger.info(f"   {method}: {regime}")
        
        # Find consensus
        regime_votes = list(current_regimes.values())
        consensus = max(set(regime_votes), key=regime_votes.count)
        confidence = regime_votes.count(consensus) / len(regime_votes) * 100
        
        logger.info(f"\n🎯 CONSENSUS: {consensus} ({confidence:.0f}% confidence)")
        logger.info("=" * 60)


def main():
    """Main function"""
    analyzer = MarketRegimeAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()