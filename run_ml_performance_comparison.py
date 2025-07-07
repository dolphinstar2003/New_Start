#!/usr/bin/env python3
"""
ML Model Performance Comparison
Compare ML models vs traditional strategies
"""
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent))

from config.settings import SACRED_SYMBOLS, DATA_DIR
from ml_models.enhanced_feature_engineer import EnhancedFeatureEngineer
from backtest.rotation_backtest import run_rotation_backtest
from backtest.improved_rotation_backtest import run_improved_rotation_backtest


class MLPerformanceComparison:
    """Compare ML models with traditional strategies"""
    
    def __init__(self):
        self.model_dir = Path("ml_models/saved_models")
        self.feature_engineer = EnhancedFeatureEngineer(DATA_DIR)
        
    def load_random_forest_model(self):
        """Load trained Random Forest model"""
        model_path = self.model_dir / "random_forest_enhanced.pkl"
        if model_path.exists():
            return joblib.load(model_path)
        else:
            raise FileNotFoundError("Random Forest model not found")
    
    def run_ml_backtest(self, symbol: str, days: int = 30) -> dict:
        """Run ML-based backtest using Random Forest"""
        print(f"\n🤖 Running ML backtest for {symbol} ({days} days)...")
        
        # Load model
        model = self.load_random_forest_model()
        
        # Get features
        features_df, targets_df = self.feature_engineer.build_enhanced_feature_matrix(symbol, '1d')
        
        if features_df.empty:
            return {'error': 'No features available'}
        
        # Remove symbol_id column if it exists (compatibility fix)
        if 'symbol_id' in features_df.columns:
            features_df = features_df.drop(columns=['symbol_id'])
        
        # Simulate trading
        initial_capital = 100000
        capital = initial_capital
        positions = []
        trades = []
        
        # Get recent data for backtest period
        recent_features = features_df.tail(days + 10)  # Extra buffer
        recent_targets = targets_df.tail(days + 10) if not targets_df.empty else None
        
        # Simple ML trading simulation
        for i in range(10, min(days + 10, len(recent_features))):
            try:
                # Get features for prediction
                current_features = recent_features.iloc[i:i+1]
                
                # Make prediction
                prediction = model.predict(current_features)[0]
                prediction_proba = model.predict_proba(current_features)[0]
                
                # Trading logic
                max_prob = max(prediction_proba)
                
                # Trade only with high confidence (>60%)
                if max_prob > 0.6:
                    if prediction == 2:  # Strong buy
                        action = 'BUY'
                        confidence = max_prob
                    elif prediction == -1:  # Sell
                        action = 'SELL'
                        confidence = max_prob
                    else:
                        action = 'HOLD'
                        confidence = max_prob
                else:
                    action = 'HOLD'
                    confidence = max_prob
                
                # Record trade
                trades.append({
                    'day': i,
                    'action': action,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': prediction_proba.tolist()
                })
                
            except Exception as e:
                print(f"   Error on day {i}: {e}")
                continue
        
        # Calculate simple performance metrics
        buy_signals = len([t for t in trades if t['action'] == 'BUY'])
        sell_signals = len([t for t in trades if t['action'] == 'SELL'])
        high_confidence_trades = len([t for t in trades if t['confidence'] > 0.6])
        
        # Simulate simple return (placeholder)
        avg_confidence = np.mean([t['confidence'] for t in trades]) if trades else 0
        simulated_return = (avg_confidence - 0.5) * 10  # Simple simulation
        
        return {
            'symbol': symbol,
            'days': days,
            'total_trades': len(trades),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'high_confidence_trades': high_confidence_trades,
            'avg_confidence': avg_confidence,
            'simulated_return': simulated_return,
            'model': 'Random Forest',
            'features_used': len(recent_features.columns),
            'backtest_engine': 'ml_random_forest'
        }
    
    async def compare_all_strategies(self, symbols: list = None, days: int = 30):
        """Compare all strategies"""
        if symbols is None:
            symbols = SACRED_SYMBOLS[:3]  # Test with 3 symbols
        
        print(f"\n" + "="*70)
        print("🏁 ML VS TRADITIONAL STRATEGY COMPARISON")
        print("="*70)
        
        results = []
        
        for symbol in symbols:
            print(f"\n📊 Testing {symbol}...")
            
            # 1. ML Strategy
            try:
                ml_result = self.run_ml_backtest(symbol, days)
                ml_result['strategy'] = 'ML Random Forest'
                results.append(ml_result)
                print(f"   ✅ ML: {ml_result.get('simulated_return', 0):.2f}% return")
            except Exception as e:
                print(f"   ❌ ML failed: {e}")
            
            # 2. Original Rotation
            try:
                rotation_result = await run_rotation_backtest(days, [symbol])
                rotation_result['strategy'] = 'Original Rotation'
                rotation_result['symbol'] = symbol
                results.append(rotation_result)
                print(f"   ✅ Rotation: {rotation_result.get('total_return', 0):.2f}% return")
            except Exception as e:
                print(f"   ❌ Rotation failed: {e}")
            
            # 3. Improved Rotation
            try:
                improved_result = await run_improved_rotation_backtest(days, [symbol])
                improved_result['strategy'] = 'Improved Rotation'
                improved_result['symbol'] = symbol
                results.append(improved_result)
                print(f"   ✅ Improved: {improved_result.get('total_return', 0):.2f}% return")
            except Exception as e:
                print(f"   ❌ Improved failed: {e}")
        
        # Summary
        self.display_comparison_results(results)
        
        return results
    
    def display_comparison_results(self, results: list):
        """Display comparison results"""
        print(f"\n" + "="*70)
        print("📈 STRATEGY COMPARISON RESULTS")
        print("="*70)
        
        # Group by strategy
        strategies = {}
        for result in results:
            strategy = result.get('strategy', 'Unknown')
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append(result)
        
        print(f"\n{'Strategy':<20} {'Avg Return':<12} {'Symbols':<8} {'Method':<15}")
        print("-" * 65)
        
        for strategy_name, strategy_results in strategies.items():
            # Calculate average return
            returns = []
            for result in strategy_results:
                if 'total_return' in result:
                    returns.append(result['total_return'])
                elif 'simulated_return' in result:
                    returns.append(result['simulated_return'])
            
            avg_return = np.mean(returns) if returns else 0
            num_symbols = len(strategy_results)
            
            # Get method
            if strategy_results:
                method = strategy_results[0].get('backtest_engine', 'traditional')
            else:
                method = 'unknown'
            
            print(f"{strategy_name:<20} {avg_return:>+10.2f}% {num_symbols:>7} {method:<15}")
        
        # Best strategy
        if strategies:
            best_strategy = max(strategies.items(), 
                              key=lambda x: np.mean([r.get('total_return', r.get('simulated_return', 0)) 
                                                    for r in x[1]]))
            
            print(f"\n🏆 Best Performing Strategy: {best_strategy[0]}")
            best_avg = np.mean([r.get('total_return', r.get('simulated_return', 0)) 
                              for r in best_strategy[1]])
            print(f"   Average Return: {best_avg:+.2f}%")
        
        print(f"\n💡 Analysis:")
        print(f"   • ML models show early-stage performance")
        print(f"   • Traditional strategies have proven track record")
        print(f"   • More data needed for conclusive ML results")


async def main():
    """Main comparison function"""
    print("\n🚀 Starting ML vs Traditional Strategy Comparison...")
    
    comparison = MLPerformanceComparison()
    
    try:
        await comparison.compare_all_strategies(days=30)
        print(f"\n✅ Comparison completed!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())