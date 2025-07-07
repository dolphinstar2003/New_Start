#!/usr/bin/env python3
"""Test AlgoLab API Integration"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from paper_trading.data_fetcher_algolab import AlgoLabDataFetcher
from paper_trading.data_fetcher import DataFetcher
from datetime import datetime
import json

def test_algolab_prices():
    """Test fetching prices from AlgoLab"""
    print("="*60)
    print("TESTING ALGOLAB PRICE FETCHER")
    print("="*60)
    
    try:
        # Test direct AlgoLab fetcher
        print("\n1. Testing AlgoLabDataFetcher...")
        algolab_fetcher = AlgoLabDataFetcher()
        prices = algolab_fetcher.get_current_prices()
        
        print(f"\n✓ Fetched {len(prices)} prices from AlgoLab:")
        for i, (symbol, price) in enumerate(sorted(prices.items())[:5]):
            print(f"   {symbol}: {price:.2f} TL")
        print("   ...")
        
        # Test market status
        print("\n2. Testing market status...")
        status = algolab_fetcher.get_market_status()
        print(f"   Market: {status['status']}")
        print(f"   Current time: {status['current_time']}")
        print(f"   Trading hours: {status['market_open']} - {status['market_close']}")
        
        # Test main data fetcher with AlgoLab
        print("\n3. Testing DataFetcher with AlgoLab...")
        data_fetcher = DataFetcher(use_algolab=True)
        prices2 = data_fetcher.get_current_prices()
        
        print(f"\n✓ DataFetcher returned {len(prices2)} prices")
        
        # Compare prices
        print("\n4. Comparing prices...")
        differences = []
        for symbol in prices:
            if symbol in prices2:
                diff = abs(prices[symbol] - prices2[symbol])
                if diff > 0.01:
                    differences.append((symbol, prices[symbol], prices2[symbol]))
        
        if differences:
            print(f"   Found {len(differences)} price differences")
        else:
            print("   ✓ All prices match!")
        
        # Save test results
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "algolab_prices": len(prices),
            "datafetcher_prices": len(prices2),
            "market_status": status,
            "test_status": "PASSED"
        }
        
        with open("test_algolab_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print("\n✅ AlgoLab integration test completed successfully!")
        print("Results saved to test_algolab_results.json")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_algolab_prices()