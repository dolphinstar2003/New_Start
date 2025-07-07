#!/usr/bin/env python3
"""Test different API endpoints for getting symbol prices"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.algolab_api import AlgoLabAPI
from utils.algolab_auth import AlgoLabAuth
from config.settings import SACRED_SYMBOLS

def test_endpoints():
    """Test various endpoints"""
    auth = AlgoLabAuth()
    api = auth.authenticate()
    
    if not api:
        print("Authentication failed!")
        return
    
    print("Testing different endpoints for price data...\n")
    
    # Test endpoints
    endpoints = [
        "/api/GetEquity",
        "/api/GetSymbol", 
        "/api/GetPrice",
        "/api/GetLast",
        "/api/GetTick",
        "/api/GetQuote",
        "/api/GetMarketData",
        "/api/GetRealtime",
        "/api/GetSnapshot"
    ]
    
    test_symbol = "GARAN"
    
    for endpoint in endpoints:
        print(f"Testing {endpoint}...")
        try:
            response = api._request("POST", endpoint, {"Symbol": test_symbol})
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {response.text[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    # Test candles endpoint with different payloads
    print("\nTesting candle endpoints...")
    candle_tests = [
        {"Symbol": test_symbol, "Period": "1", "BarCount": 5},
        {"Symbol": test_symbol, "Period": "D", "BarCount": 5},
        {"Symbol": test_symbol, "Interval": "1", "Count": 5},
        {"Symbol": test_symbol, "TimeFrame": "1", "Limit": 5}
    ]
    
    for i, payload in enumerate(candle_tests):
        print(f"Test {i+1}: {payload}")
        try:
            response = api._request("POST", "/api/GetCandles", payload)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  Response: {response.text[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
        print()

if __name__ == "__main__":
    test_endpoints()