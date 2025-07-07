#!/usr/bin/env python3
"""
Test Market Data Access
Verify AlgoLab API market data functionality
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent))

from utils.algolab_auth import AlgoLabAuth
from config.settings import SACRED_SYMBOLS


def test_market_data():
    """Test market data access via API"""
    print("\n" + "="*50)
    print("MARKET DATA TEST")
    print("="*50)
    
    # Authenticate
    auth = AlgoLabAuth()
    api = auth.authenticate()
    
    if not api:
        print("❌ Authentication failed")
        return
    
    print("\n✅ Authentication successful")
    print("\nTesting market data access for sacred symbols...")
    print("-"*50)
    
    success_count = 0
    
    # Test first 5 symbols
    for symbol in SACRED_SYMBOLS[:5]:
        clean_symbol = symbol.replace('.IS', '')
        print(f"\n{symbol}:")
        
        try:
            # Get symbol info
            result = api.get_symbol_info(clean_symbol)
            
            if result.get('success') and result.get('content'):
                data = result['content']
                
                print(f"  Last Price: {data.get('lastPrice', 'N/A')}")
                print(f"  Open: {data.get('open', 'N/A')}")
                print(f"  High: {data.get('high', 'N/A')}")
                print(f"  Low: {data.get('low', 'N/A')}")
                print(f"  Volume: {data.get('volume', 'N/A')}")
                print(f"  Change: {data.get('change', 'N/A')}%")
                
                success_count += 1
            else:
                print(f"  ❌ Failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "-"*50)
    print(f"Success rate: {success_count}/5 symbols")
    
    # Test portfolio
    print("\n" + "="*50)
    print("PORTFOLIO TEST")
    print("="*50)
    
    try:
        portfolio = api.get_portfolio()
        
        if portfolio.get('success'):
            print("✅ Portfolio access successful")
            
            # Get cash flow
            cash_flow = api.get_cash_flow()
            if cash_flow.get('success') and cash_flow.get('content'):
                content = cash_flow['content']
                print(f"\nCash available:")
                print(f"  T+0: {content.get('t0', 'N/A')}")
                print(f"  T+1: {content.get('t1', 'N/A')}")
                print(f"  T+2: {content.get('t2', 'N/A')}")
        else:
            print(f"❌ Portfolio access failed: {portfolio.get('message')}")
            
    except Exception as e:
        print(f"❌ Portfolio error: {e}")
    
    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)


if __name__ == "__main__":
    # Check market hours
    current_time = datetime.now()
    if not (10 <= current_time.hour < 18 and current_time.weekday() < 5):
        print("\n⚠️  Market is closed. Data may not be current.")
    
    test_market_data()