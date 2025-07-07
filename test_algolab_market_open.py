#!/usr/bin/env python3
"""Test AlgoLab during market hours"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.algolab_auth import AlgoLabAuth
from datetime import datetime

print("Testing AlgoLab API during market hours...")
print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")
print("Market hours: 10:00 - 18:00")
print("="*50)

# Test authentication
auth = AlgoLabAuth()
api = auth.authenticate()

if api:
    print("\n✓ Authentication successful")
    print(f"Session expires: {api.session_expires}")
    
    # Test single symbol
    print("\nTesting single symbol (GARAN)...")
    try:
        result = api.get_symbol_info("GARAN")
        print(f"Result: {result}")
        
        if result:
            print(f"\nParsed data:")
            print(f"  Name: {result.get('name')}")
            print(f"  Last: {result.get('lst')}")
            print(f"  Bid: {result.get('bid')}")
            print(f"  Ask: {result.get('ask')}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Authentication failed!")