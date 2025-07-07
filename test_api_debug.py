#!/usr/bin/env python3
"""Debug AlgoLab API connection"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.algolab_api import AlgoLabAPI
from utils.algolab_auth import AlgoLabAuth
from loguru import logger

# Enable debug logging
logger.add("debug_api.log", level="DEBUG")

def test_api():
    """Test API connection step by step"""
    print("Starting API debug test...")
    
    # Initialize auth
    auth = AlgoLabAuth()
    api = auth.authenticate()
    
    if not api:
        print("Authentication failed!")
        return
    
    print(f"\n✓ Authenticated successfully")
    print(f"Hash: {api.hash[:20]}...")
    print(f"Session expires: {api.session_expires}")
    
    # Test 1: Portfolio
    print("\n1. Testing Portfolio API...")
    try:
        result = api.get_portfolio()
        print(f"Portfolio response: {result}")
    except Exception as e:
        print(f"Portfolio error: {e}")
    
    # Test 2: Symbol Info
    print("\n2. Testing Symbol Info API...")
    try:
        result = api.get_symbol_info("GARAN")
        print(f"Symbol info response: {result}")
    except Exception as e:
        print(f"Symbol info error: {e}")
    
    # Test 3: Direct request test
    print("\n3. Testing direct request...")
    try:
        # Test with a simple endpoint
        response = api._request("POST", "/api/GetSubAccounts", {})
        print(f"Direct request status: {response.status_code}")
        print(f"Direct request response: {response.text[:200]}")
    except Exception as e:
        print(f"Direct request error: {e}")
    
    print("\nDebug test completed - check debug_api.log for details")

if __name__ == "__main__":
    test_api()