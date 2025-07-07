"""
Test AlgoLab Portfolio Functions
Test various API endpoints after successful authentication
"""
import sys
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.append(str(Path(__file__).parent))

from core.algolab_api import AlgoLabAPI
from config.settings import ALGOLAB_CONFIG


def test_portfolio_functions():
    """Test portfolio related functions"""
    print("\n" + "="*60)
    print("ALGOLAB PORTFOLIO TEST")
    print("="*60)
    
    # Initialize API with cached session
    api = AlgoLabAPI(
        ALGOLAB_CONFIG.get('api_key'),
        ALGOLAB_CONFIG.get('username'),
        ALGOLAB_CONFIG.get('password')
    )
    
    # Check if authenticated
    if not api.is_authenticated():
        print("❌ Not authenticated! Please run test_algolab_auth.py first")
        return
    
    print(f"✅ Using cached session (expires: {api.session_expires})")
    
    # Test functions
    tests = [
        ("Subaccounts", lambda: api.get_subaccounts()),
        ("Portfolio", lambda: api.get_portfolio()),
        ("Instant Position", lambda: api.get_instant_position()),
        ("Today's Transactions", lambda: api.get_todays_transactions()),
        ("Cash Flow", lambda: api.get_cash_flow()),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Testing: {test_name}")
        print('='*40)
        
        try:
            result = test_func()
            
            if result.get('success'):
                print(f"✅ {test_name} - Success")
                content = result.get('content', {})
                
                # Print some basic info
                if isinstance(content, list):
                    print(f"   Items count: {len(content)}")
                    if content and len(content) > 0:
                        print(f"   First item: {content[0]}")
                elif isinstance(content, dict):
                    print(f"   Keys: {list(content.keys())[:5]}...")
                    # Print some values
                    for key in list(content.keys())[:3]:
                        print(f"   {key}: {content.get(key)}")
                else:
                    print(f"   Content type: {type(content)}")
                    print(f"   Content: {str(content)[:100]}...")
            else:
                print(f"❌ {test_name} - Failed")
                print(f"   Error: {result.get('message')}")
                
        except Exception as e:
            print(f"❌ {test_name} - Exception")
            print(f"   Error: {str(e)}")
    
    # Test symbol info with a specific symbol
    print(f"\n{'='*40}")
    print("Testing: Symbol Info (GARAN)")
    print('='*40)
    
    try:
        result = api.get_symbol_info("GARAN")
        
        if result.get('success'):
            print("✅ Symbol Info - Success")
            content = result.get('content', {})
            print(f"   Symbol: {content.get('symbol')}")
            print(f"   Name: {content.get('name')}")
            print(f"   Last Price: {content.get('lastPrice')}")
            print(f"   Change: {content.get('change')}%")
        else:
            print("❌ Symbol Info - Failed")
            print(f"   Error: {result.get('message')}")
            
    except Exception as e:
        print("❌ Symbol Info - Exception")
        print(f"   Error: {str(e)}")


if __name__ == "__main__":
    test_portfolio_functions()