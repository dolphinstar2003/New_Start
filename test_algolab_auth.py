"""
Test AlgoLab Authentication
Safe test script with clear instructions
"""
import sys
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

from utils.algolab_auth import AlgoLabAuth
from config.settings import ALGOLAB_CONFIG


def main():
    """Test AlgoLab authentication"""
    print("\n" + "="*60)
    print("ALGOLAB AUTHENTICATION TEST")
    print("="*60)
    print("\nIMPORTANT: You have only 2 SMS attempts!")
    print("Make sure you have your phone ready.")
    print("="*60)
    
    # Check current configuration
    print("\nChecking configuration...")
    print(f"API Key configured: {'Yes' if ALGOLAB_CONFIG.get('api_key') else 'No'}")
    print(f"Username configured: {'Yes' if ALGOLAB_CONFIG.get('username') else 'No'}")
    print(f"Password configured: {'Yes' if ALGOLAB_CONFIG.get('password') else 'No'}")
    
    # Confirm before proceeding
    print("\n" + "="*60)
    response = input("Do you want to proceed with authentication? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Authentication cancelled.")
        return
    
    # Initialize auth helper
    auth = AlgoLabAuth()
    
    # Perform authentication
    api = auth.authenticate()
    
    if api:
        print("\n✅ Authentication successful!")
        print(f"Session expires: {api.session_expires}")
        
        # Optional: Test API with a simple request
        test_api = input("\nDo you want to test the API connection? (yes/no): ").strip().lower()
        
        if test_api == 'yes':
            if auth.test_connection():
                print("✅ API test successful!")
                
                # Optional: Get portfolio info
                show_portfolio = input("\nShow portfolio summary? (yes/no): ").strip().lower()
                
                if show_portfolio == 'yes':
                    try:
                        portfolio = api.get_portfolio()
                        if portfolio.get('success'):
                            print("\nPortfolio Summary:")
                            print("-" * 40)
                            content = portfolio.get('content', {})
                            print(f"Account: {content.get('accountNumber', 'N/A')}")
                            print(f"Total Value: {content.get('totalValue', 'N/A')}")
                            print("-" * 40)
                        else:
                            print(f"Failed to get portfolio: {portfolio.get('message')}")
                    except Exception as e:
                        print(f"Error getting portfolio: {e}")
    else:
        print("\n❌ Authentication failed!")
        print("Please check your credentials and try again.")


if __name__ == "__main__":
    main()