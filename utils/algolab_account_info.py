"""
AlgoLab Account Information Utility
Get and display account information
"""
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from core.algolab_api import AlgoLabAPI
from config.settings import ALGOLAB_CONFIG
from loguru import logger


class AlgoLabAccountInfo:
    """Utility class for getting account information"""
    
    def __init__(self):
        """Initialize with API"""
        self.api = AlgoLabAPI(
            ALGOLAB_CONFIG.get('api_key'),
            ALGOLAB_CONFIG.get('username'),
            ALGOLAB_CONFIG.get('password')
        )
        
        if not self.api.is_authenticated():
            raise ValueError("Not authenticated. Please run authentication first.")
    
    def get_account_summary(self):
        """Get comprehensive account summary"""
        print("\n" + "="*60)
        print("ALGOLAB ACCOUNT SUMMARY")
        print("="*60)
        print(f"Session expires: {self.api.session_expires}")
        print("="*60)
        
        # 1. Get Subaccounts
        print("\n📁 SUBACCOUNTS:")
        print("-"*40)
        try:
            subaccounts = self.api.get_subaccounts()
            if subaccounts.get('success'):
                content = subaccounts.get('content', [])
                if content:
                    for acc in content:
                        print(f"  • {acc}")
                else:
                    print("  No subaccounts found")
            else:
                print(f"  Error: {subaccounts.get('message')}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 2. Get Instant Position
        print("\n💼 INSTANT POSITION:")
        print("-"*40)
        try:
            position = self.api.get_instant_position()
            if position.get('success'):
                content = position.get('content', [])
                for item in content:
                    if item.get('code') != '-':
                        print(f"  Symbol: {item.get('code')}")
                        print(f"  Quantity: {item.get('totalstock')}")
                        print(f"  Cost: {item.get('cost')}")
                        print(f"  Value: {item.get('totalamount')}")
                        print(f"  P&L: {item.get('profit')}")
                        print("  " + "-"*20)
                    elif item.get('explanation') == 'total':
                        print(f"  💰 Total Cost: {item.get('cost')}")
                        print(f"  💰 Total Value: {item.get('totalamount')}")
            else:
                print(f"  Error: {position.get('message')}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 3. Cash Flow
        print("\n💵 CASH FLOW:")
        print("-"*40)
        try:
            cash_flow = self.api.get_cash_flow()
            if cash_flow.get('success'):
                content = cash_flow.get('content', {})
                print(f"  T+0: {content.get('t0', 0)}")
                print(f"  T+1: {content.get('t1', 0)}")
                print(f"  T+2: {content.get('t2', 0)}")
            else:
                print(f"  Error: {cash_flow.get('message')}")
        except Exception as e:
            print(f"  Error: {e}")
        
        # 4. Today's Transactions
        print("\n📊 TODAY'S TRANSACTIONS:")
        print("-"*40)
        try:
            transactions = self.api.get_todays_transactions()
            if transactions.get('success'):
                content = transactions.get('content', [])
                if content:
                    for tx in content[:5]:  # Show first 5
                        print(f"  • {tx}")
                    if len(content) > 5:
                        print(f"  ... and {len(content) - 5} more")
                else:
                    print("  No transactions today")
            else:
                print(f"  Error: {transactions.get('message')}")
        except Exception as e:
            print(f"  Error: {e}")
    
    def save_account_data(self, filename: str = None):
        """Save account data to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/analysis/account_data_{timestamp}.json"
        
        data = {
            "timestamp": str(datetime.now()),
            "session_expires": str(self.api.session_expires),
            "data": {}
        }
        
        # Collect all data
        try:
            data["data"]["subaccounts"] = self.api.get_subaccounts()
        except:
            data["data"]["subaccounts"] = {"error": "Failed to fetch"}
        
        try:
            data["data"]["instant_position"] = self.api.get_instant_position()
        except:
            data["data"]["instant_position"] = {"error": "Failed to fetch"}
        
        try:
            data["data"]["cash_flow"] = self.api.get_cash_flow()
        except:
            data["data"]["cash_flow"] = {"error": "Failed to fetch"}
        
        try:
            data["data"]["todays_transactions"] = self.api.get_todays_transactions()
        except:
            data["data"]["todays_transactions"] = {"error": "Failed to fetch"}
        
        # Save to file
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Account data saved to: {filepath}")
        return filepath


def main():
    """Main function"""
    try:
        account_info = AlgoLabAccountInfo()
        
        # Show summary
        account_info.get_account_summary()
        
        # Ask to save
        save = input("\n\nSave account data to file? (yes/no): ").strip().lower()
        if save == 'yes':
            account_info.save_account_data()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Account info error")


if __name__ == "__main__":
    main()