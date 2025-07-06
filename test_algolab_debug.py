"""
AlgoLab Debug Test
Detailed debugging for authentication issues
"""
import sys
from pathlib import Path
from loguru import logger

# Configure detailed logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("logs/algolab_debug.log", level="DEBUG", rotation="10 MB")

# Add project to path
sys.path.append(str(Path(__file__).parent))

from utils.algolab_auth import AlgoLabAuth
from config.settings import ALGOLAB_CONFIG


def main():
    """Debug AlgoLab authentication"""
    print("\n" + "="*60)
    print("ALGOLAB DEBUG TEST")
    print("="*60)
    print("This will show detailed debug information")
    print("Check logs/algolab_debug.log for full details")
    print("="*60)
    
    # Initialize auth helper
    auth = AlgoLabAuth()
    
    # Get API details
    print(f"\nAPI Key prefix: {auth.api_key[:20]}..." if auth.api_key else "No API key")
    print(f"Username: {auth.username}" if auth.username else "No username")
    
    # Confirm
    response = input("\nProceed with debug authentication? (yes/no): ").strip().lower()
    if response != 'yes':
        return
    
    # Try authentication
    try:
        api = auth.authenticate()
        if api:
            print("\n✅ Authentication successful!")
        else:
            print("\n❌ Authentication failed!")
            print("Check logs/algolab_debug.log for details")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Authentication error")


if __name__ == "__main__":
    main()