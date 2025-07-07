#!/usr/bin/env python3
"""
Refresh AlgoLab Session
Clears old session and creates a new one
"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from utils.algolab_auth import AlgoLabAuth

def refresh_session():
    """Clear old session and authenticate again"""
    print("AlgoLab Session Refresh Tool")
    print("="*50)
    
    # Remove old session file
    session_file = Path("data/cache/algolab_session.pkl")
    if session_file.exists():
        os.remove(session_file)
        print("✓ Old session removed")
    
    # Create new session
    print("\nCreating new session...")
    auth = AlgoLabAuth()
    api = auth.authenticate()
    
    if api:
        print("\n✅ New session created successfully!")
        print(f"Session expires: {api.session_expires}")
        
        # Test the connection
        print("\nTesting connection...")
        if auth.test_connection():
            print("✅ Connection verified!")
        else:
            print("❌ Connection test failed")
    else:
        print("\n❌ Failed to create new session")

if __name__ == "__main__":
    refresh_session()