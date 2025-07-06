"""
AlgoLab Authentication Helper
Safe SMS authentication management
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import getpass
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.algolab_api import AlgoLabAPI
from config.settings import ALGOLAB_CONFIG


class AlgoLabAuth:
    """Helper class for AlgoLab authentication"""
    
    def __init__(self):
        """Initialize authentication helper"""
        self.api = None
        self.max_sms_attempts = 2  # Maximum SMS attempts
        self.current_attempts = 0
        
        # Load credentials from environment
        self.api_key = ALGOLAB_CONFIG.get('api_key')
        self.username = ALGOLAB_CONFIG.get('username')
        self.password = ALGOLAB_CONFIG.get('password')
        
        logger.info("AlgoLab Auth helper initialized")
    
    def check_credentials(self) -> bool:
        """Check if credentials are available"""
        if not self.api_key:
            logger.error("API key not found in environment")
            return False
        
        if not self.username:
            logger.error("Username not found in environment")
            return False
        
        if not self.password:
            logger.error("Password not found in environment")
            return False
        
        return True
    
    def get_credentials_interactive(self):
        """Get credentials interactively if not in environment"""
        if not self.api_key:
            self.api_key = input("Enter AlgoLab API Key: ").strip()
        
        if not self.username:
            self.username = input("Enter Username/TC: ").strip()
        
        if not self.password:
            self.password = getpass.getpass("Enter Password: ")
        
        # Save to .env file for future use
        self._save_credentials()
    
    def _save_credentials(self):
        """Save credentials to .env file"""
        env_file = Path(".env")
        
        # Read existing content
        existing_content = ""
        if env_file.exists():
            with open(env_file, 'r') as f:
                existing_content = f.read()
        
        # Update credentials
        lines = existing_content.split('\n')
        updated_lines = []
        
        cred_dict = {
            'ALGOLAB_API_KEY': self.api_key,
            'ALGOLAB_USERNAME': self.username,
            'ALGOLAB_PASSWORD': self.password
        }
        
        for line in lines:
            if '=' in line:
                key = line.split('=')[0].strip()
                if key not in cred_dict:
                    updated_lines.append(line)
        
        # Add new credentials
        for key, value in cred_dict.items():
            if value:
                updated_lines.append(f"{key}={value}")
        
        # Write back
        with open(env_file, 'w') as f:
            f.write('\n'.join(updated_lines))
        
        logger.info("Credentials saved to .env file")
    
    def authenticate(self) -> Optional[AlgoLabAPI]:
        """
        Perform authentication with AlgoLab
        
        Returns:
            AlgoLabAPI instance if successful, None otherwise
        """
        # Check or get credentials
        if not self.check_credentials():
            self.get_credentials_interactive()
        
        # Initialize API
        self.api = AlgoLabAPI(self.api_key, self.username, self.password)
        
        # Check if we have cached session
        if self.api.is_authenticated():
            logger.info("Using cached session")
            return self.api
        
        # Perform login
        logger.info("Starting new login process")
        print("\n" + "="*50)
        print("ALGOLAB AUTHENTICATION")
        print("="*50)
        print("SMS will be sent to your registered phone number")
        print("You have 2 attempts to enter the correct SMS code")
        print("="*50 + "\n")
        
        try:
            # Step 1: Login and send SMS
            login_result = self.api.login_user()
            
            if not login_result.get('success'):
                logger.error(f"Login failed: {login_result.get('message')}")
                return None
            
            print("✓ Login successful - SMS sent to your phone")
            
            # Step 2: Get SMS code and verify
            while self.current_attempts < self.max_sms_attempts:
                self.current_attempts += 1
                print(f"\nAttempt {self.current_attempts}/{self.max_sms_attempts}")
                
                sms_code = input("Enter SMS code: ").strip()
                
                if not sms_code:
                    print("SMS code cannot be empty")
                    continue
                
                # Verify SMS
                verify_result = self.api.login_user_control(sms_code)
                
                if verify_result.get('success'):
                    print("✓ Authentication successful!")
                    print(f"✓ Session valid until: {self.api.session_expires}")
                    return self.api
                else:
                    error_msg = verify_result.get('message', 'Unknown error')
                    print(f"✗ Verification failed: {error_msg}")
                    
                    if self.current_attempts < self.max_sms_attempts:
                        print("Please try again...")
                    else:
                        print("Maximum attempts reached!")
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            print(f"Error during authentication: {e}")
        
        return None
    
    def test_connection(self) -> bool:
        """Test API connection with portfolio request"""
        if not self.api or not self.api.is_authenticated():
            logger.error("Not authenticated")
            return False
        
        try:
            # Test with portfolio request
            result = self.api.get_portfolio()
            
            if result.get('success'):
                logger.info("Connection test successful")
                print("\n✓ API connection verified")
                return True
            else:
                logger.error(f"Connection test failed: {result.get('message')}")
                return False
                
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            return False


def main():
    """Main authentication flow"""
    auth = AlgoLabAuth()
    
    # Authenticate
    api = auth.authenticate()
    
    if api:
        print("\n" + "="*50)
        print("AUTHENTICATION SUCCESSFUL")
        print("="*50)
        
        # Test connection
        if auth.test_connection():
            print("API is ready to use!")
            print(f"Session expires: {api.session_expires}")
        else:
            print("Connection test failed")
    else:
        print("\nAuthentication failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()