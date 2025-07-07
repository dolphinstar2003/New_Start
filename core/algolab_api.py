"""
AlgoLab API Wrapper
SMS authentication için dikkatli kullanılmalı - 2 giriş hakkı var!
"""
import requests
import hashlib
import json
import time
import base64
from typing import Dict, Any, Optional
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from loguru import logger

# Configure logger
logger.add("logs/algolab_api.log", rotation="1 day", retention="7 days")


class AlgoLabAPI:
    """AlgoLab API wrapper with SMS authentication support"""
    
    def __init__(self, api_key: str, username: str = None, password: str = None):
        """
        Initialize AlgoLab API client
        
        Args:
            api_key: API key from AlgoLab
            username: TC/Username for Denizbank
            password: Internet banking password
        """
        # API configuration
        self.hostname = "www.algolab.com.tr"
        self.api_hostname = f"https://{self.hostname}"
        self.api_url = f"{self.api_hostname}/api"
        
        # Extract API code from key
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        
        self.api_key = f"API-{self.api_code}"
        self.username = username
        self.password = password
        
        # Session management
        self.token = None
        self.hash = None
        self.session_expires = None
        self.last_request = 0.0
        self.LOCK = False
        
        # Cache directory for session
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.session_cache_file = self.cache_dir / "algolab_session.pkl"
        
        # Try to load existing session
        self._load_session()
        
        logger.info("AlgoLab API initialized")
    
    def _encrypt(self, text: str) -> str:
        """
        Encrypt text using AES with API code as key
        
        Args:
            text: Text to encrypt
            
        Returns:
            Base64 encoded encrypted text
        """
        iv = b'\0' * 16
        key = base64.b64decode(self.api_code.encode('utf-8'))
        cipher = AES.new(key, AES.MODE_CBC, iv)
        bytes_data = text.encode()
        padded_bytes = pad(bytes_data, 16)
        encrypted = cipher.encrypt(padded_bytes)
        return base64.b64encode(encrypted).decode("utf-8")
    
    def _make_checker(self, endpoint: str, payload: Dict) -> str:
        """
        Create checker hash for request validation
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            SHA256 hash hex string
        """
        body = json.dumps(payload).replace(' ', '') if payload else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _request(self, method: str, endpoint: str, payload: Dict = None, 
                 headers: Dict = None, login: bool = False) -> requests.Response:
        """
        Make HTTP request with rate limiting
        
        Args:
            method: HTTP method (GET/POST)
            endpoint: API endpoint
            payload: Request payload
            headers: Request headers
            login: Whether this is a login request
            
        Returns:
            Response object
        """
        # Wait for lock
        while self.LOCK:
            time.sleep(0.1)
        
        self.LOCK = True
        try:
            # Rate limiting - 5 seconds between requests
            current_time = time.time()
            time_diff = current_time - self.last_request
            
            if self.last_request > 0 and time_diff < 5.0:
                wait_time = 5.0 - time_diff + 0.1
                logger.info(f"Rate limiting: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
            
            # Prepare headers
            if headers is None:
                headers = {"APIKEY": self.api_key}
            
            # Add authorization for non-login requests
            if not login and self.hash:
                headers["Authorization"] = self.hash
                headers["Checker"] = self._make_checker(endpoint, payload or {})
            
            headers["Content-Type"] = "application/json"
            
            # Make request
            url = self.api_url + endpoint
            
            if method == "POST":
                response = requests.post(url, json=payload, headers=headers)
            else:
                response = requests.get(url, params=payload, headers=headers)
            
            self.last_request = time.time()
            
            # Log request
            logger.debug(f"{method} {endpoint} - Status: {response.status_code}")
            
            return response
            
        finally:
            self.LOCK = False
    
    def _save_session(self):
        """Save session data to cache"""
        if self.hash and self.session_expires:
            session_data = {
                'hash': self.hash,
                'expires': self.session_expires,
                'saved_at': datetime.now()
            }
            with open(self.session_cache_file, 'wb') as f:
                pickle.dump(session_data, f)
            logger.info("Session saved to cache")
    
    def _load_session(self):
        """Load session from cache if valid"""
        if self.session_cache_file.exists():
            try:
                with open(self.session_cache_file, 'rb') as f:
                    session_data = pickle.load(f)
                
                # Check if session is still valid
                if session_data['expires'] > datetime.now():
                    self.hash = session_data['hash']
                    self.session_expires = session_data['expires']
                    logger.info("Session loaded from cache")
                    return True
                else:
                    logger.info("Cached session expired")
                    
            except Exception as e:
                logger.error(f"Failed to load session: {e}")
        
        return False
    
    def login_user(self) -> Dict[str, Any]:
        """
        First step of login - sends SMS code
        
        Returns:
            Response with token for SMS verification
        """
        if not self.username or not self.password:
            raise ValueError("Username and password required for login")
        
        logger.info("Starting login process - SMS will be sent")
        
        try:
            # Encrypt credentials
            encrypted_username = self._encrypt(self.username)
            encrypted_password = self._encrypt(self.password)
            
            payload = {
                "Username": encrypted_username,
                "Password": encrypted_password
            }
            
            response = self._request("POST", "/api/LoginUser", payload, login=True)
            result = response.json()
            
            if result.get('success'):
                self.token = result['content']['token']
                logger.info("Login successful - SMS sent to registered phone")
                return result
            else:
                logger.error(f"Login failed: {result.get('message')}")
                return result
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            raise
    
    def login_user_control(self, sms_code: str) -> Dict[str, Any]:
        """
        Second step of login - verify SMS code
        
        Args:
            sms_code: SMS verification code
            
        Returns:
            Response with hash for session
        """
        if not self.token:
            raise ValueError("Must call login_user first to get token")
        
        logger.info(f"Verifying SMS code: {sms_code}")
        logger.debug(f"Token before encryption: {self.token[:20]}...")
        
        try:
            # Encrypt BOTH token and SMS code according to AlgoLab example
            encrypted_token = self._encrypt(self.token)
            encrypted_sms = self._encrypt(sms_code)
            
            logger.debug(f"Encrypted token: {encrypted_token[:20]}...")
            logger.debug(f"Encrypted SMS: {encrypted_sms}")
            
            # Note: field name is lowercase 'password' not 'Password'
            payload = {
                "token": encrypted_token,
                "password": encrypted_sms
            }
            
            logger.debug(f"Payload: {payload}")
            
            response = self._request("POST", "/api/LoginUserControl", payload, login=True)
            result = response.json()
            
            logger.debug(f"Response: {result}")
            
            if result.get('success'):
                self.hash = result['content']['hash']
                # Session valid for 24 hours
                self.session_expires = datetime.now() + timedelta(hours=24)
                self._save_session()
                logger.info("SMS verification successful - session created")
                return result
            else:
                logger.error(f"SMS verification failed: {result.get('message')}")
                return result
                
        except Exception as e:
            logger.error(f"SMS verification error: {e}", exc_info=True)
            raise
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        return bool(self.hash and self.session_expires and self.session_expires > datetime.now())
    
    def get_portfolio(self, subaccount: str = "") -> Dict[str, Any]:
        """
        Get portfolio information using InstantPosition endpoint
        
        Args:
            subaccount: Subaccount name (optional)
            
        Returns:
            Portfolio data
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        # Use InstantPosition as Portfolio endpoint doesn't exist
        return self.get_instant_position(subaccount)
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get symbol information
        
        Args:
            symbol: Symbol code (e.g., 'GARAN')
            
        Returns:
            Symbol information
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        payload = {"symbol": symbol}  # Lowercase as per documentation
        response = self._request("POST", "/api/GetEquityInfo", payload)
        
        try:
            result = response.json()
            if response.status_code == 200 and result.get('success'):
                return result.get('content', {})
            else:
                logger.error(f"Failed to get symbol info: {result.get('message', 'Unknown error')}")
                return {}
        except Exception as e:
            logger.error(f"Error parsing symbol info response: {e}")
            return {}
    
    def get_subaccounts(self) -> Dict[str, Any]:
        """
        Get subaccount information
        
        Returns:
            Subaccount list
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        response = self._request("POST", "/api/GetSubAccounts", {})
        return response.json()
    
    def get_instant_position(self, subaccount: str = "") -> Dict[str, Any]:
        """
        Get instant position information
        
        Args:
            subaccount: Subaccount name (optional)
            
        Returns:
            Instant position data
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        payload = {"Subaccount": subaccount}
        response = self._request("POST", "/api/InstantPosition", payload)
        return response.json()
    
    def get_todays_transactions(self, subaccount: str = "") -> Dict[str, Any]:
        """
        Get today's transactions
        
        Args:
            subaccount: Subaccount name (optional)
            
        Returns:
            Today's transactions
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        payload = {"Subaccount": subaccount}
        response = self._request("POST", "/api/TodaysTransaction", payload)
        return response.json()
    
    def get_cash_flow(self, subaccount: str = "") -> Dict[str, Any]:
        """
        Get cash flow information
        
        Args:
            subaccount: Subaccount name (optional)
            
        Returns:
            Cash flow data
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        payload = {"Subaccount": subaccount}
        response = self._request("POST", "/api/CashFlow", payload)
        return response.json()
    
    def get_account_extre(self, subaccount: str = "", start_date: str = "", end_date: str = "") -> Dict[str, Any]:
        """
        Get account statement/extract
        
        Args:
            subaccount: Subaccount name (optional)
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            Account statement data
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        payload = {
            "Subaccount": subaccount,
            "StartDate": start_date,
            "EndDate": end_date
        }
        response = self._request("POST", "/api/AccountExtre", payload)
        return response.json()
    
    def send_order(self, symbol: str, direction: str, price: float, 
                   quantity: int, order_type: str = "limit", 
                   subaccount: str = "") -> Dict[str, Any]:
        """
        Send trading order
        
        Args:
            symbol: Symbol code
            direction: 'buy' or 'sell'
            price: Order price
            quantity: Order quantity
            order_type: 'limit' or 'market'
            subaccount: Subaccount name (optional)
            
        Returns:
            Order response
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        payload = {
            "Symbol": symbol,
            "Direction": direction.upper(),
            "Price": price,
            "Quantity": quantity,
            "OrderType": order_type.upper(),
            "Subaccount": subaccount
        }
        
        response = self._request("POST", "/api/SendOrder", payload)
        return response.json()
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from subaccounts
        
        Returns:
            Account information
        """
        if not self.is_authenticated():
            raise ValueError("Not authenticated. Please login first.")
        
        try:
            response = self._request("POST", "/api/GetSubAccounts", {})
            result = response.json()
            
            if response.status_code == 200 and result.get('success'):
                return result.get('content', {})
            else:
                logger.error(f"Failed to get account info: {result.get('message', 'Unknown error')}")
                return {}
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}


if __name__ == "__main__":
    # Test example - DO NOT RUN without proper credentials
    print("AlgoLab API module loaded successfully")