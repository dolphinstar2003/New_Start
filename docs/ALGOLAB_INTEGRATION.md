# AlgoLab API Integration Guide

## Overview
This guide explains how to use the AlgoLab API integration in the BIST100 Trading System.

## ⚠️ IMPORTANT WARNINGS
- **You have only 2 SMS authentication attempts!**
- **Make sure your phone is ready before starting authentication**
- **Session expires after 24 hours**

## Setup

### 1. Environment Variables
Create a `.env` file with your AlgoLab credentials:
```env
ALGOLAB_API_KEY=your_api_key_here
ALGOLAB_USERNAME=your_tc_or_username
ALGOLAB_PASSWORD=your_internet_banking_password
```

### 2. Install Dependencies
```bash
pip install pycryptodome requests websocket-client
```

## Authentication Flow

### Step 1: Initial Setup
```python
from utils.algolab_auth import AlgoLabAuth

# Initialize auth helper
auth = AlgoLabAuth()

# This will:
# 1. Check for credentials in .env
# 2. Ask for missing credentials
# 3. Send SMS to your registered phone
# 4. Ask for SMS code (2 attempts max)
api = auth.authenticate()
```

### Step 2: Using the API
```python
if api and api.is_authenticated():
    # Get portfolio
    portfolio = api.get_portfolio()
    
    # Get symbol info
    symbol_info = api.get_symbol_info("GARAN")
    
    # More operations...
```

## Test Script

Run the test script to verify authentication:
```bash
python test_algolab_auth.py
```

This will:
1. Check your configuration
2. Ask for confirmation before proceeding
3. Guide you through the SMS authentication
4. Test the API connection
5. Optionally show portfolio summary

## API Methods

### Core Methods
- `login_user()` - First step, sends SMS
- `login_user_control(sms_code)` - Verify SMS code
- `is_authenticated()` - Check session validity
- `get_portfolio()` - Get portfolio data
- `get_symbol_info(symbol)` - Get symbol information

### WebSocket (Real-time Data)
```python
from core.algolab_socket import AlgoLabSocket

# Initialize with API key and hash from login
socket = AlgoLabSocket(api_key, hash_token)

# Set callbacks
socket.set_callback('price', on_price_update)
socket.set_callback('depth', on_depth_update)

# Connect and subscribe
socket.connect()
socket.subscribe('GARAN', ['price', 'depth'])
```

## Session Management

### Automatic Session Caching
- Sessions are cached in `data/cache/algolab_session.pkl`
- Valid for 24 hours
- Automatically loaded on initialization

### Manual Session Refresh
```python
# Check if session is valid
if not api.is_authenticated():
    # Re-authenticate
    api = auth.authenticate()
```

## Error Handling

### Common Errors
1. **Invalid Credentials**: Check username/password
2. **SMS Code Wrong**: You have 2 attempts only
3. **Session Expired**: Re-authenticate
4. **Rate Limiting**: 5 seconds between requests

### Best Practices
1. Always check `is_authenticated()` before API calls
2. Handle SMS input carefully (2 attempts only!)
3. Save credentials in .env file
4. Use session caching to avoid re-authentication
5. Implement proper error handling

## Rate Limiting
- API has 5-second rate limit between requests
- Automatically handled by the wrapper
- Don't make parallel requests

## Security Notes
1. Never commit .env file
2. Keep API key secure
3. Session tokens expire after 24 hours
4. Use environment variables for credentials

## Troubleshooting

### Authentication Fails
1. Check internet connection
2. Verify credentials are correct
3. Make sure you're in Turkey (API geo-restricted)
4. Check if account has API access enabled

### SMS Not Received
1. Check phone number in AlgoLab account
2. Wait 1-2 minutes
3. Check SMS spam folder
4. Contact AlgoLab support

### Session Expires Frequently
1. Check system time is correct
2. Use session caching
3. Implement auto-refresh logic