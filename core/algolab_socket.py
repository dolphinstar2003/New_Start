"""
AlgoLab WebSocket Client
Real-time data streaming
"""
import json
import time
import threading
from typing import Dict, Any, Callable, Optional
import websocket
from loguru import logger
from datetime import datetime

# Configure logger
logger.add("logs/algolab_socket.log", rotation="1 day", retention="7 days")


class AlgoLabSocket:
    """WebSocket client for AlgoLab real-time data"""
    
    def __init__(self, api_key: str, hash_token: str):
        """
        Initialize WebSocket client
        
        Args:
            api_key: API key from AlgoLab
            hash_token: Session hash from login
        """
        self.api_key = api_key
        self.hash = hash_token
        self.ws_url = "wss://www.algolab.com.tr/api/ws"
        
        self.ws = None
        self.connected = False
        self.running = False
        
        # Callbacks for different message types
        self.callbacks = {
            'price': None,
            'depth': None,
            'trade': None,
            'index': None,
            'error': None
        }
        
        # Subscribed symbols
        self.subscribed_symbols = set()
        
        # Connection thread
        self.connection_thread = None
        
        logger.info("AlgoLab WebSocket initialized")
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get('type', 'unknown')
            
            # Log message
            logger.debug(f"Received {msg_type}: {data}")
            
            # Call appropriate callback
            if msg_type in self.callbacks and self.callbacks[msg_type]:
                self.callbacks[msg_type](data)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            if self.callbacks.get('error'):
                self.callbacks['error'](str(e))
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        if self.callbacks.get('error'):
            self.callbacks['error'](str(error))
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        self.connected = False
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(self, ws):
        """Handle WebSocket open"""
        self.connected = True
        logger.info("WebSocket connected")
        
        # Send authentication
        auth_message = {
            "type": "auth",
            "apikey": self.api_key,
            "hash": self.hash
        }
        ws.send(json.dumps(auth_message))
        
        # Re-subscribe to symbols
        for symbol in self.subscribed_symbols:
            self.subscribe(symbol)
    
    def connect(self):
        """Connect to WebSocket server"""
        if self.connected:
            logger.warning("Already connected")
            return
        
        self.running = True
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run in separate thread
        self.connection_thread = threading.Thread(
            target=self._run_forever,
            daemon=True
        )
        self.connection_thread.start()
        
        # Wait for connection
        timeout = 10
        start_time = time.time()
        while not self.connected and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if self.connected:
            logger.info("WebSocket connection established")
        else:
            logger.error("WebSocket connection timeout")
            raise ConnectionError("Failed to connect to WebSocket")
    
    def _run_forever(self):
        """Run WebSocket connection"""
        while self.running:
            try:
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket run error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5 seconds...")
                    time.sleep(5)
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        self.running = False
        if self.ws:
            self.ws.close()
        self.connected = False
        logger.info("WebSocket disconnected")
    
    def subscribe(self, symbol: str, data_types: list = None):
        """
        Subscribe to symbol data
        
        Args:
            symbol: Symbol code (e.g., 'GARAN')
            data_types: List of data types to subscribe ['price', 'depth', 'trade']
        """
        if not self.connected:
            logger.error("Not connected to WebSocket")
            return
        
        if data_types is None:
            data_types = ['price', 'depth', 'trade']
        
        message = {
            "type": "subscribe",
            "symbol": symbol,
            "data": data_types
        }
        
        self.ws.send(json.dumps(message))
        self.subscribed_symbols.add(symbol)
        logger.info(f"Subscribed to {symbol} - {data_types}")
    
    def unsubscribe(self, symbol: str):
        """
        Unsubscribe from symbol data
        
        Args:
            symbol: Symbol code
        """
        if not self.connected:
            logger.error("Not connected to WebSocket")
            return
        
        message = {
            "type": "unsubscribe",
            "symbol": symbol
        }
        
        self.ws.send(json.dumps(message))
        self.subscribed_symbols.discard(symbol)
        logger.info(f"Unsubscribed from {symbol}")
    
    def set_callback(self, msg_type: str, callback: Callable):
        """
        Set callback for message type
        
        Args:
            msg_type: Message type ('price', 'depth', 'trade', 'index', 'error')
            callback: Callback function
        """
        if msg_type in self.callbacks:
            self.callbacks[msg_type] = callback
            logger.info(f"Callback set for {msg_type}")
        else:
            logger.error(f"Unknown message type: {msg_type}")
    
    def send_heartbeat(self):
        """Send heartbeat to keep connection alive"""
        if self.connected:
            message = {"type": "ping"}
            self.ws.send(json.dumps(message))


# Example callbacks
def on_price_update(data: Dict[str, Any]):
    """Example price update callback"""
    symbol = data.get('symbol')
    price = data.get('price')
    volume = data.get('volume')
    timestamp = data.get('timestamp')
    
    logger.info(f"Price Update - {symbol}: {price} @ {volume} - {timestamp}")


def on_depth_update(data: Dict[str, Any]):
    """Example depth update callback"""
    symbol = data.get('symbol')
    bids = data.get('bids', [])
    asks = data.get('asks', [])
    
    logger.info(f"Depth Update - {symbol}: Bids: {len(bids)}, Asks: {len(asks)}")


def on_trade_update(data: Dict[str, Any]):
    """Example trade update callback"""
    symbol = data.get('symbol')
    price = data.get('price')
    quantity = data.get('quantity')
    side = data.get('side')
    
    logger.info(f"Trade - {symbol}: {side} {quantity} @ {price}")


if __name__ == "__main__":
    print("AlgoLab WebSocket module loaded successfully")