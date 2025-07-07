#!/bin/bash

# Trading Monitor Runner

echo "Starting Trading Monitor Dashboard..."
echo "===================================="
echo "Features:"
echo "- BIST100 Index"
echo "- 20 Sacred Symbols with live prices"
echo "- Calculated entry, stop loss, and targets"
echo "- Buy/Sell signals with strength"
echo "- Auto-refresh every 30 seconds"
echo "===================================="

# Activate virtual environment if exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run streamlit monitor
streamlit run trading_monitor.py --server.port 8502 --server.address localhost

echo "Monitor stopped."