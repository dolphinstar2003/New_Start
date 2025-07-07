#!/bin/bash

# Paper Trading Dashboard Runner

echo "Starting Paper Trading Dashboard..."
echo "=================================="
echo "Data Source: AlgoLab API (Real-time)"
echo "Portfolios: Balanced, Aggressive, Conservative"
echo "=================================="

# Activate virtual environment if exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run streamlit dashboard
streamlit run dashboard_simple.py --server.port 8501 --server.address localhost

echo "Dashboard stopped."