#!/usr/bin/env python3
"""
Simplified Trading Monitor for Streamlit Deployment
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Trading Monitor - Fixed Daily Targets",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📊 Trading Monitor - Fixed Daily Targets")
st.markdown("---")

# Mock data for demonstration
SYMBOLS = [
    'GARAN', 'AKBNK', 'ISCTR', 'YKBNK', 'SAHOL',
    'KCHOL', 'SISE', 'EREGL', 'KRDMD', 'TUPRS',
    'ASELS', 'THYAO', 'TCELL', 'BIMAS', 'MGROS',
    'ULKER', 'AKSEN', 'ENKAI', 'PETKM', 'KOZAL'
]

# Mock prices
current_prices = {
    'GARAN': 136.3, 'AKBNK': 68.1, 'ISCTR': 13.82, 'YKBNK': 31.9,
    'SAHOL': 90.95, 'KCHOL': 157.2, 'SISE': 35.7, 'EREGL': 26.92,
    'KRDMD': 25.72, 'TUPRS': 146.1, 'ASELS': 151.3, 'THYAO': 288.0,
    'TCELL': 97.0, 'BIMAS': 500.5, 'MGROS': 497.25, 'ULKER': 108.8,
    'AKSEN': 32.82, 'ENKAI': 65.95, 'PETKM': 17.26, 'KOZAL': 24.0
}

# Mock daily changes
daily_changes = {
    'GARAN': 1.25, 'AKBNK': -0.85, 'ISCTR': 2.10, 'YKBNK': -1.50,
    'SAHOL': 0.75, 'KCHOL': -0.25, 'SISE': 1.85, 'EREGL': -2.20,
    'KRDMD': 0.45, 'TUPRS': 1.90, 'ASELS': -0.60, 'THYAO': 3.25,
    'TCELL': -0.35, 'BIMAS': 0.80, 'MGROS': -1.10, 'ULKER': 1.45,
    'AKSEN': 2.50, 'ENKAI': -0.95, 'PETKM': 0.25, 'KOZAL': -1.75
}

# Sidebar
with st.sidebar:
    st.header("📊 Settings")
    selected_strategy = st.selectbox(
        "Select Strategy",
        ["Aggressive", "Balanced", "Conservative"],
        index=1
    )
    
    st.markdown("---")
    st.markdown("### Strategy Details")
    if selected_strategy == "Aggressive":
        st.info("**Aggressive**: Targets 1.5-2% pullback")
        pullback = 0.98
    elif selected_strategy == "Balanced":
        st.info("**Balanced**: Targets 2.5-3% pullback")
        pullback = 0.97
    else:
        st.info("**Conservative**: Targets 3-4% pullback")
        pullback = 0.96

# Calculate targets
data_rows = []
for symbol in SYMBOLS:
    price = current_prices[symbol]
    change = daily_changes[symbol]
    
    # Calculate target based on strategy
    if price > 100:
        target = round(price * pullback, 2)
    else:
        target = round(price * (pullback + 0.005), 2)
    
    # Calculate derived values
    stop_loss = round(target * 0.97, 2)
    tp1 = round(target * 1.03, 2)
    tp2 = round(target * 1.05, 2)
    tp3 = round(target * 1.08, 2)
    
    distance = round((target - price) / price * 100, 2)
    status = "READY" if price <= target else "WAIT"
    
    data_rows.append({
        'Symbol': symbol,
        'Current': f"₺{price:.2f}",
        'Daily %': f"{change:+.2f}%",
        'Target': f"₺{target:.2f}",
        'Distance': f"{distance:.1f}%",
        'Stop Loss': f"₺{stop_loss:.2f}",
        'TP1 (3%)': f"₺{tp1:.2f}",
        'TP2 (5%)': f"₺{tp2:.2f}",
        'TP3 (8%)': f"₺{tp3:.2f}",
        'Status': status
    })

# Create DataFrame
df = pd.DataFrame(data_rows)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    ready_count = len(df[df['Status'] == 'READY'])
    st.metric("Ready to Buy", ready_count, f"{ready_count/len(df)*100:.1f}%")

with col2:
    st.metric("Total Symbols", len(SYMBOLS))

with col3:
    st.metric("Strategy", selected_strategy)

with col4:
    st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))

# Display table
st.markdown("---")
st.subheader(f"📈 {selected_strategy} Strategy - Fixed Daily Targets")

# Style the dataframe
def color_status(val):
    if val == 'READY':
        color = 'background-color: #90EE90'
    else:
        color = 'background-color: #F0E68C'
    return color

def color_daily(val):
    try:
        num = float(val.strip('%+'))
        if num > 0:
            color = 'color: green'
        elif num < 0:
            color = 'color: red'
        else:
            color = 'color: black'
    except:
        color = 'color: black'
    return color

# Apply styling
styled_df = df.style.map(color_status, subset=['Status'])
styled_df = styled_df.map(color_daily, subset=['Daily %'])

st.dataframe(styled_df, use_container_width=True, height=600)

# Footer
st.markdown("---")
st.caption("💡 Targets are fixed for the day and calculated at market open")
st.caption("🔄 Auto-refresh: Every 30 seconds")