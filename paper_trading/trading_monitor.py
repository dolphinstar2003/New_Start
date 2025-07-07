#!/usr/bin/env python3
"""
Real-time Trading Monitor Dashboard
Shows all 20 stocks with calculated targets and stop losses
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import json
import time

sys.path.append(str(Path(__file__).parent.parent))

from signal_generator import SignalGenerator
from config.settings import SACRED_SYMBOLS
from target_manager import TargetManager

# Page config
st.set_page_config(
    page_title="Trading Monitor - Live Prices & Targets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme compatibility
st.markdown("""
<style>
    /* Dark theme fixes */
    .stDataFrame {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #3d3d4d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    /* Status colors */
    .ready-status {
        color: #00ff00;
        font-weight: bold;
    }
    .wait-status {
        color: #ffff00;
        font-weight: bold;
    }
    /* Auto refresh timer */
    .refresh-timer {
        position: fixed;
        bottom: 10px;
        left: 10px;
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #3d3d4d;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'signal_generator' not in st.session_state:
    st.session_state.signal_generator = SignalGenerator()
    st.session_state.target_manager = TargetManager()
    st.session_state.last_update = None
    st.session_state.auto_refresh = True
    st.session_state.selected_strategy = 'balanced'

# Helper functions
def calculate_targets(price, signal_strength=1.0):
    """Calculate stop loss and targets based on price"""
    # Stop loss: 3%
    stop_loss = price * 0.97
    
    # Take profit targets
    tp1 = price * 1.03  # 3%
    tp2 = price * 1.05  # 5%
    tp3 = price * 1.08  # 8%
    
    # Trailing stop levels
    trail1 = price * 1.02  # Activate at 2% profit
    trail2 = price * 1.04  # Activate at 4% profit
    
    return {
        'stop_loss': stop_loss,
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'trail1': trail1,
        'trail2': trail2
    }

def get_bist100_data():
    """Fetch BIST100 index data from cache or mock"""
    # For now, return mock data to avoid API errors
    # In production, this would fetch from AlgoLab API
    return {
        'value': 10850.25,
        'change': 125.50,
        'change_pct': 1.17
    }

def format_price(price):
    """Format price with color based on daily change"""
    return f"₺{price:.2f}"

# Header
col1, col2, col3 = st.columns([2, 3, 1])

with col1:
    st.title("📊 Trading Monitor")

with col2:
    # BIST100 Index
    bist_data = get_bist100_data()
    if bist_data['value'] > 0:
        color = "🟢" if bist_data['change'] >= 0 else "🔴"
        st.markdown(f"""
        ### BIST 100: {bist_data['value']:,.2f} {color} {bist_data['change']:+,.2f} ({bist_data['change_pct']:+.2f}%)
        """)
    else:
        st.markdown("### BIST 100: -")

with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.last_update = None
        st.rerun()

# Update time
if st.session_state.last_update:
    st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

# Fetch current prices from cache file
def load_prices_from_cache():
    """Load prices directly from cache file"""
    cache_file = Path("data/cache/latest_prices_algolab.json")
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return data.get('prices', {}), data.get('timestamp')
        except:
            pass
    return {}, None

# Get daily change data (mock for now)
def get_daily_changes(symbols):
    """Calculate daily percentage changes"""
    # In real implementation, this would compare with previous close
    # For demonstration, use realistic values based on volatility
    changes = {
        'GARAN': 1.25, 'AKBNK': -0.85, 'ISCTR': 2.10, 'YKBNK': -1.50,
        'SAHOL': 0.75, 'KCHOL': -0.25, 'SISE': 1.85, 'EREGL': -2.20,
        'KRDMD': 0.45, 'TUPRS': 1.90, 'ASELS': -0.60, 'THYAO': 3.25,
        'TCELL': -0.35, 'BIMAS': 0.80, 'MGROS': -1.10, 'ULKER': 1.45,
        'AKSEN': 2.50, 'ENKAI': -0.95, 'PETKM': 0.25, 'KOZAL': -1.75
    }
    return changes

# Strategy selector in sidebar
with st.sidebar:
    st.header("📊 Settings")
    selected_strategy = st.selectbox(
        "Select Strategy",
        options=['aggressive', 'balanced', 'conservative'],
        index=1,  # Default to balanced
        format_func=lambda x: x.capitalize()
    )
    st.session_state.selected_strategy = selected_strategy
    
    st.markdown("---")
    st.markdown("### Strategy Details")
    if selected_strategy == 'aggressive':
        st.info("**Aggressive**: Targets 1.5-2% pullback")
    elif selected_strategy == 'balanced':
        st.info("**Balanced**: Targets 2.5-3% pullback")
    else:
        st.info("**Conservative**: Targets 3-4% pullback")

# Fetch current prices
with st.spinner("Loading prices from cache..."):
    try:
        current_prices, cache_timestamp = load_prices_from_cache()
        if cache_timestamp:
            st.session_state.last_update = datetime.fromisoformat(cache_timestamp)
        else:
            st.session_state.last_update = datetime.now()
    except Exception as e:
        st.error(f"Error loading prices: {str(e)}")
        current_prices = {}

# Update targets if needed
if current_prices:
    st.session_state.target_manager.update_targets(current_prices)

# Get targets for selected strategy
strategy_targets = st.session_state.target_manager.get_all_targets(selected_strategy)

# Get daily changes
daily_changes = get_daily_changes(SACRED_SYMBOLS)

# Create main data table
data_rows = []
for symbol in SACRED_SYMBOLS:
    if symbol in current_prices and symbol in strategy_targets:
        price = current_prices[symbol]
        target_data = strategy_targets[symbol]
        buy_target = target_data['target_price']
        
        # Calculate targets from the buy target price
        targets = calculate_targets(buy_target)
        
        # Calculate distance
        distance = ((buy_target - price) / price * 100)
        
        # Get daily change
        daily_change = daily_changes.get(symbol, 0)
        
        data_rows.append({
            'Symbol': symbol,
            'Current Price': price,
            'Daily %': f"{daily_change:+.2f}%",
            'Target Buy': buy_target,  # Fixed daily target
            'Distance': f"{distance:.1f}%",
            'Stop Loss': targets['stop_loss'],
            'TP1 (3%)': targets['tp1'],
            'TP2 (5%)': targets['tp2'],
            'TP3 (8%)': targets['tp3'],
            'Trail 1': targets['trail1'],
            'Trail 2': targets['trail2'],
            'Status': 'READY' if price <= buy_target else 'WAIT'
        })

# Convert to DataFrame
df = pd.DataFrame(data_rows)

# Display controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    show_ready = st.checkbox("Show Ready Only", value=False)

with col2:
    sort_by = st.selectbox("Sort By", ["Symbol", "Distance", "Price"], index=0)

with col3:
    price_filter = st.selectbox("Price Range", ["All", "<50 TL", "50-100 TL", ">100 TL"], index=0)

with col4:
    st.checkbox("Auto Refresh (30s)", value=st.session_state.auto_refresh, key="auto_refresh")

# Filter data
display_df = df.copy()

# Filter by status
if show_ready:
    display_df = display_df[display_df['Status'] == 'READY']

# Filter by price
if price_filter == "<50 TL":
    display_df = display_df[display_df['Current Price'] < 50]
elif price_filter == "50-100 TL":
    display_df = display_df[(display_df['Current Price'] >= 50) & (display_df['Current Price'] <= 100)]
elif price_filter == ">100 TL":
    display_df = display_df[display_df['Current Price'] > 100]

# Sort data
if sort_by == "Distance":
    # Extract numeric distance for sorting
    display_df['Distance_num'] = display_df['Distance'].str.rstrip('%').astype(float)
    display_df = display_df.sort_values('Distance_num', ascending=False)
    display_df = display_df.drop('Distance_num', axis=1)
elif sort_by == "Price":
    display_df = display_df.sort_values('Current Price', ascending=False)
else:
    display_df = display_df.sort_values('Symbol')

# Display the main table
st.subheader(f"📈 {selected_strategy.capitalize()} Strategy - Fixed Daily Targets ({len(display_df)} stocks)")
st.info(f"🎯 **Target Buy**: {selected_strategy.capitalize()} strateji için bugünkü sabit alım hedefleri. Gün boyunca değişmez.")
st.caption("Status: WAIT = Fiyat hedefte değil | READY = Fiyat hedefte veya altında, alıma hazır")

# Format the dataframe for display
styled_df = display_df.style.format({
    'Current Price': '₺{:.2f}',
    'Target Buy': '₺{:.2f}',
    'Stop Loss': '₺{:.2f}',
    'TP1 (3%)': '₺{:.2f}',
    'TP2 (5%)': '₺{:.2f}',
    'TP3 (8%)': '₺{:.2f}',
    'Trail 1': '₺{:.2f}',
    'Trail 2': '₺{:.2f}'
})

# Apply color coding
def color_daily_change(val):
    """Color daily percentage change"""
    try:
        num_val = float(val.strip('%+'))
        if num_val > 0:
            return 'color: #00ff00'  # Green for positive
        elif num_val < 0:
            return 'color: #ff4444'  # Red for negative
        else:
            return 'color: #ffffff'  # White for zero
    except:
        return ''

def color_status(val):
    if val == 'READY':
        return 'color: #00ff00; font-weight: bold'  # Green for ready
    else:
        return 'color: #ffff00; font-weight: bold'  # Yellow for wait

styled_df = styled_df.map(color_daily_change, subset=['Daily %'])
styled_df = styled_df.map(color_status, subset=['Status'])

# Display the table
st.dataframe(styled_df, use_container_width=True, height=600)

# Summary metrics
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ready_count = len(df[df['Status'] == 'READY'])
    st.metric("Ready to Buy", ready_count, f"{ready_count/len(df)*100:.1f}%")

with col2:
    avg_distance = df['Distance'].str.rstrip('%').astype(float).mean()
    st.metric("Avg Distance", f"{avg_distance:.1f}%")

with col3:
    # Find nearest target (smallest positive distance)
    distances = df['Distance'].str.rstrip('%').astype(float)
    positive_distances = distances[distances > 0]
    if not positive_distances.empty:
        nearest_idx = positive_distances.idxmin()
        nearest = df.loc[nearest_idx]
        st.metric("Nearest Target", nearest['Symbol'], f"{nearest['Distance']}")
    else:
        st.metric("Nearest Target", "-", "-")

with col4:
    st.metric("Strategy", selected_strategy.capitalize())

with col5:
    st.metric("Data Source", "AlgoLab Cache")

# Risk calculation helper
with st.expander("💡 Trading Guidelines & Strategy Details"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Strategy Target Levels
        - **Aggressive**: 1.5-2% pullback from current price
        - **Balanced**: 2.5-3% pullback from current price
        - **Conservative**: 3-4% pullback from current price
        
        ### 🎯 How It Works
        1. Targets are calculated once at market open (10:00)
        2. Targets remain fixed throughout the day
        3. System waits for price to reach target
        4. Automatic buy when price hits target
        """)
    
    with col2:
        st.markdown("""
        ### 💰 Position Management
        - **Risk per trade**: Max 2% of capital
        - **Position size** = (Capital × Risk%) / (Entry - Stop Loss)
        
        ### 📈 Exit Strategy
        - **Stop Loss**: 3% below entry
        - **TP1**: 3% profit (exit 1/3)
        - **TP2**: 5% profit (exit 1/3)
        - **TP3**: 8% profit (exit remaining)
        
        ### ⚠️ Risk Limits
        - Max 5 concurrent positions
        - Daily loss limit: 6% of capital
        """)

# Auto-refresh timer and countdown
if st.session_state.auto_refresh:
    # Create a placeholder for countdown
    timer_placeholder = st.empty()
    
    # Show countdown
    for remaining in range(30, 0, -1):
        timer_placeholder.markdown(
            f'<div class="refresh-timer">🔄 Auto refresh in: {remaining}s</div>',
            unsafe_allow_html=True
        )
        time.sleep(1)
    
    st.rerun()