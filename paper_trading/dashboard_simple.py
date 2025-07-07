"""
Simple Paper Trading Dashboard
Real-time monitoring for paper trading with AlgoLab integration
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from portfolio_manager import PortfolioManager
from performance_tracker import PerformanceTracker
from data_fetcher import DataFetcher
from config.settings import SACRED_SYMBOLS

# Page config
st.set_page_config(
    page_title="Paper Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.portfolios = {}
    st.session_state.tracker = None
    st.session_state.fetcher = None

# Header
st.title("📈 Paper Trading Dashboard - AlgoLab Integration")
st.markdown("Real-time monitoring of paper trading portfolios")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    if st.button("🔌 Initialize System"):
        with st.spinner("Loading portfolios..."):
            try:
                # Load portfolios
                st.session_state.portfolios = {
                    'balanced': PortfolioManager('balanced'),
                    'aggressive': PortfolioManager('aggressive'),
                    'conservative': PortfolioManager('conservative')
                }
                st.session_state.tracker = PerformanceTracker()
                st.session_state.fetcher = DataFetcher(use_algolab=True)
                st.session_state.initialized = True
                st.success("✅ System initialized!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.initialized:
        st.markdown("---")
        
        # Portfolio selector
        selected_portfolio = st.selectbox(
            "Select Portfolio",
            options=['all'] + list(st.session_state.portfolios.keys())
        )
        
        # Update prices button
        if st.button("🔄 Update Prices"):
            with st.spinner("Fetching from AlgoLab..."):
                try:
                    prices = st.session_state.fetcher.get_current_prices()
                    st.success(f"✅ Updated {len(prices)} prices")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Main content
if st.session_state.initialized:
    # Get current prices
    try:
        current_prices = st.session_state.fetcher.get_current_prices()
    except:
        current_prices = {}
    
    # Display based on selection
    if selected_portfolio == 'all':
        # Overview of all portfolios
        st.subheader("📊 All Portfolios Overview")
        
        cols = st.columns(3)
        for i, (name, pm) in enumerate(st.session_state.portfolios.items()):
            with cols[i]:
                # Update position prices first
                for symbol, position in pm.positions.items():
                    if symbol in current_prices:
                        position.update_price(current_prices[symbol])
                
                value = pm.get_portfolio_value()
                ret_pct = ((value - pm.initial_capital) / pm.initial_capital) * 100
                
                st.metric(
                    label=f"{name.title()}",
                    value=f"₺{value:,.2f}",
                    delta=f"{ret_pct:+.2f}%"
                )
                st.write(f"Positions: {len(pm.positions)}")
                st.write(f"Cash: ₺{pm.cash:,.2f}")
                
                # Mini performance chart
                history = pm.get_performance_history()
                if history:
                    df = pd.DataFrame(history)
                    if 'timestamp' in df.columns:
                        fig = px.line(df, x='timestamp', y='portfolio_value', 
                                     title=f"{name.title()} Performance")
                        fig.update_layout(height=200, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Detailed view of selected portfolio
        pm = st.session_state.portfolios[selected_portfolio]
        
        # Update position prices first
        for symbol in pm.positions:
            if symbol in current_prices:
                pm.positions[symbol].current_price = current_prices[symbol]
        
        value = pm.get_portfolio_value()
        ret_amt = value - pm.initial_capital
        ret_pct = (ret_amt / pm.initial_capital) * 100
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", f"₺{value:,.2f}", f"₺{ret_amt:+,.2f}")
        
        with col2:
            st.metric("Total Return", f"{ret_pct:.2f}%", f"from ₺{pm.initial_capital:,.0f}")
        
        with col3:
            st.metric("Available Cash", f"₺{pm.cash:,.2f}", f"{(pm.cash/value*100):.1f}% of portfolio")
        
        with col4:
            win_rate = pm.calculate_win_rate()
            st.metric("Win Rate", f"{win_rate:.1f}%", f"{len(pm.trades_history)} trades")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Positions", "📈 Performance", "📋 Trade History"])
        
        with tab1:
            st.subheader("Current Positions")
            
            if pm.positions:
                # Create positions dataframe
                positions_data = []
                for symbol, pos in pm.positions.items():
                    current_price = current_prices.get(symbol, pos.get('current_price', pos['entry_price']))
                    value = pos['shares'] * current_price
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                    pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    
                    positions_data.append({
                        'Symbol': symbol,
                        'Shares': pos['shares'],
                        'Entry Price': f"₺{pos['entry_price']:.2f}",
                        'Current Price': f"₺{current_price:.2f}",
                        'Value': f"₺{value:,.2f}",
                        'P&L': f"₺{pnl:+,.2f}",
                        'P&L %': f"{pnl_pct:+.2f}%",
                        'Date': pos['entry_date']
                    })
                
                df = pd.DataFrame(positions_data)
                st.dataframe(df, use_container_width=True)
                
                # Allocation pie chart
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Position allocation
                    allocation_data = []
                    for symbol, pos in pm.positions.items():
                        current_price = current_prices.get(symbol, pos['entry_price'])
                        value = pos['shares'] * current_price
                        allocation_data.append({'Symbol': symbol, 'Value': value})
                    
                    allocation_data.append({'Symbol': 'Cash', 'Value': pm.cash})
                    
                    allocation_df = pd.DataFrame(allocation_data)
                    fig = px.pie(allocation_df, values='Value', names='Symbol', 
                                title='Portfolio Allocation')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Performance by position
                    perf_data = []
                    for symbol, pos in pm.positions.items():
                        current_price = current_prices.get(symbol, pos['entry_price'])
                        pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                        perf_data.append({'Symbol': symbol, 'Return %': pnl_pct})
                    
                    perf_df = pd.DataFrame(perf_data)
                    fig = px.bar(perf_df, x='Symbol', y='Return %', 
                                title='Position Performance',
                                color='Return %',
                                color_continuous_scale=['red', 'yellow', 'green'])
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active positions")
        
        with tab2:
            st.subheader("Performance Analysis")
            
            # Performance history
            history = pm.get_performance_history()
            if history:
                df = pd.DataFrame(history)
                
                # Portfolio value over time
                if 'timestamp' in df.columns and 'portfolio_value' in df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['portfolio_value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Add initial capital line
                    fig.add_hline(y=pm.initial_capital, line_dash="dash", 
                                 line_color="gray", annotation_text="Initial Capital")
                    
                    fig.update_layout(
                        title="Portfolio Value Over Time",
                        xaxis_title="Date",
                        yaxis_title="Value (₺)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Daily returns
                if len(df) > 1:
                    df['daily_return'] = df['portfolio_value'].pct_change() * 100
                    
                    fig = px.bar(df.dropna(), x='timestamp', y='daily_return',
                                title='Daily Returns (%)',
                                color='daily_return',
                                color_continuous_scale=['red', 'white', 'green'],
                                color_continuous_midpoint=0)
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No performance history available")
        
        with tab3:
            st.subheader("Trade History")
            
            if pm.trades_history:
                # Create trades dataframe
                trades_data = []
                for trade in pm.trades_history:
                    trades_data.append({
                        'Symbol': trade['symbol'],
                        'Type': trade['type'],
                        'Shares': trade['shares'],
                        'Entry Price': f"₺{trade['entry_price']:.2f}",
                        'Exit Price': f"₺{trade.get('exit_price', 0):.2f}",
                        'P&L': f"₺{trade.get('profit', 0):+,.2f}",
                        'P&L %': f"{trade.get('profit_pct', 0):+.2f}%",
                        'Entry Date': trade['entry_date'],
                        'Exit Date': trade.get('exit_date', 'Active'),
                        'Reason': trade.get('exit_reason', '-')
                    })
                
                df = pd.DataFrame(trades_data)
                st.dataframe(df, use_container_width=True)
                
                # Trade statistics
                closed_trades = [t for t in pm.trades_history if 'exit_date' in t]
                if closed_trades:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        wins = len([t for t in closed_trades if t.get('profit', 0) > 0])
                        losses = len([t for t in closed_trades if t.get('profit', 0) < 0])
                        st.write(f"**Wins:** {wins}")
                        st.write(f"**Losses:** {losses}")
                        st.write(f"**Win Rate:** {(wins/(wins+losses)*100 if wins+losses > 0 else 0):.1f}%")
                    
                    with col2:
                        avg_win = sum(t.get('profit_pct', 0) for t in closed_trades if t.get('profit', 0) > 0) / max(wins, 1)
                        avg_loss = sum(t.get('profit_pct', 0) for t in closed_trades if t.get('profit', 0) < 0) / max(losses, 1)
                        st.write(f"**Avg Win:** {avg_win:.2f}%")
                        st.write(f"**Avg Loss:** {avg_loss:.2f}%")
                        st.write(f"**Risk/Reward:** {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")
                    
                    with col3:
                        total_profit = sum(t.get('profit', 0) for t in closed_trades)
                        best_trade = max(closed_trades, key=lambda x: x.get('profit_pct', 0))
                        worst_trade = min(closed_trades, key=lambda x: x.get('profit_pct', 0))
                        st.write(f"**Total P&L:** ₺{total_profit:+,.2f}")
                        st.write(f"**Best Trade:** {best_trade.get('profit_pct', 0):+.2f}%")
                        st.write(f"**Worst Trade:** {worst_trade.get('profit_pct', 0):+.2f}%")
            else:
                st.info("No trades executed yet")
    
    # Last update time
    if current_prices:
        st.markdown("---")
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"Data source: {'AlgoLab' if st.session_state.fetcher.use_algolab else 'Yahoo Finance'}")

else:
    # Not initialized
    st.info("👈 Please click 'Initialize System' in the sidebar to start")
    
    st.markdown("""
    ### 📋 Paper Trading System
    
    This dashboard monitors three portfolios:
    - **Balanced**: VixFix Enhanced Supertrend strategy
    - **Aggressive**: Supertrend Only strategy
    - **Conservative**: MACD + ADX strategy
    
    ### 🚀 Features
    - Real-time price updates from AlgoLab API
    - Portfolio performance tracking
    - Position monitoring
    - Trade history analysis
    
    ### 📊 Data Source
    - **Market Hours (10:00-18:00)**: AlgoLab real-time prices
    - **After Hours**: Yahoo Finance data
    """)

# Auto-refresh
if st.sidebar.checkbox("Auto Refresh", value=False):
    st.empty()
    import time
    time.sleep(st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60))
    st.rerun()