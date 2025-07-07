"""
Paper Trading Dashboard
Real-time monitoring and control interface for the paper trading module
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from paper_trading.portfolio_manager import PortfolioManager
from paper_trading.signal_generator import SignalGenerator
from paper_trading.performance_tracker import PerformanceTracker
from paper_trading.data_fetcher import DataFetcher
from config.settings import SACRED_SYMBOLS, DATA_DIR

# Page config
st.set_page_config(
    page_title="Dynamic Portfolio Optimizer - Paper Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_managers' not in st.session_state:
    st.session_state.portfolio_managers = {}
    st.session_state.signal_generator = None
    st.session_state.performance_tracker = None
    st.session_state.data_fetcher = None
    st.session_state.is_initialized = False
    st.session_state.auto_refresh = True
    st.session_state.refresh_interval = 60  # seconds
    st.session_state.last_update = None

# Header
st.title("🚀 Paper Trading Dashboard - AlgoLab Integration")
st.markdown("Real-time monitoring of paper trading portfolios")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if st.button("🔌 Initialize System", key="init_btn"):
        with st.spinner("Initializing paper trading system..."):
            try:
                # Initialize components
                st.session_state.portfolio_managers = {
                    'balanced': PortfolioManager('balanced'),
                    'aggressive': PortfolioManager('aggressive'),
                    'conservative': PortfolioManager('conservative')
                }
                st.session_state.signal_generator = SignalGenerator()
                st.session_state.performance_tracker = PerformanceTracker()
                st.session_state.data_fetcher = DataFetcher(use_algolab=True)
                st.session_state.is_initialized = True
                st.success("✅ System initialized with AlgoLab!")
            except Exception as e:
                st.error(f"❌ Failed to initialize: {str(e)}")
    
    if st.session_state.is_initialized:
        st.markdown("---")
        
        # Portfolio selection
        selected_portfolio = st.selectbox(
            "Select Portfolio",
            options=['all', 'balanced', 'aggressive', 'conservative'],
            index=0
        )
        
        # Update data button
        if st.button("🔄 Update Market Data", key="update_btn"):
            with st.spinner("Fetching latest prices..."):
                try:
                    prices = st.session_state.data_fetcher.get_current_prices()
                    st.session_state.last_update = datetime.now()
                    st.success(f"Updated {len(prices)} prices")
                except Exception as e:
                    st.error(f"Failed to update: {str(e)}")
        
        # Auto refresh
        st.markdown("---")
        st.session_state.auto_refresh = st.checkbox(
            "🔄 Auto Refresh", 
            value=st.session_state.auto_refresh
        )
        
        st.session_state.refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=10
        )
        
        # Manual refresh
        if st.button("🔄 Refresh Now", key="refresh_btn"):
            st.rerun()

# Main content
if st.session_state.is_initialized:
    # Get latest prices
    try:
        current_prices = st.session_state.data_fetcher.get_current_prices()
    except:
        current_prices = {}
    
    # Helper function to get portfolio data
    def get_portfolio_data(portfolio_name):
        pm = st.session_state.portfolio_managers.get(portfolio_name)
        if not pm:
            return None
        
        # Update current prices
        for symbol, position in pm.positions.items():
            if symbol in current_prices:
                position['current_price'] = current_prices[symbol]
        
        return {
            'value': pm.get_portfolio_value(current_prices),
            'positions': pm.positions,
            'cash': pm.cash,
            'initial_capital': pm.initial_capital,
            'trades': pm.trades_history
        }
    
    # Show metrics based on selection
    if selected_portfolio == 'all':
        # Show all portfolios
        cols = st.columns(3)
        for i, (name, pm) in enumerate(st.session_state.portfolio_managers.items()):
            with cols[i]:
                data = get_portfolio_data(name)
                if data:
                    value = data['value']
                    ret_pct = ((value - data['initial_capital']) / data['initial_capital']) * 100
                    st.metric(
                        label=f"{name.title()} Portfolio",
                        value=f"₺{value:,.2f}",
                        delta=f"{ret_pct:+.2f}%"
                    )
                    st.write(f"Positions: {len(data['positions'])}")
                    st.write(f"Cash: ₺{data['cash']:,.2f}")
    else:
        # Show selected portfolio
        data = get_portfolio_data(selected_portfolio)
        if data:
            col1, col2, col3, col4 = st.columns(4)
            
            value = data['value']
            ret_amt = value - data['initial_capital']
            ret_pct = (ret_amt / data['initial_capital']) * 100
            
            with col1:
                st.metric(
                    label="Portfolio Value",
                    value=f"₺{value:,.2f}",
                    delta=f"₺{ret_amt:,.2f}"
                )
            
            with col2:
                st.metric(
                    label="Total Return",
                    value=f"{ret_pct:.2f}%",
                    delta=f"from ₺{data['initial_capital']:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Available Cash",
                    value=f"₺{data['cash']:,.2f}",
                    delta=f"{(data['cash']/value*100):.1f}% of portfolio"
                )
            
            with col4:
                st.metric(
                    label="Active Positions",
                    value=len(data['positions']),
                    delta=f"Win rate: {len([t for t in data['trades'] if t.get('profit', 0) > 0])/len(data['trades'])*100 if data['trades'] else 0:.1f}%"
                )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Portfolio Overview",
        "💼 Current Positions", 
        "📈 Trade History",
        "📉 Performance Analysis",
        "🎯 Market Opportunities"
    ])
    
    with tab1:
        # Portfolio composition
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio value chart
            if paper_trader.portfolio['daily_values']:
                daily_df = pd.DataFrame(paper_trader.portfolio['daily_values'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_df['datetime'],
                    y=daily_df['value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))
                
                # Add initial capital line
                fig.add_hline(
                    y=paper_trader.portfolio['initial_capital'],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Initial Capital"
                )
                
                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Portfolio allocation pie chart
            if status['positions']:
                positions_df = pd.DataFrame(status['positions'])
                positions_df['allocation_pct'] = (
                    positions_df['value'] / status['portfolio_value'] * 100
                )
                
                # Add cash
                allocation_data = positions_df[['symbol', 'value']].copy()
                allocation_data = pd.concat([
                    allocation_data,
                    pd.DataFrame({
                        'symbol': ['Cash'],
                        'value': [status['cash']]
                    })
                ])
                
                fig = px.pie(
                    allocation_data,
                    values='value',
                    names='symbol',
                    title='Portfolio Allocation'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No active positions - 100% in cash")
    
    with tab2:
        # Current positions table
        if status['positions']:
            positions_df = pd.DataFrame(status['positions'])
            
            # Format for display
            display_df = positions_df[[
                'symbol', 'shares', 'entry_price', 'current_price',
                'value', 'profit', 'profit_pct', 'holding_days'
            ]].copy()
            
            # Apply color coding
            def color_profit(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                    return f'color: {color}'
                return ''
            
            styled_df = display_df.style.applymap(
                color_profit,
                subset=['profit', 'profit_pct']
            ).format({
                'entry_price': '${:.2f}',
                'current_price': '${:.2f}',
                'value': '${:,.2f}',
                'profit': '${:,.2f}',
                'profit_pct': '{:.2f}%',
                'holding_days': '{:.0f}d'
            })
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Position details
            st.subheader("Position Details")
            selected_symbol = st.selectbox(
                "Select position for details:",
                options=positions_df['symbol'].tolist()
            )
            
            if selected_symbol:
                pos = positions_df[positions_df['symbol'] == selected_symbol].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Entry Price:** ${pos['entry_price']:.2f}")
                    st.write(f"**Current Price:** ${pos['current_price']:.2f}")
                    st.write(f"**Shares:** {pos['shares']}")
                
                with col2:
                    st.write(f"**Position Value:** ${pos['value']:,.2f}")
                    st.write(f"**Profit/Loss:** ${pos['profit']:,.2f}")
                    st.write(f"**Return:** {pos['profit_pct']:.2f}%")
                
                with col3:
                    st.write(f"**Holding Period:** {pos['holding_days']} days")
                    
                    # Stop loss info
                    stop_loss = paper_trader.portfolio['stop_losses'].get(selected_symbol)
                    if stop_loss:
                        st.write(f"**Stop Loss:** ${stop_loss:.2f}")
                    
                    # Trailing stop info
                    trailing_stop = paper_trader.portfolio['trailing_stops'].get(selected_symbol)
                    if trailing_stop:
                        st.write(f"**Trailing Stop:** ${trailing_stop:.2f}")
        else:
            st.info("No active positions")
    
    with tab3:
        # Trade history
        trades_df = paper_trader.get_trade_history()
        
        if not trades_df.empty:
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_profit = trades_df['profit'].sum()
                st.metric("Total Profit", f"${total_profit:,.2f}")
            
            with col2:
                avg_profit_pct = trades_df['profit_pct'].mean()
                st.metric("Avg Return", f"{avg_profit_pct:.2f}%")
            
            with col3:
                winning_trades = trades_df[trades_df['profit'] > 0]
                if len(winning_trades) > 0:
                    avg_win = winning_trades['profit_pct'].mean()
                    st.metric("Avg Win", f"{avg_win:.2f}%")
            
            with col4:
                losing_trades = trades_df[trades_df['profit'] < 0]
                if len(losing_trades) > 0:
                    avg_loss = losing_trades['profit_pct'].mean()
                    st.metric("Avg Loss", f"{avg_loss:.2f}%")
            
            # Trade history table
            st.subheader("Trade History")
            
            # Format for display
            display_trades = trades_df.copy()
            display_trades['entry_date'] = pd.to_datetime(display_trades['entry_date'])
            display_trades['exit_date'] = pd.to_datetime(display_trades['exit_date'])
            
            # Sort by exit date descending
            display_trades = display_trades.sort_values('exit_date', ascending=False)
            
            # Apply styling
            styled_trades = display_trades.style.applymap(
                color_profit,
                subset=['profit', 'profit_pct']
            ).format({
                'entry_price': '${:.2f}',
                'exit_price': '${:.2f}',
                'profit': '${:,.2f}',
                'profit_pct': '{:.2f}%',
                'entry_date': lambda x: x.strftime('%Y-%m-%d %H:%M'),
                'exit_date': lambda x: x.strftime('%Y-%m-%d %H:%M')
            })
            
            st.dataframe(styled_trades, use_container_width=True)
            
            # Exit reason breakdown
            st.subheader("Exit Reasons")
            reason_counts = trades_df['reason'].value_counts()
            
            fig = px.bar(
                x=reason_counts.index,
                y=reason_counts.values,
                title="Trade Exit Reasons",
                labels={'x': 'Reason', 'y': 'Count'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No trades executed yet")
    
    with tab4:
        # Performance analysis
        if metrics:
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Returns")
                st.write(f"**Total Return:** {status['total_return_pct']:.2f}%")
                st.write(f"**Average Trade:** {metrics['avg_profit_pct']:.2f}%")
                st.write(f"**Best Trade:** {metrics['max_win']:.2f}%")
                st.write(f"**Worst Trade:** {metrics['max_loss']:.2f}%")
            
            with col2:
                st.subheader("📈 Risk Metrics")
                st.write(f"**Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}")
                st.write(f"**Max Drawdown:** {metrics['max_drawdown']:.2f}%")
                st.write(f"**Profit Factor:** {metrics['profit_factor']:.2f}")
                st.write(f"**Win Rate:** {metrics['win_rate']:.1f}%")
            
            with col3:
                st.subheader("📉 Trade Stats")
                st.write(f"**Total Trades:** {metrics['total_trades']}")
                st.write(f"**Avg Win:** {metrics['avg_win']:.2f}%")
                st.write(f"**Avg Loss:** {metrics['avg_loss']:.2f}%")
                win_loss_ratio = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
                st.write(f"**Win/Loss Ratio:** {win_loss_ratio:.2f}")
            
            # Cumulative returns chart
            if paper_trader.portfolio['daily_values']:
                daily_df = pd.DataFrame(paper_trader.portfolio['daily_values'])
                daily_df['cumulative_return'] = (
                    (daily_df['value'] - paper_trader.portfolio['initial_capital']) / 
                    paper_trader.portfolio['initial_capital'] * 100
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_df['datetime'],
                    y=daily_df['cumulative_return'],
                    mode='lines',
                    name='Cumulative Return %',
                    line=dict(color='green' if daily_df['cumulative_return'].iloc[-1] > 0 else 'red', width=2)
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="Cumulative Returns",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet")
    
    with tab5:
        # Market opportunities
        st.subheader("🎯 Current Market Opportunities")
        
        opportunities = paper_trader.evaluate_all_opportunities()
        
        if opportunities:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_score = st.slider(
                    "Minimum Score",
                    min_value=0,
                    max_value=100,
                    value=20,
                    step=5
                )
            
            with col2:
                show_positions = st.checkbox("Show Current Positions", value=False)
            
            with col3:
                max_opportunities = st.slider(
                    "Max Opportunities",
                    min_value=5,
                    max_value=20,
                    value=10
                )
            
            # Filter opportunities
            filtered_opps = [
                opp for opp in opportunities
                if opp['score'] >= min_score and (show_positions or not opp['in_position'])
            ][:max_opportunities]
            
            # Display opportunities
            if filtered_opps:
                opp_df = pd.DataFrame(filtered_opps)
                
                # Format for display
                display_opps = opp_df[[
                    'symbol', 'score', 'price', 'volume_ratio',
                    'momentum_1h', 'momentum_day', 'in_position'
                ]].copy()
                
                # Apply color coding
                def color_score(val):
                    if val >= 60:
                        return 'background-color: #90EE90'  # Light green
                    elif val >= 40:
                        return 'background-color: #FFFFE0'  # Light yellow
                    else:
                        return ''
                
                styled_opps = display_opps.style.applymap(
                    color_score,
                    subset=['score']
                ).format({
                    'price': '${:.2f}',
                    'volume_ratio': '{:.2f}x',
                    'momentum_1h': '{:+.2f}%',
                    'momentum_day': '{:+.2f}%',
                    'score': '{:.1f}'
                })
                
                st.dataframe(styled_opps, use_container_width=True)
                
                # Opportunity score distribution
                fig = px.bar(
                    opp_df.head(max_opportunities),
                    x='symbol',
                    y='score',
                    title='Opportunity Scores',
                    color='score',
                    color_continuous_scale='RdYlGn'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No opportunities meeting criteria")
        else:
            st.warning("No market data available")

else:
    # Not initialized
    st.info("👈 Please initialize the system using the sidebar controls")
    
    # Show some information
    st.markdown("""
    ### 📋 System Overview
    
    This paper trading module implements the **Dynamic Portfolio Optimizer** strategy with:
    
    - **Real-time AlgoLab data integration**
    - **Up to 10 concurrent positions**
    - **Dynamic position sizing (20-30%)**
    - **Portfolio rotation for maximum returns**
    - **3% stop loss with trailing stops**
    - **8% take profit targets**
    
    ### 🚀 Getting Started
    
    1. Click **Initialize System** in the sidebar
    2. Wait for AlgoLab connection
    3. Click **Start Trading** to begin auto-trading
    4. Monitor performance in real-time
    
    ### 📊 Key Features
    
    - Live portfolio tracking
    - Real-time position monitoring
    - Performance analytics
    - Trade history with detailed metrics
    - Market opportunity scanner
    """)

# Auto refresh
if st.session_state.auto_refresh and st.session_state.is_initialized:
    st.empty()
    import time
    time.sleep(st.session_state.refresh_interval)
    st.rerun()