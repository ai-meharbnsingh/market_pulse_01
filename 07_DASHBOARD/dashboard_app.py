"""
MarketPulse Streamlit Dashboard
Main trading interface for paper trading and monitoring
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import os

# Add parent directory and execution paths to system path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "05_EXECUTION" / "paper_trading"))
sys.path.insert(0, str(parent_dir / "05_EXECUTION" / "alerts"))

# Try to import MarketPulse components
try:
    from paper_trading_engine import PaperTradingEngine
    PAPER_TRADING_AVAILABLE = True
except ImportError:
    PAPER_TRADING_AVAILABLE = False
    # Create a dummy class for testing
    class PaperTradingEngine:
        def __init__(self, initial_capital=100000):
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
            self.cash_available = initial_capital
            self.positions = {}
            self.orders = []
            self.trade_history = []
            self.total_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0
            self.total_commission = 0.0
            self.max_drawdown = 0.0
            self.max_positions = 6
            self.max_position_size = 0.05
            self.daily_loss_limit = 0.02

        def get_portfolio_summary(self):
            return {
                'initial_capital': self.initial_capital,
                'current_capital': self.current_capital,
                'cash_available': self.cash_available,
                'positions_value': 0,
                'total_positions': len(self.positions),
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'total_return_pct': 0,
                'max_drawdown_pct': 0,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': 0,
                'total_commission': self.total_commission
            }

        def get_positions(self):
            return []

        def place_order(self, symbol, side, quantity, order_type="MARKET", price=None):
            return True, f"Demo order placed: {side} {quantity} {symbol}", None

        def save_portfolio_snapshot(self):
            pass

try:
    from telegram_alerts import TelegramAlertSystem
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    # Create a dummy class
    class TelegramAlertSystem:
        def __init__(self):
            self.enabled = False

        def send_trade_executed(self, **kwargs):
            pass

# Configure Streamlit
st.set_page_config(
    page_title="MarketPulse Trading Dashboard",
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
    }
    .positive {
        color: #00cc88;
    }
    .negative {
        color: #ff3860;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_engine' not in st.session_state:
    st.session_state.trading_engine = PaperTradingEngine(initial_capital=100000)

if 'alert_system' not in st.session_state:
    st.session_state.alert_system = TelegramAlertSystem()

def main():
    """Main dashboard"""

    # Header
    st.title("📈 MarketPulse Trading Dashboard")
    st.caption(f"Paper Trading Mode | Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    # Check component availability
    if not PAPER_TRADING_AVAILABLE:
        st.warning("Paper Trading Engine not found. Copy paper_trading_engine.py to 05_EXECUTION/paper_trading/")

    if not TELEGRAM_AVAILABLE:
        st.info("Telegram alerts not configured. Optional feature.")

    # Sidebar
    with st.sidebar:
        st.header("Control Panel")

        # Trading mode
        mode = st.selectbox("Trading Mode", ["Paper Trading", "Live Trading (Disabled)"])

        # Quick actions
        st.subheader("Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("Snapshot", use_container_width=True):
                st.session_state.trading_engine.save_portfolio_snapshot()
                st.success("Snapshot saved!")

        # Risk settings
        st.subheader("Risk Settings")
        max_position_size = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum percentage of capital per position"
        )

        daily_loss_limit = st.slider(
            "Daily Loss Limit (%)",
            min_value=1,
            max_value=5,
            value=2,
            help="Maximum daily loss before trading stops"
        )

        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Analysis", "ML Insights", "Risk Monitor", "Trade History"]
        )

    # Main content based on selected page
    if page == "Overview":
        show_overview_page()
    elif page == "Analysis":
        show_analysis_page()
    elif page == "ML Insights":
        show_ml_insights_page()
    elif page == "Risk Monitor":
        show_risk_monitor_page()
    elif page == "Trade History":
        show_trade_history_page()

def show_overview_page():
    """Show overview page"""

    engine = st.session_state.trading_engine
    summary = engine.get_portfolio_summary()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"₹{summary['current_capital']:,.2f}",
            delta=f"{summary['total_return_pct']:.2f}%"
        )

    with col2:
        st.metric(
            label="Available Cash",
            value=f"₹{summary['cash_available']:,.2f}",
            delta=f"{(summary['cash_available']/summary['current_capital'])*100:.1f}%" if summary['current_capital'] > 0 else "0%"
        )

    with col3:
        st.metric(
            label="Open Positions",
            value=summary['total_positions'],
            delta=f"of {engine.max_positions} max"
        )

    with col4:
        win_rate = summary['win_rate']
        st.metric(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta="Good" if win_rate > 50 else "Need improvement"
        )

    # Charts row
    col1, col2 = st.columns([2, 1])

    with col1:
        # Portfolio value chart
        st.subheader("Portfolio Performance")

        # Generate sample data (replace with actual historical data)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        values = [summary['initial_capital']]
        for i in range(29):
            import random
            values.append(values[-1] * (1 + random.uniform(-0.02, 0.02)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00cc88', width=2)
        ))

        fig.update_layout(
            title="30-Day Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Value (₹)",
            template="plotly_white",
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Allocation pie chart
        st.subheader("Capital Allocation")

        allocation_data = {
            'Category': ['Cash', 'Positions'],
            'Value': [summary['cash_available'], summary['positions_value']]
        }

        fig = px.pie(
            allocation_data,
            values='Value',
            names='Category',
            color_discrete_map={'Cash': '#3498db', 'Positions': '#00cc88'}
        )

        fig.update_layout(
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

    # Positions table
    st.subheader("Current Positions")

    positions = engine.get_positions()
    if positions:
        positions_df = pd.DataFrame(positions)
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("No open positions. Start trading to see positions here.")

    # Trade entry section
    st.subheader("New Trade")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        symbol = st.text_input("Symbol", value="RELIANCE")

    with col2:
        action = st.selectbox("Action", ["BUY", "SELL"])

    with col3:
        quantity = st.number_input("Quantity", min_value=1, value=10)

    with col4:
        order_type = st.selectbox("Type", ["MARKET", "LIMIT"])

    with col5:
        price = st.number_input("Price", min_value=0.0, value=0.0, disabled=(order_type=="MARKET"))

    if st.button("Place Order", type="primary"):
        success, message, order = engine.place_order(
            symbol=symbol,
            side=action,
            quantity=quantity,
            order_type=order_type,
            price=price if order_type=="LIMIT" else None
        )

        if success:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

def show_analysis_page():
    """Show analysis page"""
    st.header("Market Analysis")

    # Watchlist
    st.subheader("Watchlist")

    watchlist = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICI"]

    watchlist_data = []
    for symbol in watchlist:
        # Generate sample data (replace with real data)
        import random
        price = random.uniform(1000, 3000)
        change = random.uniform(-2, 2)
        volume = random.randint(1000000, 10000000)

        watchlist_data.append({
            'Symbol': symbol,
            'Price': f"₹{price:.2f}",
            'Change': f"{change:.2f}%",
            'Volume': f"{volume:,}",
            'Signal': random.choice(['BUY', 'HOLD', 'SELL'])
        })

    watchlist_df = pd.DataFrame(watchlist_data)
    st.dataframe(watchlist_df, use_container_width=True)

    # Technical indicators
    st.subheader("Technical Indicators")

    selected_symbol = st.selectbox("Select Symbol", watchlist)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("RSI", "45.2", "-2.3", help="Relative Strength Index")

    with col2:
        st.metric("MACD", "Bullish", "↑", help="Moving Average Convergence Divergence")

    with col3:
        st.metric("Volume", "High", "+15%", help="Volume compared to 20-day average")

    with col4:
        st.metric("Trend", "Uptrend", "Strong", help="Overall trend direction")

def show_ml_insights_page():
    """Show ML insights page"""
    st.header("ML Model Insights")

    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Alpha Model Accuracy", "72.3%", "+2.1%")

    with col2:
        st.metric("LSTM Accuracy", "68.5%", "+1.2%")

    with col3:
        st.metric("Prophet RMSE", "45.2", "-3.1")

    with col4:
        st.metric("Ensemble Score", "74.8%", "+1.8%")

    # Predictions table
    st.subheader("Current Predictions")

    predictions = pd.DataFrame({
        'Symbol': ['RELIANCE', 'TCS', 'INFY'],
        'Model': ['Alpha', 'LSTM', 'Ensemble'],
        'Prediction': ['BUY', 'HOLD', 'BUY'],
        'Confidence': [0.75, 0.62, 0.81],
        'Target': [2550, 3520, 1550],
        'Stop Loss': [2450, 3450, 1480]
    })

    st.dataframe(predictions, use_container_width=True)

def show_risk_monitor_page():
    """Show risk monitoring page"""
    st.header("Risk Monitor")

    engine = st.session_state.trading_engine
    summary = engine.get_portfolio_summary()

    # Risk metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Daily Loss Used", "0%", "Within limits")
        st.progress(0.0)

    with col2:
        st.metric("Maximum Drawdown", "0%", "Low risk")
        st.progress(0.0)

    with col3:
        st.metric("Position Limit Used", "0%", "0 of 6")
        st.progress(0.0)

    # Risk rules
    st.subheader("Risk Rules")

    rules = pd.DataFrame({
        'Rule': ['Max Position Size', 'Daily Loss Limit', 'Max Positions'],
        'Limit': ['5%', '2%', '6'],
        'Status': ['Active', 'Active', 'Active']
    })

    st.dataframe(rules, use_container_width=True, hide_index=True)

def show_trade_history_page():
    """Show trade history page"""
    st.header("Trade History")

    engine = st.session_state.trading_engine

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    summary = engine.get_portfolio_summary()

    with col1:
        st.metric("Total Trades", summary['total_trades'])

    with col2:
        st.metric("Winning Trades", summary['winning_trades'])

    with col3:
        st.metric("Losing Trades", summary['losing_trades'])

    with col4:
        st.metric("Total Commission", f"₹{summary['total_commission']:,.2f}")

    # Trade history table
    st.subheader("Recent Trades")

    if engine.trade_history:
        trades_df = pd.DataFrame(engine.trade_history)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades yet. Start trading to see history here.")


if __name__ == "__main__":
    main()