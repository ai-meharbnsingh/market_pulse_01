# 07_DASHBOARD/live_trading_dashboard.py
"""
Phase 3, Step 5: Live Trading Dashboard
Real-time monitoring and control dashboard for live trading system

Features:
- Real-time portfolio monitoring
- Live market data display
- Risk management alerts
- Order execution tracking
- Performance analytics
- System health monitoring
- Trading controls

Location: #07_DASHBOARD/live_trading_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import json
import sys
from pathlib import Path
import logging
import time

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "05_EXECUTION"))
sys.path.append(str(project_root / "04_RISK"))
sys.path.append(str(project_root / "06_DATA"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="MarketPulse Live Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


class LiveTradingDashboard:
    """Live trading dashboard controller"""

    def __init__(self):
        self.db_path = "marketpulse_production.db"

        # Initialize session state
        if 'trading_engine' not in st.session_state:
            st.session_state.trading_engine = None
        if 'risk_manager' not in st.session_state:
            st.session_state.risk_manager = None
        if 'market_data_fetcher' not in st.session_state:
            st.session_state.market_data_fetcher = None
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

    def initialize_components(self):
        """Initialize trading system components"""

        try:
            # Initialize trading engine
            if st.session_state.trading_engine is None:
                from live_trading_engine import LiveTradingEngine, TradingMode
                st.session_state.trading_engine = LiveTradingEngine(
                    db_path=self.db_path,
                    trading_mode=TradingMode.PAPER
                )

            # Initialize risk manager
            if st.session_state.risk_manager is None:
                from advanced_risk_management import AdvancedRiskManager
                st.session_state.risk_manager = AdvancedRiskManager(db_path=self.db_path)

            # Initialize market data fetcher
            if st.session_state.market_data_fetcher is None:
                try:
                    from live_market_data_fetcher import LiveMarketDataFetcher
                    st.session_state.market_data_fetcher = LiveMarketDataFetcher(db_path=self.db_path)
                except ImportError:
                    try:
                        from enhanced.data_fetcher import MarketDataFetcher
                        st.session_state.market_data_fetcher = MarketDataFetcher(db_path=self.db_path)
                    except ImportError:
                        st.error("No market data fetcher available")

            return True

        except Exception as e:
            st.error(f"Error initializing components: {e}")
            logger.error(f"Component initialization error: {e}")
            return False

    def connect_database(self):
        """Connect to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return None

    def get_portfolio_data(self):
        """Get current portfolio data"""

        try:
            if st.session_state.trading_engine:
                portfolio = st.session_state.trading_engine.get_portfolio()
                return {
                    'total_value': portfolio.total_value,
                    'available_cash': portfolio.available_cash,
                    'invested_value': portfolio.invested_value,
                    'unrealized_pnl': portfolio.unrealized_pnl,
                    'realized_pnl': portfolio.realized_pnl,
                    'day_pnl': portfolio.day_pnl,
                    'positions': {k: v.to_dict() for k, v in portfolio.positions.items()},
                    'last_updated': portfolio.last_updated.isoformat()
                }
            else:
                # Fallback to database query
                conn = self.connect_database()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT * FROM portfolios 
                        ORDER BY created_at DESC LIMIT 1
                    """)
                    portfolio_data = cursor.fetchone()
                    conn.close()

                    if portfolio_data:
                        return dict(portfolio_data)

                return {
                    'total_value': 0.0,
                    'available_cash': 0.0,
                    'invested_value': 0.0,
                    'unrealized_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'day_pnl': 0.0,
                    'positions': {},
                    'last_updated': datetime.now().isoformat()
                }

        except Exception as e:
            st.error(f"Error getting portfolio data: {e}")
            return {}

    def get_market_data(self, symbols=['AAPL', 'MSFT', 'GOOGL', 'TSLA'], days=30):
        """Get market data for dashboard"""

        try:
            conn = self.connect_database()
            if not conn:
                return pd.DataFrame()

            cursor = conn.cursor()

            # Get historical data
            placeholders = ','.join(['?' for _ in symbols])
            cursor.execute(f"""
                SELECT symbol, timestamp, close_price, volume
                FROM market_data 
                WHERE symbol IN ({placeholders})
                ORDER BY timestamp DESC
                LIMIT ?
            """, symbols + [days * len(symbols)])

            data = cursor.fetchall()
            conn.close()

            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            st.error(f"Error getting market data: {e}")
            return pd.DataFrame()

    def get_risk_metrics(self):
        """Get risk management metrics"""

        try:
            if st.session_state.risk_manager:
                portfolio_data = self.get_portfolio_data()
                metrics = st.session_state.risk_manager.calculate_portfolio_risk_metrics(portfolio_data)
                alerts = st.session_state.risk_manager.check_risk_alerts(portfolio_data)
                dashboard = st.session_state.risk_manager.get_risk_dashboard()

                return {
                    'metrics': metrics.to_dict() if metrics else {},
                    'alerts': [alert.to_dict() for alert in alerts],
                    'dashboard': dashboard
                }
            else:
                return {'metrics': {}, 'alerts': [], 'dashboard': {}}

        except Exception as e:
            st.error(f"Error getting risk metrics: {e}")
            return {'metrics': {}, 'alerts': [], 'dashboard': {}}

    def get_trading_activity(self, limit=50):
        """Get recent trading activity"""

        try:
            conn = self.connect_database()
            if not conn:
                return pd.DataFrame()

            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_id, symbol, trade_type, quantity, 
                       executed_price, timestamp, commission
                FROM trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))

            data = cursor.fetchall()
            conn.close()

            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                return pd.DataFrame()

        except Exception as e:
            st.error(f"Error getting trading activity: {e}")
            return pd.DataFrame()

    def render_header(self):
        """Render dashboard header"""

        st.title("📈 MarketPulse Live Trading Dashboard")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if st.button("🔄 Refresh", help="Refresh all data"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh

        with col3:
            st.write(f"**Status:** {'🟢 Live' if st.session_state.trading_engine else '🔴 Offline'}")

        with col4:
            st.write(f"**Mode:** Paper Trading")

        with col5:
            st.write(f"**Updated:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")

        st.markdown("---")

    def render_portfolio_overview(self, portfolio_data):
        """Render portfolio overview section"""

        st.subheader("💰 Portfolio Overview")

        col1, col2, col3, col4, col5 = st.columns(5)

        total_value = portfolio_data.get('total_value', 0)
        available_cash = portfolio_data.get('available_cash', 0)
        invested_value = portfolio_data.get('invested_value', 0)
        unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)
        day_pnl = portfolio_data.get('day_pnl', 0)

        with col1:
            st.metric(
                label="Total Value",
                value=f"${total_value:,.2f}",
                delta=f"${day_pnl:,.2f}" if day_pnl != 0 else None
            )

        with col2:
            st.metric(
                label="Available Cash",
                value=f"${available_cash:,.2f}",
                delta=f"{(available_cash / total_value * 100):.1f}%" if total_value > 0 else None
            )

        with col3:
            st.metric(
                label="Invested Value",
                value=f"${invested_value:,.2f}",
                delta=f"{(invested_value / total_value * 100):.1f}%" if total_value > 0 else None
            )

        with col4:
            st.metric(
                label="Unrealized P&L",
                value=f"${unrealized_pnl:,.2f}",
                delta=f"{(unrealized_pnl / invested_value * 100):.2f}%" if invested_value > 0 else None
            )

        with col5:
            st.metric(
                label="Day P&L",
                value=f"${day_pnl:,.2f}",
                delta=f"{(day_pnl / total_value * 100):.2f}%" if total_value > 0 else None
            )

    def render_positions_table(self, portfolio_data):
        """Render positions table"""

        st.subheader("📊 Current Positions")

        positions = portfolio_data.get('positions', {})

        if positions:
            position_data = []
            for symbol, pos in positions.items():
                position_data.append({
                    'Symbol': symbol,
                    'Quantity': pos.get('quantity', 0),
                    'Avg Cost': f"${pos.get('avg_cost', 0):.2f}",
                    'Market Value': f"${pos.get('market_value', 0):,.2f}",
                    'Unrealized P&L': f"${pos.get('unrealized_pnl', 0):,.2f}",
                    'Weight': f"{(pos.get('market_value', 0) / portfolio_data.get('total_value', 1) * 100):.1f}%"
                })

            df = pd.DataFrame(position_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No positions currently held")

    def render_market_data_charts(self, market_data):
        """Render market data charts"""

        st.subheader("📈 Market Data")

        if not market_data.empty:
            # Price chart
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price Movement', 'Volume'),
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}]],
                vertical_spacing=0.1
            )

            symbols = market_data['symbol'].unique()
            colors = px.colors.qualitative.Set1

            for i, symbol in enumerate(symbols):
                symbol_data = market_data[market_data['symbol'] == symbol].sort_values('timestamp')

                # Price line
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['timestamp'],
                        y=symbol_data['close_price'],
                        mode='lines',
                        name=f'{symbol} Price',
                        line=dict(color=colors[i % len(colors)]),
                        hovertemplate=f'{symbol}<br>Price: $%{{y:.2f}}<br>%{{x}}<extra></extra>'
                    ),
                    row=1, col=1
                )

                # Volume bars
                fig.add_trace(
                    go.Bar(
                        x=symbol_data['timestamp'],
                        y=symbol_data['volume'],
                        name=f'{symbol} Volume',
                        marker_color=colors[i % len(colors)],
                        opacity=0.7,
                        hovertemplate=f'{symbol}<br>Volume: %{{y:,}}<br>%{{x}}<extra></extra>'
                    ),
                    row=2, col=1
                )

            fig.update_layout(
                height=600,
                title="Market Data Overview",
                showlegend=True,
                hovermode='x unified'
            )

            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No market data available")

    def render_risk_management(self, risk_data):
        """Render risk management section"""

        st.subheader("⚠️ Risk Management")

        metrics = risk_data.get('metrics', {})
        alerts = risk_data.get('alerts', [])

        # Risk metrics
        if metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                risk_level = metrics.get('risk_level', 'UNKNOWN')
                risk_color = {
                    'LOW': 'green',
                    'MEDIUM': 'orange',
                    'HIGH': 'red',
                    'CRITICAL': 'darkred'
                }.get(risk_level, 'gray')

                st.markdown(f"**Risk Level:** :{risk_color}[{risk_level}]")

            with col2:
                var_1_day = metrics.get('var_1_day', 0)
                st.metric("1-Day VaR", f"${var_1_day:,.2f}")

            with col3:
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            with col4:
                max_drawdown = metrics.get('max_drawdown', 0)
                st.metric("Max Drawdown", f"{max_drawdown:.2%}")

        # Risk alerts
        if alerts:
            st.warning(f"⚠️ {len(alerts)} Active Risk Alerts")

            alert_data = []
            for alert in alerts:
                alert_data.append({
                    'Severity': alert.get('severity', 'UNKNOWN'),
                    'Symbol': alert.get('symbol', 'Portfolio'),
                    'Message': alert.get('message', ''),
                    'Created': alert.get('created_at', '')[:19]
                })

            if alert_data:
                df = pd.DataFrame(alert_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.success("✅ No active risk alerts")

    def render_trading_activity(self, activity_data):
        """Render trading activity section"""

        st.subheader("📋 Recent Trading Activity")

        if not activity_data.empty:
            # Format the data for display
            display_data = activity_data.copy()
            display_data['executed_price'] = display_data['executed_price'].apply(lambda x: f"${x:.2f}")
            display_data['commission'] = display_data['commission'].apply(lambda x: f"${x:.2f}")
            display_data['timestamp'] = display_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(display_data, use_container_width=True)

            # Trading summary chart
            if len(display_data) > 1:
                fig = px.scatter(
                    activity_data,
                    x='timestamp',
                    y='executed_price',
                    color='trade_type',
                    size='quantity',
                    hover_data=['symbol', 'commission'],
                    title="Trading Activity Timeline"
                )

                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recent trading activity")

    def render_trading_controls(self):
        """Render trading control panel"""

        st.subheader("🎮 Trading Controls")

        with st.expander("Place New Order", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                symbol = st.text_input("Symbol", value="AAPL")
                side = st.selectbox("Side", ["BUY", "SELL"])
                order_type = st.selectbox("Order Type", ["MARKET", "LIMIT"])

            with col2:
                quantity = st.number_input("Quantity", min_value=1, value=10)
                if order_type == "LIMIT":
                    limit_price = st.number_input("Limit Price", min_value=0.01, value=100.00, step=0.01)
                else:
                    limit_price = None

            if st.button("Place Order"):
                if st.session_state.trading_engine:
                    try:
                        from live_trading_engine import OrderSide, OrderType

                        success, message, order_id = st.session_state.trading_engine.create_order(
                            symbol=symbol,
                            side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                            order_type=OrderType.MARKET if order_type == "MARKET" else OrderType.LIMIT,
                            quantity=quantity,
                            price=limit_price
                        )

                        if success:
                            st.success(f"✅ Order placed successfully: {order_id}")
                        else:
                            st.error(f"❌ Order failed: {message}")

                    except Exception as e:
                        st.error(f"❌ Error placing order: {e}")
                else:
                    st.error("❌ Trading engine not available")

        with st.expander("Open Orders", expanded=False):
            if st.session_state.trading_engine:
                open_orders = st.session_state.trading_engine.get_open_orders()

                if open_orders:
                    for order in open_orders:
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            st.write(f"{order.symbol} {order.side.value} {order.quantity} @ {order.price or 'MARKET'}")

                        with col2:
                            st.write(f"Status: {order.status.value}")

                        with col3:
                            if st.button(f"Cancel", key=f"cancel_{order.order_id}"):
                                success, message = st.session_state.trading_engine.cancel_order(order.order_id)
                                if success:
                                    st.success("Order cancelled")
                                    st.rerun()
                                else:
                                    st.error(f"Cancel failed: {message}")
                else:
                    st.info("No open orders")
            else:
                st.error("Trading engine not available")

    def run(self):
        """Main dashboard application"""

        # Initialize components
        if not self.initialize_components():
            st.error("Failed to initialize trading system components")
            return

        # Auto refresh
        if st.session_state.auto_refresh:
            if (datetime.now() - st.session_state.last_refresh).seconds > 30:
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        # Render header
        self.render_header()

        # Get data
        portfolio_data = self.get_portfolio_data()
        market_data = self.get_market_data()
        risk_data = self.get_risk_metrics()
        activity_data = self.get_trading_activity()

        # Main dashboard layout
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Portfolio", "Market Data", "Risk Management", "Trading Activity", "Controls"
        ])

        with tab1:
            self.render_portfolio_overview(portfolio_data)
            st.markdown("---")
            self.render_positions_table(portfolio_data)

        with tab2:
            self.render_market_data_charts(market_data)

        with tab3:
            self.render_risk_management(risk_data)

        with tab4:
            self.render_trading_activity(activity_data)

        with tab5:
            self.render_trading_controls()

        # Sidebar with system status
        with st.sidebar:
            st.header("System Status")

            status_items = [
                ("Trading Engine", "🟢 Online" if st.session_state.trading_engine else "🔴 Offline"),
                ("Risk Manager", "🟢 Online" if st.session_state.risk_manager else "🔴 Offline"),
                ("Market Data", "🟢 Online" if st.session_state.market_data_fetcher else "🔴 Offline"),
                ("Database", "🟢 Connected" if Path(self.db_path).exists() else "🔴 Disconnected")
            ]

            for item, status in status_items:
                st.write(f"**{item}:** {status}")

            st.markdown("---")

            st.header("Quick Stats")
            total_value = portfolio_data.get('total_value', 0)
            unrealized_pnl = portfolio_data.get('unrealized_pnl', 0)
            positions_count = len(portfolio_data.get('positions', {}))

            st.metric("Portfolio Value", f"${total_value:,.2f}")
            st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")
            st.metric("Active Positions", positions_count)

            # Risk alerts count
            risk_alerts = risk_data.get('alerts', [])
            if risk_alerts:
                st.warning(f"⚠️ {len(risk_alerts)} Risk Alerts")
            else:
                st.success("✅ No Risk Alerts")


def main():
    """Main function to run the dashboard"""

    # Create dashboard instance
    dashboard = LiveTradingDashboard()

    # Run the dashboard
    dashboard.run()


if __name__ == "__main__":
    main()