# File: src/dashboard/confluence_dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import sys
from pathlib import Path
from chart_components import create_enhanced_timeframe_chart, create_rsi_chart
import json

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_Framework_Core"))
sys.path.insert(0, str(project_root / "src" / "ai_trading"))

from ai_confluence_scorer import AIConfluenceScorer

st.set_page_config(
    page_title="MarketPulse - AI Confluence Dashboard",
    page_icon="📈",
    layout="wide"
)


# Cache the AI scorer to avoid reinitializing
@st.cache_resource
def get_ai_scorer():
    return AIConfluenceScorer()


def create_timeframe_chart(data_collection, symbol):
    """Create multi-timeframe price charts"""

    # Create subplots for different timeframes
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['1-Minute', '5-Minute', '15-Minute', '1-Hour', '4-Hour', 'Daily'],
        vertical_spacing=0.08,
        horizontal_spacing=0.05
    )

    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]

    for i, (tf, pos) in enumerate(zip(timeframes, positions)):
        if tf in data_collection:
            data = data_collection[tf].tail(50)  # Last 50 points

            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name=f"{tf} Price",
                    showlegend=False
                ),
                row=pos[0], col=pos[1]
            )

            # Add SMA lines
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        mode='lines',
                        line=dict(color='orange', width=1),
                        name=f"{tf} SMA20",
                        showlegend=False
                    ),
                    row=pos[0], col=pos[1]
                )

    fig.update_layout(
        title=f"{symbol} - Multi-Timeframe Analysis",
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_confluence_gauge(confluence_score, direction):
    """Create a gauge chart for confluence strength"""

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confluence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confluence Strength - {direction}"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_signal_table(signals):
    """Create a table showing signals from each timeframe"""

    df_signals = []
    for timeframe, signal in signals.items():
        df_signals.append({
            'Timeframe': timeframe.upper(),
            'Trend': signal['trend'],
            'RSI': f"{signal['rsi']:.1f}" if not pd.isna(signal['rsi']) else "N/A",
            'Volume': signal['volume'],
            'Price Change %': f"{signal['price_change_pct']:.2f}%" if not pd.isna(
                signal['price_change_pct']) else "N/A",
            'Weight': f"{signal['weight']:.2f}"
        })

    return pd.DataFrame(df_signals)


async def run_analysis(symbol, analysis_type):
    """Run the AI confluence analysis"""
    scorer = get_ai_scorer()
    result = await scorer.analyze_confluence(symbol, analysis_type)
    return result


def main():
    st.title("📈 MarketPulse - AI Confluence Dashboard")
    st.markdown("**Multi-Timeframe AI-Powered Trading Analysis**")

    # Sidebar controls
    st.sidebar.header("Analysis Configuration")

    # Symbol selection
    symbol = st.sidebar.text_input(
        "Stock Symbol",
        value="RELIANCE.NS",
        help="Enter NSE symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)"
    )

    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Trading Style",
        ["swing_trade", "day_trade", "long_term"],
        help="Choose your trading timeframe"
    )

    # Run analysis button
    if st.sidebar.button("🎯 Run AI Analysis", type="primary"):
        if symbol:
            with st.spinner(f"Running AI confluence analysis for {symbol}..."):
                try:
                    # Run the async analysis
                    result = asyncio.run(run_analysis(symbol, analysis_type))

                    if "error" in result:
                        st.error(f"Analysis failed: {result['error']}")
                        return

                    # Store result in session state
                    st.session_state['analysis_result'] = result
                    st.success("✅ Analysis completed!")

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    return
        else:
            st.warning("Please enter a stock symbol")

    # Display results if available
    if 'analysis_result' in st.session_state:
        result = st.session_state['analysis_result']

        # Main metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            basic_conf = result['basic_confluence']
            st.metric(
                "Direction",
                basic_conf['direction'],
                delta=f"{basic_conf['confluence_strength']:.1f}% strength"
            )

        with col2:
            if 'final_recommendation' in result:
                rec = result['final_recommendation']
                action = rec.get('action', 'N/A')
                confidence = rec.get('confidence', 'N/A')
                st.metric("Recommendation", action, delta=f"{confidence}% confidence")

        with col3:
            if 'ai_analysis' in result and 'trade_setup' in result['ai_analysis']:
                setup = result['ai_analysis']['trade_setup']
                quality = setup.get('quality_score', 'N/A')
                st.metric("Setup Quality", f"{quality}/10")

        with col4:
            if 'ai_analysis' in result and 'trade_setup' in result['ai_analysis']:
                setup = result['ai_analysis']['trade_setup']
                rr = setup.get('risk_reward_ratio', 'N/A')
                st.metric("Risk/Reward", f"{rr}:1" if rr != 'N/A' else 'N/A')

        # Confluence gauge
        st.subheader("📊 Confluence Strength")
        basic_conf = result['basic_confluence']
        gauge_fig = create_confluence_gauge(
            basic_conf['confluence_strength'],
            basic_conf['direction']
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        # Multi-timeframe charts
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']

            # Get the raw data (you'll need to modify your scorer to return this)
            st.subheader("📈 Multi-Timeframe Price Charts")

            # For now, let's create a placeholder
            st.info("📊 Interactive multi-timeframe charts coming in next update!")

            # RSI comparison chart
            st.subheader("📊 RSI Momentum Analysis")
            st.info("⚡ RSI comparison across all timeframes coming next!")


        # Timeframe signals table
        st.subheader("⏱️ Multi-Timeframe Signals")
        signals_df = create_signal_table(result['timeframe_signals'])
        st.dataframe(signals_df, use_container_width=True)

        # AI Analysis details
        if 'ai_analysis' in result and 'error' not in result['ai_analysis']:
            st.subheader("🧠 AI Analysis Details")
            ai = result['ai_analysis']

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Confluence Assessment:**")
                st.write(ai.get('confluence_assessment', 'N/A'))

                if 'key_levels' in ai:
                    st.write("**Key Levels:**")
                    levels = ai['key_levels']
                    if 'support' in levels and levels['support']:
                        st.write(f"Support: {levels['support']}")
                    if 'resistance' in levels and levels['resistance']:
                        st.write(f"Resistance: {levels['resistance']}")

            with col2:
                st.write("**Momentum Analysis:**")
                st.write(ai.get('momentum_analysis', 'N/A'))

                if 'red_flags' in ai and ai['red_flags']:
                    st.write("**⚠️ Red Flags:**")
                    for flag in ai['red_flags']:
                        st.write(f"• {flag}")

        # Final recommendation details
        if 'final_recommendation' in result:
            st.subheader("🎯 Trading Recommendation")
            rec = result['final_recommendation']

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Action:** {rec.get('action', 'N/A')}")
                st.write(f"**Confidence:** {rec.get('confidence', 'N/A')}%")
                st.write(f"**Time Horizon:** {rec.get('time_horizon', 'N/A')}")

            with col2:
                st.write("**Reasoning:**")
                st.write(rec.get('reasoning', 'N/A'))

                if 'key_catalysts' in rec and rec['key_catalysts']:
                    st.write("**Key Catalysts:**")
                    for catalyst in rec['key_catalysts']:
                        st.write(f"• {catalyst}")

        # Raw data expander
        with st.expander("📋 View Raw Analysis Data"):
            st.json(result)

    else:
        # Welcome message
        st.info(
            "👋 Welcome to MarketPulse AI Confluence Dashboard! Enter a stock symbol and click 'Run AI Analysis' to get started.")


if __name__ == "__main__":
    main()