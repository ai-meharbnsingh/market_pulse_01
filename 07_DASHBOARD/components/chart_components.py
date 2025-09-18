# File: src/dashboard/chart_components.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_enhanced_timeframe_chart(data_collection, symbol, signals):
    """Create enhanced multi-timeframe charts with signals overlay"""

    # Determine which timeframes we have
    available_tfs = list(data_collection.keys())

    if len(available_tfs) >= 4:
        # Create 2x2 grid for main timeframes
        main_tfs = ['15m', '1h', '4h', '1d'] if all(
            tf in available_tfs for tf in ['15m', '1h', '4h', '1d']) else available_tfs[:4]

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{tf.upper()} - {signals.get(tf, {}).get("trend", "N/A")} Trend' for tf in main_tfs],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )

        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for i, (tf, pos) in enumerate(zip(main_tfs, positions)):
            if tf in data_collection:
                data = data_collection[tf].tail(50)  # Last 50 candlesticks
                signal = signals.get(tf, {})

                # Add candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name=f"{tf} Price",
                        showlegend=False,
                        increasing=dict(fillcolor='green', line_color='darkgreen'),
                        decreasing=dict(fillcolor='red', line_color='darkred')
                    ),
                    row=pos[0], col=pos[1]
                )

                # Add moving averages if available
                if 'SMA_20' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['SMA_20'],
                            mode='lines',
                            line=dict(color='orange', width=2),
                            name=f"{tf} SMA20",
                            showlegend=False
                        ),
                        row=pos[0], col=pos[1]
                    )

        fig.update_layout(
            title=f"{symbol} - Multi-Timeframe Technical Analysis",
            height=700,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )

        return fig

    else:
        # Simple placeholder
        fig = go.Figure()
        fig.add_annotation(text="Chart data will appear here",
                           xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Chart Placeholder", height=400)
        return fig


def create_rsi_chart(data_collection, signals):
    """Create RSI comparison chart across timeframes"""

    fig = go.Figure()
    fig.add_annotation(text="RSI comparison chart will appear here",
                       xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False)
    fig.update_layout(title="RSI Comparison - All Timeframes", height=400)
    return fig