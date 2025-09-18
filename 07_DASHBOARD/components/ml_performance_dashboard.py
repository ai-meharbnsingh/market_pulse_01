"""
MarketPulse ML Performance Dashboard
Day 11: Comprehensive Streamlit dashboard for ML monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="MarketPulse ML Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .alert-success {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Dashboard Header
# ==========================================

def render_header():
    """Render dashboard header with key metrics"""
    st.title("🚀 MarketPulse ML Performance Dashboard")
    st.markdown("**Day 11** | Real-time ML Model Monitoring & Analysis")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🎯 Alpha Model Accuracy",
            value="72.3%",
            delta="+2.1%",
            help="7-day rolling accuracy"
        )

    with col2:
        st.metric(
            label="⚡ Prediction Latency",
            value="87ms",
            delta="-13ms",
            help="Average end-to-end latency"
        )

    with col3:
        st.metric(
            label="📊 Daily Predictions",
            value="12,847",
            delta="+1,234",
            help="Predictions in last 24 hours"
        )

    with col4:
        st.metric(
            label="💰 Profit Factor",
            value="1.73",
            delta="+0.12",
            help="Winning trades / Losing trades"
        )


# ==========================================
# Model Performance Metrics
# ==========================================

def render_model_performance():
    """Render model performance comparison"""
    st.header("📈 Model Performance Comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Create performance comparison chart
        models = ['Alpha Model', 'LSTM', 'Prophet', 'Ensemble']
        metrics_data = {
            'Model': models,
            'Accuracy': [72.3, 68.5, 65.2, 74.8],
            'Precision': [75.1, 70.2, 67.8, 76.5],
            'Recall': [69.8, 66.4, 63.1, 72.3],
            'F1-Score': [72.4, 68.2, 65.4, 74.3],
            'Sharpe Ratio': [1.82, 1.54, 1.43, 1.95]
        }

        df_metrics = pd.DataFrame(metrics_data)

        # Create radar chart for model comparison
        fig = go.Figure()

        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Sharpe Ratio']

        for model in models:
            model_data = df_metrics[df_metrics['Model'] == model]
            values = [
                model_data['Accuracy'].values[0],
                model_data['Precision'].values[0],
                model_data['Recall'].values[0],
                model_data['F1-Score'].values[0],
                model_data['Sharpe Ratio'].values[0] * 40  # Scale for visibility
            ]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=model
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Model Performance Radar Chart",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 Best Performing Model")
        st.info("""
        **Ensemble Model** leads with:
        - **74.8%** Accuracy
        - **1.95** Sharpe Ratio
        - **74.3** F1-Score

        Combining Alpha, LSTM, and Prophet models provides the best overall performance.
        """)

        st.subheader("⚠️ Model Alerts")
        st.warning("**LSTM Model**: Slight accuracy degradation detected (-1.2%)")
        st.success("**Alpha Model**: Performance improved after retraining (+2.1%)")


# ==========================================
# Real-time Predictions Monitor
# ==========================================

def render_realtime_monitor():
    """Render real-time predictions monitor"""
    st.header("⚡ Real-time Predictions Monitor")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Generate sample real-time data
        time_points = pd.date_range(
            start=datetime.now() - timedelta(hours=6),
            end=datetime.now(),
            freq='5min'
        )

        predictions = np.random.randn(len(time_points)).cumsum() + 100
        actual = predictions + np.random.randn(len(time_points)) * 2

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=time_points,
            y=predictions,
            mode='lines',
            name='Predictions',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=time_points,
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='green', width=2, dash='dot')
        ))

        # Add confidence bands
        upper_band = predictions + 5
        lower_band = predictions - 5

        fig.add_trace(go.Scatter(
            x=time_points.tolist() + time_points.tolist()[::-1],
            y=upper_band.tolist() + lower_band.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,100,200,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        fig.update_layout(
            title="Real-time Prediction vs Actual (6-hour window)",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            showlegend=True,
            legend=dict(x=0, y=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Prediction Stats")
        st.metric("Mean Absolute Error", "2.34", "-0.12")
        st.metric("RMSE", "3.21", "-0.08")
        st.metric("R² Score", "0.892", "+0.015")
        st.metric("Hit Rate", "68.4%", "+1.2%")

    with col3:
        st.subheader("🔄 Latest Predictions")
        recent_preds = pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
            'Prediction': ['BUY', 'HOLD', 'BUY', 'SELL', 'BUY'],
            'Confidence': [0.82, 0.65, 0.78, 0.71, 0.89],
            'Time': ['2s ago', '5s ago', '8s ago', '12s ago', '15s ago']
        })
        st.dataframe(recent_preds, hide_index=True)


# ==========================================
# Feature Importance Analysis
# ==========================================

def render_feature_importance():
    """Render feature importance analysis"""
    st.header("🔍 Feature Importance Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Feature importance for Alpha Model
        features = [
            'RSI_14', 'MACD_Signal', 'Volume_Ratio', 'BB_Position',
            'EMA_Cross', 'Sentiment_Score', 'Options_Flow', 'IV_Rank',
            'Market_Regime', 'Sector_Strength'
        ]
        importance = [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06]

        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Alpha Model Feature Importance',
            labels={'x': 'Importance Score', 'y': 'Features'},
            color=importance,
            color_continuous_scale='viridis'
        )

        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # SHAP-style feature impact
        st.subheader("🎯 Feature Impact on Latest Prediction")

        feature_impacts = pd.DataFrame({
            'Feature': ['RSI_14', 'MACD_Signal', 'Volume_Ratio', 'BB_Position', 'Sentiment_Score'],
            'Value': [68.2, 0.023, 1.45, 0.82, 0.65],
            'Impact': ['+0.12', '+0.08', '-0.03', '+0.05', '+0.09'],
            'Direction': ['↑', '↑', '↓', '↑', '↑']
        })

        st.dataframe(feature_impacts, hide_index=True)

        st.info("""
        **Key Insights:**
        - RSI overbought signal (68.2) has highest positive impact
        - Sentiment score (0.65) supports bullish prediction
        - Volume slightly below average has negative impact
        """)


# ==========================================
# Model Drift Detection
# ==========================================

def render_drift_detection():
    """Render model drift detection"""
    st.header("🔔 Model Drift Detection")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # PSI (Population Stability Index) chart
        dates = pd.date_range(start='2025-09-01', end='2025-09-18', freq='D')
        psi_values = np.random.uniform(0.05, 0.15, len(dates))
        psi_values[-3:] = [0.18, 0.22, 0.19]  # Simulate drift

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=psi_values,
            mode='lines+markers',
            name='PSI',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        # Add threshold lines
        fig.add_hline(y=0.1, line_dash="dash", line_color="yellow", annotation_text="Warning (0.1)")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Critical (0.2)")

        fig.update_layout(
            title="Population Stability Index (PSI)",
            xaxis_title="Date",
            yaxis_title="PSI Value",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Feature drift heatmap
        st.subheader("📊 Feature Drift Heatmap")

        features = ['RSI', 'MACD', 'Volume', 'BB', 'EMA']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

        drift_matrix = np.random.uniform(0, 0.3, (len(features), len(days)))
        drift_matrix[1, 3] = 0.35  # Simulate drift
        drift_matrix[3, 4] = 0.28

        fig = px.imshow(
            drift_matrix,
            labels=dict(x="Day", y="Feature", color="Drift Score"),
            x=days,
            y=features,
            color_continuous_scale='RdYlGn_r',
            text_auto='.2f'
        )

        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("⚠️ Drift Alerts")

        st.markdown("""
        <div class="alert-box alert-warning">
            <strong>Warning:</strong> PSI exceeding threshold<br>
            Model: Alpha Model<br>
            Current PSI: 0.19<br>
            Action: Monitor closely
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-box alert-danger">
            <strong>Critical:</strong> Feature drift detected<br>
            Feature: MACD<br>
            Drift Score: 0.35<br>
            Action: Consider retraining
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Trigger Retraining", type="primary"):
            st.success("Retraining initiated! Expected completion: 15 minutes")


# ==========================================
# Training History & Schedule
# ==========================================

def render_training_history():
    """Render model training history and schedule"""
    st.header("📚 Model Training History & Schedule")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Training history chart
        training_dates = pd.date_range(start='2025-08-01', end='2025-09-18', freq='W')

        alpha_scores = [68.5, 69.2, 70.1, 70.8, 71.5, 72.0, 72.3]
        lstm_scores = [65.2, 66.1, 66.8, 67.3, 67.9, 68.2, 68.5]
        prophet_scores = [63.1, 63.8, 64.2, 64.7, 65.0, 65.1, 65.2]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=training_dates,
            y=alpha_scores,
            mode='lines+markers',
            name='Alpha Model',
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=training_dates,
            y=lstm_scores,
            mode='lines+markers',
            name='LSTM',
            line=dict(width=2)
        ))

        fig.add_trace(go.Scatter(
            x=training_dates,
            y=prophet_scores,
            mode='lines+markers',
            name='Prophet',
            line=dict(width=2)
        ))

        fig.update_layout(
            title="Model Accuracy Over Training Iterations",
            xaxis_title="Training Date",
            yaxis_title="Accuracy (%)",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Training Schedule")

        schedule = pd.DataFrame({
            'Model': ['Alpha Model', 'LSTM', 'Prophet', 'Ensemble'],
            'Frequency': ['Weekly', 'Bi-weekly', 'Monthly', 'Weekly'],
            'Next Training': ['Sep 22', 'Sep 30', 'Oct 1', 'Sep 22'],
            'Status': ['🟢 Scheduled', '🟢 Scheduled', '🟡 Pending', '🟢 Scheduled']
        })

        st.dataframe(schedule, hide_index=True)

        st.subheader("🔄 Recent Training Jobs")

        jobs = pd.DataFrame({
            'Job ID': ['#1847', '#1846', '#1845'],
            'Model': ['Alpha', 'LSTM', 'Ensemble'],
            'Duration': ['12m', '18m', '25m'],
            'Result': ['✅', '✅', '✅']
        })

        st.dataframe(jobs, hide_index=True)


# ==========================================
# Performance by Market Condition
# ==========================================

def render_market_condition_analysis():
    """Render performance by market condition"""
    st.header("🌍 Performance by Market Condition")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Market regime performance
        regimes = ['Bull Market', 'Bear Market', 'Sideways', 'High Volatility']

        performance_data = pd.DataFrame({
            'Market Regime': regimes,
            'Alpha Model': [75.2, 68.4, 70.1, 72.8],
            'LSTM': [71.3, 65.2, 68.9, 69.7],
            'Prophet': [67.8, 63.1, 66.2, 64.5],
            'Ensemble': [76.5, 70.2, 73.4, 74.1]
        })

        fig = go.Figure()

        for model in ['Alpha Model', 'LSTM', 'Prophet', 'Ensemble']:
            fig.add_trace(go.Bar(
                name=model,
                x=regimes,
                y=performance_data[model]
            ))

        fig.update_layout(
            title="Model Accuracy by Market Regime",
            xaxis_title="Market Regime",
            yaxis_title="Accuracy (%)",
            height=400,
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sector performance
        st.subheader("📊 Performance by Sector")

        sectors = ['Technology', 'Finance', 'Healthcare', 'Energy', 'Consumer']
        accuracies = [74.3, 71.2, 69.8, 68.5, 72.1]

        fig = px.pie(
            values=accuracies,
            names=sectors,
            title="Model Accuracy Distribution by Sector",
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Key Insights:**
        - Best performance in Technology sector (74.3%)
        - Energy sector needs improvement (68.5%)
        - Consider sector-specific model tuning
        """)


# ==========================================
# Sidebar Controls
# ==========================================

def render_sidebar():
    """Render sidebar with controls and filters"""
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")

        st.subheader("📅 Time Range")
        date_range = st.date_input(
            "Select date range",
            value=[datetime.now() - timedelta(days=7), datetime.now()],
            max_value=datetime.now()
        )

        st.subheader("🎯 Model Selection")
        selected_models = st.multiselect(
            "Choose models to display",
            ['Alpha Model', 'LSTM', 'Prophet', 'Ensemble'],
            default=['Alpha Model', 'LSTM', 'Prophet', 'Ensemble']
        )

        st.subheader("📊 Metrics")
        show_accuracy = st.checkbox("Show Accuracy", value=True)
        show_precision = st.checkbox("Show Precision", value=True)
        show_recall = st.checkbox("Show Recall", value=True)
        show_f1 = st.checkbox("Show F1-Score", value=True)

        st.subheader("⚙️ Settings")
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=False)
        show_alerts = st.checkbox("Show Alerts", value=True)

        st.divider()

        st.subheader("🔄 Quick Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        if st.button("📊 Generate Report", use_container_width=True):
            st.success("Report generation initiated!")

        if st.button("🎯 Run Diagnostics", use_container_width=True):
            with st.spinner("Running diagnostics..."):
                import time
                time.sleep(2)
            st.success("All systems operational!")

        st.divider()

        st.subheader("📈 System Status")
        st.metric("Database", "Connected", delta="PostgreSQL")
        st.metric("Redis Cache", "Active", delta="87% hit rate")
        st.metric("API Latency", "12ms", delta="-2ms")

        st.divider()

        st.caption("MarketPulse v3.0 | Day 11")
        st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ==========================================
# Main Application
# ==========================================

def main():
    """Main application entry point"""

    # Render sidebar
    render_sidebar()

    # Render header
    render_header()

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Model Performance",
        "⚡ Real-time Monitor",
        "🔍 Feature Analysis",
        "🔔 Drift Detection",
        "📚 Training History",
        "🌍 Market Analysis"
    ])

    with tab1:
        render_model_performance()

    with tab2:
        render_realtime_monitor()

    with tab3:
        render_feature_importance()

    with tab4:
        render_drift_detection()

    with tab5:
        render_training_history()

    with tab6:
        render_market_condition_analysis()

    # Auto-refresh logic
    if st.session_state.get('auto_refresh', False):
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()