"""Enhanced Streamlit frontend with rich visualizations for Chronos."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional


# Page config
st.set_page_config(
    page_title='Chronos - Productivity Forecasting',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# API base URL
API_BASE = st.sidebar.text_input('API Base URL', value='http://localhost:8000')


@st.cache_data
def load_csv(path: str = 'data/sample_timeseries.csv') -> pd.DataFrame:
    """Load time-series data."""
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def call_api(endpoint: str, payload: dict) -> Optional[dict]:
    """Call Chronos API endpoint."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None


def plot_forecast_curve(historical: List[float], forecast: List[float], confidence_intervals: Optional[List[Dict]] = None):
    """Plot forecast curve with confidence intervals."""
    fig = go.Figure()
    
    # Historical data
    hist_x = list(range(len(historical)))
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=historical,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    forecast_x = list(range(len(historical), len(historical) + len(forecast)))
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence intervals
    if confidence_intervals:
        lower = [ci.get('lower', f - 5) for f, ci in zip(forecast, confidence_intervals)]
        upper = [ci.get('upper', f + 5) for f, ci in zip(forecast, confidence_intervals)]
        
        fig.add_trace(go.Scatter(
            x=forecast_x + forecast_x[::-1],
            y=upper + lower[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title='Productivity Forecast',
        xaxis_title='Time Step',
        yaxis_title='Productivity Score',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_radar_chart(features: Dict[str, float]):
    """Plot behavioral fingerprint as radar chart."""
    categories = list(features.keys())
    values = list(features.values())
    
    # Normalize values to 0-1 for better visualization
    max_val = max(values) if values else 1.0
    normalized_values = [v / max_val if max_val > 0 else 0 for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],  # Close the loop
        theta=categories + [categories[0]],
        fill='toself',
        name='Behavioral Profile',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title='Behavioral Fingerprint',
        height=400
    )
    
    return fig


def plot_heatmap(series: pd.Series):
    """Plot daily productivity heatmap."""
    # Resample to hourly
    hourly = series.resample('1H').mean()
    
    # Create DataFrame with day of week and hour
    df_heatmap = pd.DataFrame({
        'value': hourly.values,
        'day_of_week': hourly.index.dayofweek,
        'hour': hourly.index.hour
    })
    
    # Pivot for heatmap
    heatmap_data = df_heatmap.pivot_table(
        values='value',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(24)),
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        colorscale='Viridis',
        colorbar=dict(title='Productivity')
    ))
    
    fig.update_layout(
        title='Daily Productivity Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )
    
    return fig


def plot_trend_analysis(series: pd.Series, trend: str, trend_strength: float):
    """Plot trend analysis."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Series with Trend', 'Trend Classification'),
        vertical_spacing=0.15
    )
    
    # Time series
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Productivity',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Trend classification
    trend_colors = {
        'rising': 'green',
        'declining': 'red',
        'plateauing': 'orange',
        'recovering': 'blue'
    }
    
    fig.add_trace(
        go.Bar(
            x=[trend],
            y=[trend_strength],
            name='Trend Strength',
            marker_color=trend_colors.get(trend, 'gray')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Trend Analysis',
        height=600,
        showlegend=True
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.title('📊 Chronos - AI-Driven Productivity Forecasting')
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load data
    data_file = st.sidebar.file_uploader("Upload CSV", type=['csv'], help="CSV with columns: timestamp, series_id, value")
    
    if data_file:
        df = pd.read_csv(data_file, parse_dates=['timestamp'])
    else:
        df = load_csv()
    
    if df.empty:
        st.warning("Please upload a CSV file or ensure data/sample_timeseries.csv exists")
        return
    
    # Series selection
    series_ids = df['series_id'].unique().tolist()
    selected_series = st.sidebar.selectbox('Select Series', series_ids)
    
    # Filter data
    sdf = df[df['series_id'] == selected_series].set_index('timestamp')
    values = sdf['value'].astype(float).dropna()
    
    if len(values) < 60:
        st.error("Series too short. Need at least 60 data points.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecast", "🔍 Patterns", "💡 Recommendations", "📊 Analytics", "🔬 Explainability"
    ])
    
    # Tab 1: Forecast
    with tab1:
        st.header("Productivity Forecast")
        
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Forecast Horizon", 1, 30, 7)
        with col2:
            model_type = st.selectbox("Model Type", ["ensemble", "lstm", "tcn", "transformer"])
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                window = values.tail(60).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "horizon": horizon,
                    "model_type": model_type
                }
                
                result = call_api("/forecast", payload)
                
                if result:
                    st.success(f"Forecast generated using {result['model']} model")
                    
                    # Plot
                    fig = plot_forecast_curve(
                        historical=window,
                        forecast=result['predictions'],
                        confidence_intervals=result.get('confidence_intervals')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latency", f"{result['latency_ms']:.2f} ms")
                    with col2:
                        st.metric("Forecast Range", f"{min(result['predictions']):.2f} - {max(result['predictions']):.2f}")
                    with col3:
                        avg_forecast = np.mean(result['predictions'])
                        st.metric("Average Forecast", f"{avg_forecast:.2f}")
    
    # Tab 2: Patterns
    with tab2:
        st.header("Pattern Detection")
        
        window_size = st.slider("Window Size", 10, 200, 60)
        
        if st.button("Detect Patterns", type="primary"):
            with st.spinner("Analyzing patterns..."):
                window = values.tail(min(len(values), window_size * 2)).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "window_size": window_size
                }
                
                result = call_api("/patterns", payload)
                
                if result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Trend", result['trend'].upper())
                        st.metric("Trend Strength", f"{result['trend_strength']:.2f}")
                        st.metric("Behavioral Phase", result.get('behavioral_phase', 'N/A').upper())
                    
                    with col2:
                        st.metric("Anomalies Detected", len(result.get('anomalies', [])))
                    
                    # Trend plot
                    fig = plot_trend_analysis(
                        series=values,
                        trend=result['trend'],
                        trend_strength=result['trend_strength']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Features
                    st.subheader("Behavioral Features")
                    features_df = pd.DataFrame([result['features']]).T
                    features_df.columns = ['Value']
                    st.dataframe(features_df, use_container_width=True)
                    
                    # Anomalies
                    if result.get('anomalies'):
                        st.subheader("Detected Anomalies")
                        anomalies_df = pd.DataFrame(result['anomalies'])
                        st.dataframe(anomalies_df, use_container_width=True)
    
    # Tab 3: Recommendations
    with tab3:
        st.header("Productivity Recommendations")
        
        num_recs = st.slider("Number of Recommendations", 1, 10, 3)
        multi_day = st.checkbox("Generate Multi-Day Plan", value=True)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                window = values.tail(60).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "num_recommendations": num_recs,
                    "multi_day": multi_day
                }
                
                result = call_api("/recommend", payload)
                
                if result:
                    recommendations = result.get('recommendations', [])
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"Recommendation {i}: {rec.get('category', 'N/A').upper()}", expanded=i==1):
                            st.write(f"**Priority:** {rec.get('priority', 'medium').upper()}")
                            st.write(f"**Text:** {rec.get('text', 'N/A')}")
                            
                            if rec.get('reasoning'):
                                st.write(f"**Reasoning:** {rec.get('reasoning')}")
                            
                            if rec.get('actionable_steps'):
                                st.write("**Actionable Steps:**")
                                for step in rec['actionable_steps']:
                                    st.write(f"- {step}")
                            
                            if rec.get('multi_day_plan') and multi_day:
                                st.write("**Multi-Day Plan:**")
                                plan_df = pd.DataFrame(rec['multi_day_plan'])
                                st.dataframe(plan_df, use_container_width=True)
    
    # Tab 4: Analytics
    with tab4:
        st.header("Productivity Analytics")
        
        # Time series plot
        st.subheader("Time Series Overview")
        fig = px.line(values.reset_index(), x='timestamp', y='value', title='Productivity Over Time')
        st.plotly_chart(fig, use_container_width=True)
        
        # Get features for radar chart
        if st.button("Generate Behavioral Profile", type="primary"):
            with st.spinner("Analyzing behavioral profile..."):
                window = values.tail(60).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "window_size": 60
                }
                
                result = call_api("/patterns", payload)
                
                if result and result.get('features'):
                    # Radar chart
                    fig = plot_radar_chart(result['features'])
                    st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Daily Productivity Patterns")
        if len(values) > 24 * 7:  # At least a week of data
            fig = plot_heatmap(values)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least a week of data for heatmap")
        
        # Statistics
        st.subheader("Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{values.mean():.2f}")
        with col2:
            st.metric("Std Dev", f"{values.std():.2f}")
        with col3:
            st.metric("Min", f"{values.min():.2f}")
        with col4:
            st.metric("Max", f"{values.max():.2f}")
    
    # Tab 5: Explainability
    with tab5:
        st.header("Model Explainability")
        
        top_features = st.slider("Top Features", 5, 50, 10)
        
        if st.button("Explain Prediction", type="primary"):
            with st.spinner("Generating explanation..."):
                window = values.tail(60).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "top_features": top_features
                }
                
                result = call_api("/explain", payload)
                
                if result:
                    st.write(f"**Reasoning:** {result.get('reasoning', 'N/A')}")
                    
                    # Feature importance
                    if result.get('feature_importance'):
                        importance_df = pd.DataFrame(
                            list(result['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', key=abs, ascending=False)
                        
                        st.subheader("Feature Importance")
                        fig = px.bar(
                            importance_df.head(top_features),
                            x='Feature',
                            y='Importance',
                            title='Top Feature Importance'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.dataframe(importance_df, use_container_width=True)


if __name__ == '__main__':
    main()
