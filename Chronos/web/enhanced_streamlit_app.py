"""
Enhanced Streamlit frontend with rich visualisations for Chronos.

This is the full-featured Chronos web dashboard, providing five tabs:

  1. Forecast        — Select a model type and horizon, then visualise the
                       predicted curve with optional confidence intervals.
  2. Patterns        — Detect trends, behavioural phase, and anomalies via the
                       /patterns API; view extracted behavioural features.
  3. Recommendations — Get personalised coaching advice and multi-day action
                       plans from the /recommend API.
  4. Analytics       — Interactive time-series overview, daily-productivity
                       heatmap, radar-chart "behavioural fingerprint", and
                       summary statistics.
  5. Explainability  — Call the /explain API to see feature importances and
                       human-readable reasoning behind a prediction.

All API calls are made to the FastAPI backend (default http://localhost:8000).

Run with:
    streamlit run web/enhanced_streamlit_app.py
"""

import streamlit as st                            # Core Streamlit framework
import pandas as pd                               # DataFrame / Series operations
import numpy as np                                # Numerical helpers (mean, etc.)
import plotly.graph_objects as go                  # Low-level Plotly figures (Scatter, Heatmap, etc.)
import plotly.express as px                        # High-level Plotly convenience API
from plotly.subplots import make_subplots          # Multi-row subplot layouts
import requests                                    # HTTP client for calling FastAPI endpoints
from datetime import datetime, timedelta           # Date arithmetic (unused here but available)
from typing import Dict, List, Optional            # Type annotations


# ── Page configuration ───────────────────────────────────────────────────
# Must be the first Streamlit command; sets browser tab title, icon, and layout
st.set_page_config(
    page_title='Chronos - Productivity Forecasting',
    page_icon='📊',
    layout='wide',                    # Use the full browser width
    initial_sidebar_state='expanded'  # Sidebar is open by default
)

# Let the user override the API base URL from the sidebar (useful for remote deploys)
API_BASE = st.sidebar.text_input('API Base URL', value='http://localhost:8000')


# ── Data loading ─────────────────────────────────────────────────────────
@st.cache_data
def load_csv(path: str = 'data/sample_timeseries.csv') -> pd.DataFrame:
    """Load the sample time-series CSV and cache it across Streamlit re-runs.

    Args:
        path: File path to the CSV with columns (timestamp, series_id, value).

    Returns:
        Parsed DataFrame, or an empty DataFrame on error.
    """
    try:
        df = pd.read_csv(path, parse_dates=['timestamp'])  # Parse timestamp column as datetime
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")  # Surface the error in the UI
        return pd.DataFrame()                 # Return empty frame so downstream code can check .empty


# ── API helper ───────────────────────────────────────────────────────────
def call_api(endpoint: str, payload: dict) -> Optional[dict]:
    """POST a JSON payload to a Chronos API endpoint and return the response.

    Handles HTTP errors and connection failures gracefully by displaying a
    Streamlit error widget and returning None.

    Args:
        endpoint: API path, e.g. "/forecast" or "/recommend".
        payload: Dictionary that will be JSON-serialised as the request body.

    Returns:
        Parsed JSON response dict, or None on failure.
    """
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()  # Parse and return the JSON body
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {e}")
        return None


# ── Plotting helpers ─────────────────────────────────────────────────────

def plot_forecast_curve(historical: List[float], forecast: List[float], confidence_intervals: Optional[List[Dict]] = None):
    """Build a Plotly figure showing historical data, forecast line, and CI band.

    The historical segment is drawn as a solid blue line; the forecast segment as
    a dashed red line with markers. If confidence intervals are provided, a
    semi-transparent red fill area is drawn around the forecast.

    Args:
        historical: List of historical productivity values.
        forecast: List of predicted future values.
        confidence_intervals: Optional list of dicts with 'lower' and 'upper' keys.

    Returns:
        plotly.graph_objects.Figure ready for st.plotly_chart().
    """
    fig = go.Figure()

    # X-axis indices for the historical segment
    hist_x = list(range(len(historical)))
    # Draw the historical data as a solid blue line
    fig.add_trace(go.Scatter(
        x=hist_x,
        y=historical,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))

    # X-axis indices for the forecast segment (continue from where history ends)
    forecast_x = list(range(len(historical), len(historical) + len(forecast)))
    # Draw the forecast as a dashed red line with circle markers
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Optionally draw confidence-interval bands around the forecast
    if confidence_intervals:
        # Extract lower and upper bounds; fall back to ±5 if keys are missing
        lower = [ci.get('lower', f - 5) for f, ci in zip(forecast, confidence_intervals)]
        upper = [ci.get('upper', f + 5) for f, ci in zip(forecast, confidence_intervals)]

        # Filled area between upper and reversed-lower bounds (Plotly "toself" fill)
        fig.add_trace(go.Scatter(
            x=forecast_x + forecast_x[::-1],          # Forward then backward x for closed polygon
            y=upper + lower[::-1],                     # Upper bound forward, lower bound reversed
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',             # Semi-transparent red
            line=dict(color='rgba(255,255,255,0)'),    # Invisible border
            name='Confidence Interval',
            showlegend=True
        ))

    # Layout adjustments
    fig.update_layout(
        title='Productivity Forecast',
        xaxis_title='Time Step',
        yaxis_title='Productivity Score',
        hovermode='x unified',  # Show all traces' values when hovering at an x position
        height=400
    )

    return fig


def plot_radar_chart(features: Dict[str, float]):
    """Build a polar (radar) chart showing the user's behavioural fingerprint.

    Each feature becomes a spoke on the radar; values are min-max normalised
    to [0, 1] so all features are visually comparable.

    Args:
        features: Dictionary of feature_name → raw value.

    Returns:
        Plotly Figure.
    """
    categories = list(features.keys())    # Spoke labels
    values = list(features.values())      # Raw feature magnitudes

    # Normalise to [0, 1] range so the radar is balanced regardless of units
    max_val = max(values) if values else 1.0
    normalized_values = [v / max_val if max_val > 0 else 0 for v in values]

    fig = go.Figure()

    # Scatterpolar with the loop closed by repeating the first value/category
    fig.add_trace(go.Scatterpolar(
        r=normalized_values + [normalized_values[0]],   # Close the polygon
        theta=categories + [categories[0]],             # Close the polygon
        fill='toself',                                  # Fill the interior
        name='Behavioral Profile',
        line=dict(color='blue')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Fixed radial range after normalisation
            )
        ),
        title='Behavioral Fingerprint',
        height=400
    )

    return fig


def plot_heatmap(series: pd.Series):
    """Build a day-of-week × hour-of-day heatmap of average productivity.

    The series is resampled to hourly resolution, then pivoted so that rows
    are days of the week (Mon–Sun) and columns are hours (0–23).

    Args:
        series: Datetime-indexed productivity Series.

    Returns:
        Plotly Figure.
    """
    # Resample to hourly mean to smooth out minute-level noise
    hourly = series.resample('1H').mean()

    # Create a working DataFrame with day-of-week and hour columns
    df_heatmap = pd.DataFrame({
        'value': hourly.values,
        'day_of_week': hourly.index.dayofweek,  # 0 = Monday, 6 = Sunday
        'hour': hourly.index.hour               # 0–23
    })

    # Pivot so rows = days, columns = hours, cells = mean productivity
    heatmap_data = df_heatmap.pivot_table(
        values='value',
        index='day_of_week',
        columns='hour',
        aggfunc='mean'
    )

    # Build the Plotly Heatmap trace
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,                                          # 2-D value matrix
        x=list(range(24)),                                              # Hour labels
        y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],          # Day labels
        colorscale='Viridis',                                           # Colour palette
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
    """Build a two-panel figure: time-series with trend overlay + trend bar chart.

    Top panel: the full productivity series.
    Bottom panel: a single bar showing the trend classification and its strength.

    Args:
        series: Datetime-indexed productivity Series.
        trend: Trend label string ('rising', 'declining', 'plateauing', 'recovering').
        trend_strength: Numeric strength value for the bar height.

    Returns:
        Plotly Figure with two subplot rows.
    """
    # Create a 2-row subplot layout
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Time Series with Trend', 'Trend Classification'),
        vertical_spacing=0.15  # Gap between panels
    )

    # Top panel: raw time-series
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

    # Colour map for trend labels
    trend_colors = {
        'rising': 'green',
        'declining': 'red',
        'plateauing': 'orange',
        'recovering': 'blue'
    }

    # Bottom panel: single coloured bar representing trend strength
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


# ── Main application ─────────────────────────────────────────────────────

def main():
    """Entry point for the enhanced Streamlit dashboard.

    Orchestrates data loading, sidebar configuration, and rendering of the
    five analysis tabs: Forecast, Patterns, Recommendations, Analytics,
    and Explainability.
    """
    # Page header
    st.title('📊 Chronos - AI-Driven Productivity Forecasting')
    st.markdown("---")

    # ── Sidebar configuration ────────────────────────────────────────────
    st.sidebar.header("Configuration")

    # Allow the user to upload their own CSV or fall back to the sample file
    data_file = st.sidebar.file_uploader("Upload CSV", type=['csv'], help="CSV with columns: timestamp, series_id, value")

    if data_file:
        df = pd.read_csv(data_file, parse_dates=['timestamp'])  # Parse uploaded file
    else:
        df = load_csv()  # Load the default sample dataset

    # Guard: stop early if no data is available
    if df.empty:
        st.warning("Please upload a CSV file or ensure data/sample_timeseries.csv exists")
        return

    # Let the user pick which series to analyse
    series_ids = df['series_id'].unique().tolist()
    selected_series = st.sidebar.selectbox('Select Series', series_ids)

    # Filter the DataFrame to the chosen series and set timestamp as index
    sdf = df[df['series_id'] == selected_series].set_index('timestamp')
    values = sdf['value'].astype(float).dropna()  # Drop NaN values for clean analysis

    # Need at least 60 points to fill the default feature-extraction window
    if len(values) < 60:
        st.error("Series too short. Need at least 60 data points.")
        return

    # ── Tab layout ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Forecast", "🔍 Patterns", "💡 Recommendations", "📊 Analytics", "🔬 Explainability"
    ])

    # ────────────────────────────────────────────────────────────────────
    # Tab 1: Forecast
    # ────────────────────────────────────────────────────────────────────
    with tab1:
        st.header("Productivity Forecast")

        # Two-column layout for horizon slider and model selector
        col1, col2 = st.columns(2)
        with col1:
            horizon = st.slider("Forecast Horizon", 1, 30, 7)  # How many steps ahead to predict
        with col2:
            model_type = st.selectbox("Model Type", ["ensemble", "lstm", "tcn", "transformer"])

        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                window = values.tail(60).tolist()  # Last 60 values as input context
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "horizon": horizon,
                    "model_type": model_type
                }

                result = call_api("/forecast", payload)  # POST to /forecast endpoint

                if result:
                    st.success(f"Forecast generated using {result['model']} model")

                    # Render the forecast line chart with optional confidence bands
                    fig = plot_forecast_curve(
                        historical=window,
                        forecast=result['predictions'],
                        confidence_intervals=result.get('confidence_intervals')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display key metrics in a three-column row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latency", f"{result['latency_ms']:.2f} ms")
                    with col2:
                        st.metric("Forecast Range", f"{min(result['predictions']):.2f} - {max(result['predictions']):.2f}")
                    with col3:
                        avg_forecast = np.mean(result['predictions'])
                        st.metric("Average Forecast", f"{avg_forecast:.2f}")

    # ────────────────────────────────────────────────────────────────────
    # Tab 2: Patterns
    # ────────────────────────────────────────────────────────────────────
    with tab2:
        st.header("Pattern Detection")

        window_size = st.slider("Window Size", 10, 200, 60)  # Adjustable analysis window

        if st.button("Detect Patterns", type="primary"):
            with st.spinner("Analyzing patterns..."):
                # Send up to 2× the window size so the API has enough data for trend comparison
                window = values.tail(min(len(values), window_size * 2)).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "window_size": window_size
                }

                result = call_api("/patterns", payload)  # POST to /patterns endpoint

                if result:
                    # Display key pattern metrics in two columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Trend", result['trend'].upper())
                        st.metric("Trend Strength", f"{result['trend_strength']:.2f}")
                        st.metric("Behavioral Phase", result.get('behavioral_phase', 'N/A').upper())

                    with col2:
                        st.metric("Anomalies Detected", len(result.get('anomalies', [])))

                    # Render the trend analysis subplot figure
                    fig = plot_trend_analysis(
                        series=values,
                        trend=result['trend'],
                        trend_strength=result['trend_strength']
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show the full extracted behavioural feature table
                    st.subheader("Behavioral Features")
                    features_df = pd.DataFrame([result['features']]).T  # Transpose: features as rows
                    features_df.columns = ['Value']
                    st.dataframe(features_df, use_container_width=True)

                    # Show anomalies table if any were detected
                    if result.get('anomalies'):
                        st.subheader("Detected Anomalies")
                        anomalies_df = pd.DataFrame(result['anomalies'])
                        st.dataframe(anomalies_df, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # Tab 3: Recommendations
    # ────────────────────────────────────────────────────────────────────
    with tab3:
        st.header("Productivity Recommendations")

        num_recs = st.slider("Number of Recommendations", 1, 10, 3)
        multi_day = st.checkbox("Generate Multi-Day Plan", value=True)

        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                window = values.tail(60).tolist()  # Context window for the recommendation engine
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "num_recommendations": num_recs,
                    "multi_day": multi_day
                }

                result = call_api("/recommend", payload)  # POST to /recommend endpoint

                if result:
                    recommendations = result.get('recommendations', [])

                    # Render each recommendation as a collapsible expander
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

                            # Render the multi-day plan as a table if requested and available
                            if rec.get('multi_day_plan') and multi_day:
                                st.write("**Multi-Day Plan:**")
                                plan_df = pd.DataFrame(rec['multi_day_plan'])
                                st.dataframe(plan_df, use_container_width=True)

    # ────────────────────────────────────────────────────────────────────
    # Tab 4: Analytics
    # ────────────────────────────────────────────────────────────────────
    with tab4:
        st.header("Productivity Analytics")

        # Interactive Plotly line chart of the full series
        st.subheader("Time Series Overview")
        fig = px.line(values.reset_index(), x='timestamp', y='value', title='Productivity Over Time')
        st.plotly_chart(fig, use_container_width=True)

        # Behavioural profile radar chart (requires an API call to /patterns)
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
                    fig = plot_radar_chart(result['features'])  # Radar chart of normalised features
                    st.plotly_chart(fig, use_container_width=True)

        # Daily productivity heatmap (needs at least 1 week of data)
        st.subheader("Daily Productivity Patterns")
        if len(values) > 24 * 7:
            fig = plot_heatmap(values)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least a week of data for heatmap")

        # Summary statistics in a four-column row
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

    # ────────────────────────────────────────────────────────────────────
    # Tab 5: Explainability
    # ────────────────────────────────────────────────────────────────────
    with tab5:
        st.header("Model Explainability")

        top_features = st.slider("Top Features", 5, 50, 10)  # How many features to show

        if st.button("Explain Prediction", type="primary"):
            with st.spinner("Generating explanation..."):
                window = values.tail(60).tolist()
                payload = {
                    "series_id": selected_series,
                    "window": window,
                    "top_features": top_features
                }

                result = call_api("/explain", payload)  # POST to /explain endpoint

                if result:
                    st.write(f"**Reasoning:** {result.get('reasoning', 'N/A')}")

                    # Feature-importance bar chart
                    if result.get('feature_importance'):
                        # Convert the importance dict to a sorted DataFrame
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

                        # Also show the full importance table
                        st.dataframe(importance_df, use_container_width=True)


# ── Script entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    main()
