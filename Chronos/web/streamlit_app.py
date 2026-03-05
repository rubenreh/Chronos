"""
Simple Streamlit dashboard for the Chronos time-series forecasting demo.

This is the lightweight version of the Chronos web frontend. It provides:
  • A sidebar dropdown to select which time-series to visualise.
  • A line chart showing the most recent 500 data points.
  • A "Predict next step" button that sends the last 60 values to the
    FastAPI /predict endpoint and displays the one-step-ahead forecast.

For the full-featured dashboard with tabs for Patterns, Recommendations,
Analytics, and Explainability, see enhanced_streamlit_app.py.

Run with:
    streamlit run web/streamlit_app.py
"""

import streamlit as st  # Streamlit framework for interactive web UIs
import pandas as pd     # DataFrame operations for loading and filtering CSV data
import requests         # HTTP client for calling the FastAPI backend

# Configure the browser tab title for the Streamlit page
st.set_page_config(page_title='Chronos Demo')
# Render the main heading at the top of the page
st.title('Chronos — Forecast & Anomaly Demo')


@st.cache_data
def load_csv(path='data/sample_timeseries.csv'):
    """Load the sample time-series CSV and cache it for fast re-renders.

    Streamlit's @cache_data decorator ensures the CSV is only read from disk
    once; subsequent calls return the cached DataFrame instantly.

    Args:
        path: File path to the CSV with columns (timestamp, series_id, value).

    Returns:
        pandas DataFrame with a parsed datetime 'timestamp' column.
    """
    return pd.read_csv(path, parse_dates=['timestamp'])


# Load the dataset into a DataFrame
df = load_csv()

# Extract the unique series identifiers for the sidebar selector
series_ids = df['series_id'].unique().tolist()

# Sidebar dropdown lets the user pick which series to view
sel = st.sidebar.selectbox('Series', series_ids)

# Filter the DataFrame to only the selected series and index by timestamp
sdf = df[df['series_id'] == sel].set_index('timestamp')
# Extract the 'value' column as a float Series for plotting and API calls
values = sdf['value'].astype(float)

# Display the most recent 500 data points as a Streamlit line chart
st.line_chart(values.tail(500))

# Section for invoking the model prediction
st.write('Call model:')
if st.button('Predict next step'):
    # Take the last 60 non-null values as the input window for the model
    window = values.dropna().tail(60).tolist()
    # Build the JSON payload expected by the FastAPI /predict endpoint
    payload = {'series_id': sel, 'window': window, 'horizon': 1}
    try:
        # POST the prediction request to the locally-running FastAPI server
        r = requests.post('http://localhost:8000/predict', json=payload, timeout=5)
        if r.status_code == 200:
            res = r.json()  # Parse the JSON response body
            # Display the returned predictions in a green success box
            st.success(f"Prediction: {res['predictions']}")
        else:
            # Non-200 status → show the error code and response body
            st.error(f"Server returned {r.status_code}: {r.text}")
    except Exception as e:
        # Network errors, timeouts, etc.
        st.error(f"Error contacting server: {e}")
