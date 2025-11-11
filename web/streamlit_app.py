"""Simple Streamlit demo to visualize time-series and call the Chronos predict API."""
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title='Chronos Demo')
st.title('Chronos — Forecast & Anomaly Demo')

@st.cache_data
def load_csv(path='data/sample_timeseries.csv'):
    return pd.read_csv(path, parse_dates=['timestamp'])

df = load_csv()
series_ids = df['series_id'].unique().tolist()
sel = st.sidebar.selectbox('Series', series_ids)

sdf = df[df['series_id']==sel].set_index('timestamp')
values = sdf['value'].astype(float)

st.line_chart(values.tail(500))

st.write('Call model:')
if st.button('Predict next step'):
    window = values.dropna().tail(60).tolist()
    payload = {'series_id': sel, 'window': window, 'horizon': 1}
    try:
        r = requests.post('http://localhost:8000/predict', json=payload, timeout=5)
        if r.status_code==200:
            res = r.json()
            st.success(f"Prediction: {res['predictions']}")
        else:
            st.error(f"Server returned {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Error contacting server: {e}")
