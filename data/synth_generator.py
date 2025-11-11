"""Synthetic time-series generator for Chronos demo.
Produces CSV with columns: timestamp, series_id, value
Each series contains trend + seasonality + noise + optional injected anomalies.
"""

import argparse
from datetime import datetime, timedelta
import math
import random
import csv
import os


def generate_series(length=1440, freq_minutes=1, seed=None, inject_anomalies=True):
    if seed is not None:
        random.seed(seed)
    start = datetime(2025,1,1)
    timestamps = [start + timedelta(minutes=i*freq_minutes) for i in range(length)]

    # trend + seasonality + noise
    trend = random.uniform(-0.001, 0.001)
    amp = random.uniform(5, 20)
    period = random.choice([24*60, 12*60, 60, 7*24*60])

    values = []
    baseline = random.uniform(20, 80)
    for i in range(length):
        t = i
        seasonal = amp * math.sin(2*math.pi * t / period)
        val = baseline + trend * t + seasonal + random.gauss(0, amp*0.15)
        values.append(val)

    # inject anomalies
    if inject_anomalies:
        for _ in range(max(1, length//400)):
            idx = random.randint(20, length-20)
            # spike or dip
            if random.random() < 0.6:
                values[idx] += random.uniform(amp*3, amp*7)
            else:
                values[idx] -= random.uniform(amp*3, amp*7)
    return timestamps, values


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/sample_timeseries.csv')
    p.add_argument('--n-series', type=int, default=3)
    p.add_argument('--length', type=int, default=1440)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    except Exception:
        pass

    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp','series_id','value'])
        for s in range(args.n_series):
            ts, vals = generate_series(length=args.length, seed=args.seed + s)
            sid = f"series_{s}"
            for t, v in zip(ts, vals):
                writer.writerow([t.isoformat(), sid, f"{v:.4f}"])

    print(f"Wrote sample data to {args.out}")
