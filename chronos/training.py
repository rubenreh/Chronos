"""Training script: simple LSTM baseline training using sample CSV input.
This script is intentionally clear and verbose for learning and interviewing.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from chronos.preprocess import load_timeseries, pivot_series, resample_and_fill, sliding_windows
from chronos.models.lstm_model import make_model


class SequenceDataset(Dataset):
    def __init__(self, series_values, input_len=60, horizon=1):
        # series_values: 1D numpy array
        x, y = sliding_windows(series_values, input_len=input_len, horizon=horizon)
        # use only single-step horizon for baseline
        self.x = x.astype(np.float32)
        self.y = y.squeeze().astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx][:, None], self.y[idx]

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred.squeeze(), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

def eval_loop(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred.squeeze(), yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(dataloader.dataset)

def main(args):
    os.makedirs(os.path.dirname(args.model_out) or '.', exist_ok=True)
    df = load_timeseries(args.data)
    series = pivot_series(df)

    # combine all series for baseline training
    all_vals = None
    for sid, s in series.items():
        s2 = resample_and_fill(s, freq=args.freq)
        vals = s2.values
        if all_vals is None:
            all_vals = vals
        else:
            all_vals = np.concatenate([all_vals, vals])

    # normalize
    mu = all_vals.mean()
    sigma = all_vals.std()
    print(f"global mean={mu:.4f} std={sigma:.4f}")

    dataset = SequenceDataset((all_vals - mu)/sigma, input_len=args.input_len, horizon=1)
    n = len(dataset)
    split = int(n*0.9)
    train_ds = torch.utils.data.Subset(dataset, list(range(0, split)))
    val_ds = torch.utils.data.Subset(dataset, list(range(split, n)))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(input_size=1, hidden_size=args.hidden, num_layers=2, out_steps=1).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, device)
        val_loss = eval_loop(val_loader, model, loss_fn, device)
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'mu':mu, 'sigma':sigma}, args.model_out)
    print(f"Saved model to {args.model_out}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/sample_timeseries.csv')
    p.add_argument('--model_out', default='artifacts/lstm_model.pth')
    p.add_argument('--input_len', type=int, default=60)
    p.add_argument('--freq', default='1T')
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    main(args)
