import argparse
import os
import random
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class SeqDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        seq = self.X[start:end]
        target = self.y[end]
        return seq, target

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)

def load_hourly_features(pe_path="patient_events.csv", ss_path="staff_schedule.csv", tl_path="task_log.csv", om_path="operational_metrics.csv"):
    pe = pd.read_csv(pe_path, parse_dates=["timestamp"])
    ss = pd.read_csv(ss_path, parse_dates=["shift_start", "shift_end"])
    tl = pd.read_csv(tl_path, parse_dates=["created_at", "completed_at"])
    om = pd.read_csv(om_path, parse_dates=["timestamp"])

    df_pe = pe.copy()
    df_pe["hour"] = df_pe["timestamp"].dt.floor("h")
    arrivals = df_pe.groupby("hour").agg(
        arrivals=("patient_id", "count"),
        avg_acuity=("acuity_level", "mean"),
        avg_proc_time=("processing_time_minutes", "mean")
    ).reset_index()

    df_tl = tl.copy()
    df_tl["hour"] = df_tl["created_at"].dt.floor("h")
    tasks = df_tl.groupby("hour").agg(tasks_created=("task_id", "count")).reset_index()

    ss_expanded = []
    for _, r in ss.iterrows():
        start = pd.to_datetime(r["shift_start"]).floor("h")
        end = pd.to_datetime(r["shift_end"]).floor("h")
        hrs = int(((end - start) / pd.Timedelta(hours=1)) + 1)
        for i in range(hrs):
            ts = start + pd.Timedelta(hours=i)
            ss_expanded.append((r["staff_id"], r.get("role", None), ts))
    ss_df = pd.DataFrame(ss_expanded, columns=["staff_id", "role", "hour"])
    staff_count = ss_df.groupby("hour").agg(staff_on_duty=("staff_id", "nunique")).reset_index()

    om_hour = om.copy()
    om_hour["hour"] = om_hour["timestamp"].dt.floor("h")
    om_agg = om_hour.groupby(["hour", "metric_name"]).agg(value=("value", "mean")).reset_index()
    if not om_agg.empty:
        om_pivot = om_agg.pivot(index="hour", columns="metric_name", values="value").reset_index().rename_axis(None, axis=1)
    else:
        om_pivot = pd.DataFrame(columns=["hour"])

    df = arrivals.merge(tasks, on="hour", how="left")
    df = df.merge(staff_count, on="hour", how="left")
    if "hour" in om_pivot.columns:
        df = df.merge(om_pivot, on="hour", how="left")
    df = df.sort_values("hour").reset_index(drop=True)

    df["arrivals"] = df["arrivals"].fillna(0)
    df["tasks_created"] = df["tasks_created"].fillna(0)
    df["avg_acuity"] = df["avg_acuity"].fillna(df["avg_acuity"].median() if not df["avg_acuity"].isna().all() else 0)
    df["avg_proc_time"] = df["avg_proc_time"].fillna(df["avg_proc_time"].median() if not df["avg_proc_time"].isna().all() else 0)
    df["staff_on_duty"] = df["staff_on_duty"].fillna(0)

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    df = df.set_index("hour")
    for w in [1,3,6,12,24]:
        df[f"arrivals_roll_{w}h"] = df["arrivals"].rolling(window=w, min_periods=1).mean()
        df[f"tasks_roll_{w}h"] = df["tasks_created"].rolling(window=w, min_periods=1).mean()
    df = df.reset_index()
    return df

def prepare_sequence_data(df, feature_cols=None):
    dfc = df.copy()
    dfc["target_next_hour"] = dfc["arrivals"].shift(-1)
    dfc = dfc.dropna(subset=["target_next_hour"]).reset_index(drop=True)
    if feature_cols is None:
        feature_cols = [
            "arrivals", "avg_acuity", "avg_proc_time", "tasks_created",
            "staff_on_duty", "hour_of_day", "day_of_week", "is_weekend",
            "arrivals_roll_1h","arrivals_roll_3h","arrivals_roll_6h","arrivals_roll_12h","arrivals_roll_24h",
            "tasks_roll_1h","tasks_roll_3h","tasks_roll_6h","tasks_roll_12h","tasks_roll_24h"
        ]
    for c in feature_cols:
        if c not in dfc.columns:
            dfc[c] = 0
    X = dfc[feature_cols].astype(float).values
    y = dfc["target_next_hour"].astype(float).values
    return X, y, feature_cols

def time_split(X, y, frac=0.8):
    n = len(X)
    cutoff = int(n * frac)
    return X[:cutoff], y[:cutoff], X[cutoff:], y[cutoff:]

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    df = load_hourly_features(args.patient_events, args.staff_schedule, args.task_log, args.operational_metrics)
    X, y, feature_cols = prepare_sequence_data(df)
    X_train, y_train, X_val, y_val = time_split(X, y, frac=args.train_frac)

    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    X_train_s = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_val_s = scaler.transform(X_val_flat).reshape(X_val.shape)

    seq_len = args.seq_len
    train_dataset = SeqDataset(X_train_s, y_train, seq_len)
    val_dataset = SeqDataset(X_val_s, y_val, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LSTMModel(input_size=X.shape[1], hidden_size=args.hidden_size, num_layers=args.num_layers, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = args.patience
    wait = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for seq, targ in train_loader:
            seq = seq.to(device)
            targ = targ.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = loss_fn(pred, targ)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        val_losses = []
        preds_list = []
        trues_list = []
        with torch.no_grad():
            for seq, targ in val_loader:
                seq = seq.to(device)
                targ = targ.to(device)
                pred = model(seq)
                loss = loss_fn(pred, targ)
                val_losses.append(loss.item())
                preds_list.append(pred.cpu().numpy())
                trues_list.append(targ.cpu().numpy())
        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0

        preds_arr = np.concatenate(preds_list) if preds_list else np.array([])
        trues_arr = np.concatenate(trues_list) if trues_list else np.array([])
        val_mae = float(mean_absolute_error(trues_arr, preds_arr)) if len(preds_arr) else float("nan")

        if avg_val < best_val - 1e-6:
            best_val = avg_val
            wait = 0
            os.makedirs(args.outdir, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "args": vars(args), "feature_cols": feature_cols, "seq_len": seq_len, "hidden_size": args.hidden_size, "num_layers": args.num_layers}, os.path.join(args.outdir, "lstm_model.pth"))
            joblib.dump({"scaler": scaler, "feature_cols": feature_cols, "seq_len": seq_len}, os.path.join(args.outdir, "lstm_meta.pkl"))
        else:
            wait += 1
            if wait >= patience:
                break

    # final evaluation on validation set (sequence-wise)
    model.load_state_dict(torch.load(os.path.join(args.outdir, "lstm_model.pth"))["model_state"])
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for seq, targ in val_loader:
            seq = seq.to(device)
            pred = model(seq)
            all_preds.append(pred.cpu().numpy())
            all_trues.append(targ.cpu().numpy())
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_trues = np.concatenate(all_trues)
        mae_final = float(mean_absolute_error(all_trues, all_preds))
    else:
        mae_final = float("nan")

    print(f"Training finished. Best validation loss: {best_val:.6f}. Validation MAE: {mae_final:.4f}")
    print(f"Artifacts written to: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--patient_events", default="patient_events.csv")
    p.add_argument("--staff_schedule", default="staff_schedule.csv")
    p.add_argument("--task_log", default="task_log.csv")
    p.add_argument("--operational_metrics", default="operational_metrics.csv")
    p.add_argument("--outdir", default="lstm_outputs")
    p.add_argument("--seq_len", type=int, default=6)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--use_cuda", action="store_true")
    args = p.parse_args()
    train(args)
