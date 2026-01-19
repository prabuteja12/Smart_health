import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
import shap
import matplotlib.pyplot as plt

def load_data(path_pat="patient_events.csv",
              path_staff="staff_schedule.csv",
              path_tasks="task_log.csv",
              path_ops="operational_metrics.csv"):
    pe = pd.read_csv(path_pat, parse_dates=["timestamp"])
    ss = pd.read_csv(path_staff, parse_dates=["shift_start", "shift_end"])
    tl = pd.read_csv(path_tasks, parse_dates=["created_at", "completed_at"])
    om = pd.read_csv(path_ops, parse_dates=["timestamp"])
    return pe, ss, tl, om

def build_hourly_features(pe, ss, tl, om):
    df_pe = pe.copy()
    df_pe["hour"] = df_pe["timestamp"].dt.floor("h")
    arrivals = df_pe.groupby("hour").agg(
        arrivals=("patient_id", "count"),
        avg_acuity=("acuity_level", "mean"),
        avg_proc_time=("processing_time_minutes", "mean")
    ).reset_index()

    df_tl = tl.copy()
    df_tl["hour"] = df_tl["created_at"].dt.floor("h")
    tasks = df_tl.groupby("hour").agg(
        tasks_created=("task_id", "count"),
        avg_task_duration=("completed_at", lambda s: (pd.to_datetime(s) - pd.to_datetime(df_tl.loc[s.index, "created_at"])).dt.total_seconds().mean()/60)
    ).reset_index()

    ss_expanded = []
    for _, r in ss.iterrows():
        start = pd.to_datetime(r["shift_start"]).floor("h")
        end = pd.to_datetime(r["shift_end"]).floor("h")
        hrs = int(((end - start) / pd.Timedelta(hours=1)) + 1)
        for i in range(hrs):
            ts = start + pd.Timedelta(hours=i)
            ss_expanded.append((r["staff_id"], r["role"], ts))
    ss_df = pd.DataFrame(ss_expanded, columns=["staff_id", "role", "hour"])
    staff_count = ss_df.groupby("hour").agg(staff_on_duty=("staff_id", "nunique")).reset_index()

    om_hour = om.copy()
    om_hour["hour"] = om_hour["timestamp"].dt.floor("h")
    om_agg = om_hour.groupby(["hour", "metric_name"]).agg(value=("value", "mean")).reset_index()
    om_pivot = om_agg.pivot(index="hour", columns="metric_name", values="value").reset_index().rename_axis(None, axis=1)

    df = arrivals.merge(tasks, on="hour", how="left")
    df = df.merge(staff_count, on="hour", how="left")
    df = df.merge(om_pivot, on="hour", how="left")
    df = df.sort_values("hour").reset_index(drop=True)

    df["arrivals"] = df["arrivals"].fillna(0)
    df["tasks_created"] = df["tasks_created"].fillna(0)
    df["avg_acuity"] = df["avg_acuity"].fillna(df["avg_acuity"].median())
    df["avg_proc_time"] = df["avg_proc_time"].fillna(df["avg_proc_time"].median())
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

def prepare_dataset(df, feature_cols=None):
    df = df.copy()
    df["target_next_hour"] = df["arrivals"].shift(-1)
    df = df.dropna(subset=["target_next_hour"]).reset_index(drop=True)
    if feature_cols is None:
        feature_cols = [
            "arrivals", "avg_acuity", "avg_proc_time", "tasks_created",
            "staff_on_duty", "hour_of_day", "day_of_week", "is_weekend",
            "arrivals_roll_1h","arrivals_roll_3h","arrivals_roll_6h","arrivals_roll_12h","arrivals_roll_24h",
            "tasks_roll_1h","tasks_roll_3h","tasks_roll_6h","tasks_roll_12h","tasks_roll_24h"
        ]
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_cols].astype(float)
    y = df["target_next_hour"].astype(float)
    times = df["hour"]
    return X, y, times, feature_cols

def load_artifact(path="xgb_model.pkl"):
    d = joblib.load(path)
    model = d.get("model") if isinstance(d, dict) else d
    scaler = d.get("scaler") if isinstance(d, dict) else None
    features = d.get("features") if isinstance(d, dict) else None
    return model, scaler, features

def compute_shap(model, X, nsamples=None):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    if nsamples:
        Xs = X.sample(min(nsamples, len(X)), random_state=0)
    else:
        Xs = X
    try:
        shap_values = explainer.shap_values(Xs)
    except Exception:
        shap_values = explainer(Xs).values
    return explainer, Xs, shap_values

def save_summary_plot(shap_values, X, outpath):
    plt.figure(figsize=(8,6))
    try:
        shap.summary_plot(shap_values, X, show=False)
    except Exception:
        # fallback: if shap_values is list (multioutput), pick first
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X, show=False)
        else:
            raise
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_feature_importance(shap_values, X, outjson):
    if isinstance(shap_values, list):
        vals = shap_values[0]
    else:
        vals = shap_values
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    features = list(X.columns)
    importance = [{"feature": features[i], "mean_abs_shap": float(mean_abs[i])} for i in order]
    with open(outjson, "w") as f:
        json.dump(importance, f, indent=2)

def save_per_sample(shap_values, X, outcsv):
    if isinstance(shap_values, list):
        vals = shap_values[0]
    else:
        vals = shap_values
    df_shap = pd.DataFrame(vals, columns=X.columns, index=X.index)
    df_shap.to_csv(outcsv)

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    model, scaler, feature_cols = load_artifact(args.model)
    pe, ss, tl, om = load_data(args.patient_events, args.staff_schedule, args.task_log, args.operational_metrics)
    df = build_hourly_features(pe, ss, tl, om)
    X, y, times, inferred_features = prepare_dataset(df, feature_cols)
    if scaler is not None:
        X_scaled = X.copy()
        # keep original X for SHAP plots (SHAP prefers original feature space); explainer can accept scaled too,
        # but we will compute SHAP on the model input space. If scaler exists, transform for model prediction but use X for plotting.
        X_model = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    else:
        X_model = X
        X_scaled = X

    explainer, Xs, shap_values = compute_shap(model, X_model, nsamples=args.nsamples)

    summary_png = os.path.join(args.outdir, "shap_summary.png")
    save_summary_plot(shap_values, Xs, summary_png)

    feature_json = os.path.join(args.outdir, "shap_feature_importance.json")
    save_feature_importance(shap_values, Xs, feature_json)

    per_sample_csv = os.path.join(args.outdir, "shap_per_sample.csv")
    save_per_sample(shap_values, Xs, per_sample_csv)

    meta = {
        "model_artifact": os.path.abspath(args.model),
        "n_rows_explained": int(len(Xs)),
        "summary_png": os.path.abspath(summary_png),
        "feature_importance_json": os.path.abspath(feature_json),
        "per_sample_csv": os.path.abspath(per_sample_csv)
    }
    with open(os.path.join(args.outdir, "shap_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="xgb_model.pkl")
    p.add_argument("--patient_events", default="patient_events.csv")
    p.add_argument("--staff_schedule", default="staff_schedule.csv")
    p.add_argument("--task_log", default="task_log.csv")
    p.add_argument("--operational_metrics", default="operational_metrics.csv")
    p.add_argument("--outdir", default="shap_outputs")
    p.add_argument("--nsamples", type=int, default=500)
    args = p.parse_args()
    main(args)
