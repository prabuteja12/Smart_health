import json
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib

SEED = 42
np.random.seed(SEED)

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

def prepare_dataset(df):
    df = df.copy()
    df["target_next_hour"] = df["arrivals"].shift(-1)
    df = df.dropna(subset=["target_next_hour"]).reset_index(drop=True)
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

def split_timewise(X, y, times, train_frac=0.8):
    n = len(X)
    cutoff = int(n * train_frac)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    t_train, t_test = times.iloc[:cutoff], times.iloc[cutoff:]
    return X_train, X_test, y_train, y_test, t_train, t_test

def safe_xgb_fit(model, X_train, y_train, X_val, y_val):
    used = "none"
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=25,
            verbose=False
        )
        used = "early_stopping_rounds"
    except TypeError:
        try:
            cb = []
            if hasattr(xgb, "callback") and hasattr(xgb.callback, "EarlyStopping"):
                cb.append(xgb.callback.EarlyStopping(rounds=25, save_best=True))
            if cb:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=cb,
                    verbose=False
                )
                used = "callbacks"
            else:
                raise AttributeError("callback interface not available")
        except (TypeError, AttributeError):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            used = "no_early_stopping"
    return model, used

def train_and_save(X_train, y_train, X_val, y_val, feature_cols, model_path="xgb_model.pkl", meta_path="model_meta.json"):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=4
    )

    model, fit_method = safe_xgb_fit(model, X_train_s, y_train, X_val_s, y_val)

    preds_val = model.predict(X_val_s)
    mae = mean_absolute_error(y_val, preds_val)
    mse = mean_squared_error(y_val, preds_val)
    rmse = float(np.sqrt(mse))

    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_path)
    meta = {"feature_columns": feature_cols, "model_path": os.path.abspath(model_path), "xgboost_version": getattr(xgb, "__version__", "unknown"), "fit_method": fit_method}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model, scaler, mae, rmse

if __name__ == "__main__":
    pe, ss, tl, om = load_data()
    df = build_hourly_features(pe, ss, tl, om)
    X, y, times, feature_cols = prepare_dataset(df)
    X_train, X_test, y_train, y_test, t_train, t_test = split_timewise(X, y, times, train_frac=0.8)
    model, scaler, mae_val, rmse_val = train_and_save(X_train, y_train, X_test, y_test, feature_cols)
    X_test_s = scaler.transform(X_test)
    preds_test = model.predict(X_test_s)
    mae_test = mean_absolute_error(y_test, preds_test)
    mse_test = mean_squared_error(y_test, preds_test)
    rmse_test = float(np.sqrt(mse_test))

    print(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
    print(f"Validation MAE: {mae_val:.4f}  RMSE: {rmse_val:.4f}")
    print(f"Test MAE:       {mae_test:.4f}  RMSE: {rmse_test:.4f}")
    print("Model and metadata saved to disk.")
