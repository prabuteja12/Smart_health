from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import joblib
import os
import numpy as np
import pandas as pd
import shap
import traceback

MODEL_PATH = "xgb_model.pkl"
PATIENT_EVENTS = "patient_events.csv"
STAFF_SCHEDULE = "staff_schedule.csv"
TASK_LOG = "task_log.csv"
OP_METRICS = "operational_metrics.csv"

app = FastAPI(title="Workload Balancer Model API")

class FeaturesPayload(BaseModel):
    features: Dict[str, float]

def load_artifact(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found at {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        model = obj.get("model")
        scaler = obj.get("scaler")
        feature_cols = obj.get("features")
    else:
        model = obj
        scaler = None
        feature_cols = None
    return model, scaler, feature_cols

def build_hourly_features(pe_path=PATIENT_EVENTS, ss_path=STAFF_SCHEDULE, tl_path=TASK_LOG, om_path=OP_METRICS):
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
    tasks = df_tl.groupby("hour").agg(
        tasks_created=("task_id", "count")
    ).reset_index()

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

def ensure_features_order(payload_features: Dict[str, float], feature_cols):
    X = pd.DataFrame([payload_features])
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols].astype(float)
    return X

def predict_from_dataframe(model, scaler, X_df):
    X_input = X_df.copy()
    if scaler is not None:
        X_input_trans = scaler.transform(X_input)
    else:
        X_input_trans = X_input.values
    preds = model.predict(X_input_trans)
    return preds

def compute_shap_values(model, X_input):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        try:
            explainer = shap.Explainer(model)
        except Exception:
            raise RuntimeError("No compatible SHAP explainer available for this model.")
    try:
        sv = explainer.shap_values(X_input)
    except Exception:
        # newer shap returns object; call explainer(X)
        res = explainer(X_input)
        # `res` may expose .values
        sv = res.values if hasattr(res, "values") else res
    return sv

@app.on_event("startup")
def startup_load():
    global MODEL, SCALER, FEATURE_COLS
    try:
        MODEL, SCALER, FEATURE_COLS = load_artifact(MODEL_PATH)
    except Exception as e:
        MODEL, SCALER, FEATURE_COLS = None, None, None
        print("Warning: model not loaded on startup:", str(e))

@app.post("/predict")
def predict(payload: FeaturesPayload):
    try:
        model, scaler, feature_cols = load_artifact(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")
    if feature_cols is None:
        feature_cols = list(payload.features.keys())
    X = ensure_features_order(payload.features, feature_cols)
    try:
        preds = predict_from_dataframe(model, scaler, X)
        return {"prediction": float(preds[0])}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/predict/latest")
def predict_latest():
    try:
        model, scaler, feature_cols = load_artifact(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")
    try:
        df = build_hourly_features(PATIENT_EVENTS, STAFF_SCHEDULE, TASK_LOG, OP_METRICS)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available to build features.")
        # take last row as "current" features
        last = df.iloc[[-1]]
        # prepare feature vector
        if feature_cols is None:
            feature_cols = [c for c in last.columns if c not in ["hour", "target_next_hour"]]
        X = last[feature_cols].astype(float)
        preds = predict_from_dataframe(model, scaler, X)
        return {"prediction": float(preds[0]), "features_used": feature_cols}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to build features/predict: {e}")

@app.post("/explain")
def explain(payload: Optional[FeaturesPayload] = None):
    try:
        model, scaler, feature_cols = load_artifact(MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        if payload is None:
            df = build_hourly_features(PATIENT_EVENTS, STAFF_SCHEDULE, TASK_LOG, OP_METRICS)
            if df.empty:
                raise HTTPException(status_code=400, detail="No data available for explain.")
            row = df.iloc[[-1]]
            if feature_cols is None:
                feature_cols = [c for c in row.columns if c not in ["hour", "target_next_hour"]]
            X = row[feature_cols].astype(float)
        else:
            if feature_cols is None:
                feature_cols = list(payload.features.keys())
            X = ensure_features_order(payload.features, feature_cols)

        X_model = pd.DataFrame(X.values, columns=feature_cols)
        if scaler is not None:
            X_for_model = pd.DataFrame(scaler.transform(X_model), columns=feature_cols)
        else:
            X_for_model = X_model

        preds = predict_from_dataframe(model, scaler, X_model)
        shap_vals = compute_shap_values(model, X_for_model)

        # shap_vals may be array or list; pick first output if needed
        if isinstance(shap_vals, list):
            sv = shap_vals[0]
        else:
            sv = shap_vals

        sv_row = sv[0] if sv.ndim == 2 else sv
        feature_shap = {feature_cols[i]: float(sv_row[i]) for i in range(len(feature_cols))}
        return {
            "prediction": float(preds[0]),
            "shap_values": feature_shap,
            "features": {c: float(X_model.iloc[0, i]) for i, c in enumerate(feature_cols)}
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8080, reload=True)
