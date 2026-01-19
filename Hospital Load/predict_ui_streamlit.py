# predict_ui_streamlit.py
import os
import joblib
import math
import numpy as np
import pandas as pd
import streamlit as st
import shap
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Predictor — Smart Workload Balancer", layout="centered", initial_sidebar_state="collapsed")

MODEL_ARTIFACT = "xgb_model.pkl"
PATIENT_EVENTS = "patient_events.csv"
STAFF_SCHEDULE = "staff_schedule.csv"
TASK_LOG = "task_log.csv"
OP_METRICS = "operational_metrics.csv"

# ---------- utilities ----------
@st.cache_resource
def load_model(path=MODEL_ARTIFACT):
    if not os.path.exists(path):
        return None, None, None
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model"), obj.get("scaler"), obj.get("features")
    return obj, None, None

@st.cache_data(ttl=300)
def load_latest_hourly_features():
    if not os.path.exists(PATIENT_EVENTS):
        return None
    pe = pd.read_csv(PATIENT_EVENTS, parse_dates=["timestamp"])
    tl = pd.read_csv(TASK_LOG, parse_dates=["created_at"]) if os.path.exists(TASK_LOG) else pd.DataFrame()
    ss = pd.read_csv(STAFF_SCHEDULE, parse_dates=["shift_start", "shift_end"]) if os.path.exists(STAFF_SCHEDULE) else pd.DataFrame()
    om = pd.read_csv(OP_METRICS, parse_dates=["timestamp"]) if os.path.exists(OP_METRICS) else pd.DataFrame()

    df_pe = pe.copy()
    df_pe["hour"] = df_pe["timestamp"].dt.floor("h")
    arrivals = df_pe.groupby("hour").agg(
        arrivals=("patient_id", "count"),
        avg_acuity=("acuity_level", "mean"),
        avg_proc_time=("processing_time_minutes", "mean")
    ).reset_index()

    df_tl = tl.copy() if not tl.empty else pd.DataFrame()
    if not df_tl.empty:
        df_tl["hour"] = df_tl["created_at"].dt.floor("h")
        tasks = df_tl.groupby("hour").agg(tasks_created=("task_id", "count")).reset_index()
    else:
        tasks = pd.DataFrame(columns=["hour", "tasks_created"])

    ss_expanded = []
    if not ss.empty:
        for _, r in ss.iterrows():
            start = pd.to_datetime(r["shift_start"]).floor("h")
            end = pd.to_datetime(r["shift_end"]).floor("h")
            hrs = int(((end - start) / pd.Timedelta(hours=1)) + 1)
            for i in range(max(0, hrs)):
                ts = start + pd.Timedelta(hours=i)
                ss_expanded.append((r["staff_id"], r.get("role", None), ts, r.get("baseline_capacity_per_hour", np.nan)))
    ss_df = pd.DataFrame(ss_expanded, columns=["staff_id", "role", "hour", "baseline_capacity_per_hour"]) if ss_expanded else pd.DataFrame()
    if not ss_df.empty:
        staff_count = ss_df.groupby("hour").agg(
            staff_on_duty=("staff_id", "nunique"),
            avg_capacity_per_staff=("baseline_capacity_per_hour", "mean")
        ).reset_index()
    else:
        staff_count = pd.DataFrame(columns=["hour", "staff_on_duty", "avg_capacity_per_staff"])

    om_pivot = pd.DataFrame()
    if not om.empty:
        om_hour = om.copy()
        om_hour["hour"] = om_hour["timestamp"].dt.floor("h")
        om_agg = om_hour.groupby(["hour", "metric_name"]).agg(value=("value", "mean")).reset_index()
        if not om_agg.empty:
            om_pivot = om_agg.pivot(index="hour", columns="metric_name", values="value").reset_index().rename_axis(None, axis=1)

    df = arrivals.merge(tasks, on="hour", how="left")
    df = df.merge(staff_count, on="hour", how="left")
    if not om_pivot.empty and "hour" in om_pivot.columns:
        df = df.merge(om_pivot, on="hour", how="left")

    if df.empty:
        return None

    df = df.sort_values("hour").reset_index(drop=True)
    df["arrivals"] = df["arrivals"].fillna(0)
    df["tasks_created"] = df.get("tasks_created", 0).fillna(0)
    if "avg_acuity" in df:
        df["avg_acuity"] = df["avg_acuity"].fillna(df["avg_acuity"].median())
    else:
        df["avg_acuity"] = 1.0
    if "avg_proc_time" in df:
        df["avg_proc_time"] = df["avg_proc_time"].fillna(df["avg_proc_time"].median())
    else:
        df["avg_proc_time"] = 30.0
    df["staff_on_duty"] = df["staff_on_duty"].fillna(0)
    df["avg_capacity_per_staff"] = df["avg_capacity_per_staff"].fillna(1.0)

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    df = df.set_index("hour")
    for w in [1,3,6,12,24]:
        df[f"arrivals_roll_{w}h"] = df["arrivals"].rolling(window=w, min_periods=1).mean()
        df[f"tasks_roll_{w}h"] = df["tasks_created"].rolling(window=w, min_periods=1).mean()
    df = df.reset_index()
    return df

def prepare_input_vector(row, feature_cols, scaler):
    # build a single-row dataframe with feature order aligned
    X = pd.DataFrame([row[feature_cols].astype(float)])
    if scaler is not None:
        try:
            Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
            return Xs
        except Exception:
            return X
    return X

def compute_shap_for_row(model, X_for_model, feature_cols):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    try:
        shap_vals = explainer.shap_values(X_for_model)
    except Exception:
        res = explainer(X_for_model)
        shap_vals = res.values if hasattr(res, "values") else res
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    vals = shap_vals[0] if shap_vals.ndim == 2 else shap_vals
    df = pd.DataFrame({"feature": feature_cols, "shap": vals, "abs_shap": np.abs(vals)})
    df["value"] = X_for_model.iloc[0].values
    df = df.sort_values("abs_shap", ascending=False)
    return df

def utilization_and_status(prediction, staff_on_duty, avg_capacity_per_staff, slack_threshold=0.8):
    capacity = staff_on_duty * max(avg_capacity_per_staff, 1e-6)
    if capacity == 0:
        util = float("inf")
    else:
        util = prediction / capacity
    status = "OK"
    if math.isinf(util):
        status = "NO STAFF"
    elif util >= 1.2:
        status = "CRITICAL"
    elif util >= 0.9:
        status = "HIGH"
    elif util >= slack_threshold:
        status = "WARN"
    return util, status, capacity

def recommend_actions(status, util, staff_on_duty):
    recs = []
    if status == "CRITICAL":
        recs = [
            "Activate surge protocol — call reserve staff.",
            "Temporarily postpone elective/non-urgent tasks.",
            "Redistribute tasks to less loaded departments.",
            "Open overflow areas if available."
        ]
    elif status == "HIGH":
        recs = [
            "Request overtime or on-call staff.",
            "Move float nurses from nearby units.",
            "Prioritize critical tasks; delay lower-priority ones."
        ]
    elif status == "WARN":
        recs = [
            "Monitor next-hour arrivals closely.",
            "Prepare 30-min rapid response team standby."
        ]
    elif status == "NO STAFF":
        recs = ["No staff scheduled — schedule immediate coverage."]
    else:
        recs = ["No action required; capacity adequate."]
    # concise single-line summary too
    summary = recs[0] if recs else "—"
    return summary, recs

# ---------- UI ----------
st.title("Predictor — Smart Workload Balancer")
st.markdown("A compact, professional predictor for next-hour workload. Use the inputs below to run a prediction and get quick actionable recommendations.")

model, scaler, feature_cols = load_model()

df_hourly = load_latest_hourly_features()
latest = df_hourly.iloc[-1] if (df_hourly is not None and len(df_hourly) > 0) else None

# Input card
card = st.container()
with card:
    st.markdown("### Input — current state (editable)")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        arrivals = st.number_input("Current hour arrivals", min_value=0, max_value=10000, value=int(latest["arrivals"]) if latest is not None else 0, step=1)
        avg_acuity = st.number_input("Avg acuity (1-4)", min_value=1.0, max_value=5.0, value=float(round(latest["avg_acuity"],2)) if latest is not None else 2.0, step=0.1)
    with col2:
        avg_proc_time = st.number_input("Avg proc time (min)", min_value=1.0, max_value=600.0, value=float(round(latest["avg_proc_time"],1)) if latest is not None else 30.0, step=1.0)
        tasks_created = st.number_input("Tasks created (hour)", min_value=0, max_value=10000, value=int(latest["tasks_created"]) if latest is not None else 0, step=1)
    with col3:
        staff_on_duty = st.number_input("Staff on duty", min_value=0, max_value=1000, value=int(latest["staff_on_duty"]) if latest is not None else 0, step=1)
        avg_capacity_per_staff = st.number_input("Avg capacity / staff / hr", min_value=0.1, max_value=100.0, value=float(round(latest.get("avg_capacity_per_staff", 1.0),2)) if latest is not None else 1.0, step=0.1)

    st.markdown("#### Optional: advanced")
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        hour_of_day = st.slider("Hour of day", 0, 23, int(latest["hour_of_day"]) if latest is not None else datetime.now().hour)
    with adv_col2:
        day_of_week = st.selectbox("Day of week", options=list(range(0,7)), index=int(latest["day_of_week"]) if latest is not None else datetime.now().weekday())

    predict_btn = st.button("Predict next-hour workload", type="primary")

# Prepare a feature row (dictionary) in model's expected format
base_row = {
    "arrivals": arrivals,
    "avg_acuity": avg_acuity,
    "avg_proc_time": avg_proc_time,
    "tasks_created": tasks_created,
    "staff_on_duty": staff_on_duty,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_weekend": int(day_of_week in [5,6])
}
# add rolling features if model expects them: compute simple values from base inputs
base_row.update({
    "arrivals_roll_1h": arrivals,
    "arrivals_roll_3h": arrivals,
    "arrivals_roll_6h": arrivals,
    "arrivals_roll_12h": arrivals,
    "arrivals_roll_24h": arrivals,
    "tasks_roll_1h": tasks_created,
    "tasks_roll_3h": tasks_created,
    "tasks_roll_6h": tasks_created,
    "tasks_roll_12h": tasks_created,
    "tasks_roll_24h": tasks_created
})

# Run prediction if button pressed
if predict_btn:
    if model is None:
        st.error("Model artifact not found (xgb_model.pkl). Train first or place the artifact here.")
    else:
        # Ensure we have feature order
        if feature_cols is None:
            feature_cols = list(base_row.keys())
        # make sure all expected features exist in base_row
        for c in feature_cols:
            if c not in base_row:
                base_row[c] = 0.0
        # build DataFrame and scale
        X_input = prepare_input_vector(pd.Series(base_row), feature_cols, scaler)
        try:
            pred = float(model.predict(X_input.values)[0])
        except Exception:
            # some models accept DataFrame directly
            pred = float(model.predict(X_input)[0])

        util, status, capacity = utilization_and_status(pred, staff_on_duty, avg_capacity_per_staff)
        summary, recs = recommend_actions(status, util, staff_on_duty)

        # top-row metrics
        mcol1, mcol2, mcol3 = st.columns([2,1,1])
        mcol1.metric("Predicted next-hour arrivals", f"{pred:.1f}")
        util_text = "N/A" if math.isinf(util) else f"{util*100:.1f}%"
        badge = status
        if status == "CRITICAL":
            mcol2.markdown(f"**Status:** :red_circle: {badge}")
        elif status == "HIGH":
            mcol2.markdown(f"**Status:** :orange_circle: {badge}")
        elif status == "WARN":
            mcol2.markdown(f"**Status:** :yellow_circle: {badge}")
        elif status == "NO STAFF":
            mcol2.markdown(f"**Status:** :black_circle: {badge}")
        else:
            mcol2.markdown(f"**Status:** :green_circle: {badge}")
        mcol3.metric("Estimated utilization", util_text, delta=f"Capacity ≈ {capacity:.0f}")

        st.markdown("#### Top SHAP contributors (mini)")
        X_model = prepare_input_vector(pd.Series(base_row), feature_cols, scaler)
        shap_df = compute_shap_for_row(model, X_model, feature_cols)
        # show compact table (top 6)
        st.table(shap_df[["feature", "value", "shap"]].head(6).reset_index(drop=True))

        # small horizontal bar for SHAP
        topn = shap_df.head(8).sort_values("shap")
        fig = px.bar(topn, x="shap", y="feature", orientation="h", title="SHAP (signed)", height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Quick recommendations")
        st.write(f"**Summary:** {summary}")
        for r in recs:
            st.write("- " + r)

        st.markdown("---")
        st.markdown("**Export / Audit**")
        if st.button("Export prediction as JSON"):
            out = {
                "timestamp": datetime.utcnow().isoformat(),
                "features": base_row,
                "prediction": pred,
                "utilization": None if math.isinf(util) else util,
                "status": status,
                "recommendations": recs
            }
            fname = f"prediction_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
            with open(fname, "w") as f:
                import json
                json.dump(out, f, indent=2)
            st.success(f"Saved {fname}")

# small history sparkline
st.markdown("---")
st.markdown("### Recent arrivals (sparkline)")
if df_hourly is not None and len(df_hourly) > 0:
    small = df_hourly.tail(48)[["hour", "arrivals"]].set_index("hour")
    fig2 = px.line(small, x=small.index, y="arrivals", title=None, height=180)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No historical data available to show sparkline.")

st.caption("Designed for operational users — clean, fast and actionable. • Powered by XGBoost + SHAP")
