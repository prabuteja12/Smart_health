# dashboard.py
import os
import json
import joblib
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap
from datetime import datetime, timedelta

st.set_page_config(page_title="Smart Workload Balancer", layout="wide", initial_sidebar_state="expanded")

MODEL_ARTIFACT = "xgb_model.pkl"
PATIENT_EVENTS = "patient_events.csv"
STAFF_SCHEDULE = "staff_schedule.csv"
TASK_LOG = "task_log.csv"
OP_METRICS = "operational_metrics.csv"

# ---------- Helpers ----------
@st.cache_resource
def load_model_artifact(path=MODEL_ARTIFACT):
    if not os.path.exists(path):
        return None, None, None
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj.get("model"), obj.get("scaler"), obj.get("features")
    return obj, None, None

@st.cache_data
def load_csvs(pe=PATIENT_EVENTS, ss=STAFF_SCHEDULE, tl=TASK_LOG, om=OP_METRICS):
    pe_df = pd.read_csv(pe, parse_dates=["timestamp"]) if os.path.exists(pe) else pd.DataFrame()
    ss_df = pd.read_csv(ss, parse_dates=["shift_start", "shift_end"]) if os.path.exists(ss) else pd.DataFrame()
    tl_df = pd.read_csv(tl, parse_dates=["created_at", "completed_at"]) if os.path.exists(tl) else pd.DataFrame()
    om_df = pd.read_csv(om, parse_dates=["timestamp"]) if os.path.exists(om) else pd.DataFrame()
    return pe_df, ss_df, tl_df, om_df

def build_hourly_features(pe, ss, tl, om):
    if pe.empty:
        return pd.DataFrame()
    df_pe = pe.copy()
    df_pe["hour"] = df_pe["timestamp"].dt.floor("h")
    arrivals = df_pe.groupby("hour").agg(
        arrivals=("patient_id", "count"),
        avg_acuity=("acuity_level", "mean"),
        avg_proc_time=("processing_time_minutes", "mean")
    ).reset_index()

    df_tl = tl.copy() if not tl.empty else pd.DataFrame(columns=["created_at", "task_id"])
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

    om_hour = om.copy() if not om.empty else pd.DataFrame(columns=["timestamp", "metric_name", "value"])
    if not om_hour.empty:
        om_hour["hour"] = om_hour["timestamp"].dt.floor("h")
        om_agg = om_hour.groupby(["hour", "metric_name"]).agg(value=("value", "mean")).reset_index()
        om_pivot = om_agg.pivot(index="hour", columns="metric_name", values="value").reset_index().rename_axis(None, axis=1)
    else:
        om_pivot = pd.DataFrame()

    df = arrivals.merge(tasks, on="hour", how="left")
    df = df.merge(staff_count, on="hour", how="left")
    if "hour" in om_pivot.columns:
        df = df.merge(om_pivot, on="hour", how="left")

    df = df.sort_values("hour").reset_index(drop=True)
    df["arrivals"] = df["arrivals"].fillna(0)
    df["tasks_created"] = df.get("tasks_created", 0).fillna(0)
    df["avg_acuity"] = df["avg_acuity"].fillna(df["avg_acuity"].median() if "avg_acuity" in df else 0)
    df["avg_proc_time"] = df["avg_proc_time"].fillna(df["avg_proc_time"].median() if "avg_proc_time" in df else 0)
    df["staff_on_duty"] = df["staff_on_duty"].fillna(0)
    df["avg_capacity_per_staff"] = df["avg_capacity_per_staff"].fillna(0)

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    df = df.set_index("hour")
    for w in [1,3,6,12,24]:
        df[f"arrivals_roll_{w}h"] = df["arrivals"].rolling(window=w, min_periods=1).mean()
        df[f"tasks_roll_{w}h"] = df["tasks_created"].rolling(window=w, min_periods=1).mean()
    df = df.reset_index()
    return df

def feature_vector_from_row(row, feature_cols, scaler):
    X = pd.DataFrame([row[feature_cols].astype(float)])
    if scaler is not None:
        try:
            Xs = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)
            return Xs
        except Exception:
            return X
    return X

def predict_row(model, scaler, row, feature_cols):
    X_for_model = feature_vector_from_row(row, feature_cols, scaler)
    pred = model.predict(X_for_model.values)
    return float(pred[0])

def compute_shap_local(model, scaler, row, feature_cols):
    X_for_model = feature_vector_from_row(row, feature_cols, scaler)
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.Explainer(model)
    try:
        vals = explainer.shap_values(X_for_model)
    except Exception:
        res = explainer(X_for_model)
        vals = res.values if hasattr(res, "values") else res
    if isinstance(vals, list):
        vals = vals[0]
    shap_vals = vals[0] if vals.ndim == 2 else vals
    df = pd.DataFrame({"feature": feature_cols, "shap": shap_vals, "value": row[feature_cols].astype(float).values})
    df["abs_shap"] = df["shap"].abs()
    df = df.sort_values("abs_shap", ascending=False)
    return df

def compute_staff_heatmap(ss_df, start, end):
    if ss_df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in ss_df.iterrows():
        start_r = pd.to_datetime(r["shift_start"]).floor("h")
        end_r = pd.to_datetime(r["shift_end"]).floor("h")
        hrs = int(((end_r - start_r) / pd.Timedelta(hours=1)) + 1)
        for i in range(max(0, hrs)):
            ts = start_r + pd.Timedelta(hours=i)
            if ts < start or ts > end:
                continue
            rows.append((ts, r.get("role", "Unknown")))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["hour", "role"])
    heat = df.groupby([df["hour"].dt.hour, "role"]).size().reset_index(name="count")
    pivot = heat.pivot(index="hour", columns="role", values="count").fillna(0).sort_index()
    return pivot

def alert_engine(prediction, staff_on_duty, avg_capacity_per_staff, slack_threshold=0.8):
    capacity = staff_on_duty * avg_capacity_per_staff
    if capacity == 0:
        score = float("inf")
    else:
        score = prediction / capacity
    status = "OK"
    if math.isinf(score):
        status = "NO STAFF"
    elif score >= 1.2:
        status = "CRITICAL"
    elif score >= 0.9:
        status = "HIGH"
    elif score >= slack_threshold:
        status = "WARN"
    return {"prediction": prediction, "capacity": capacity, "utilization": None if math.isinf(score) else round(score, 3), "status": status}

# ---------- UI ----------
st.sidebar.header("Controls")
model, scaler, feature_cols = load_model_artifact()
pe, ss, tl, om = load_csvs()

st.sidebar.markdown("### Data sources")
st.sidebar.write(f"- patient_events: {'found' if not pe.empty else 'missing'}")
st.sidebar.write(f"- staff_schedule: {'found' if not ss.empty else 'missing'}")
st.sidebar.write(f"- task_log: {'found' if not tl.empty else 'missing'}")
st.sidebar.write(f"- operational_metrics: {'found' if not om.empty else 'missing'}")

st.sidebar.markdown("---")
department = st.sidebar.selectbox("Department", options=["ED", "ICU", "Ward", "All"], index=0)
hours_back = st.sidebar.slider("Forecast window (hours)", 1, 24, 6)
start_date = st.sidebar.date_input("Start date (for charts)", value=(datetime.now() - timedelta(days=7)).date())
end_date = st.sidebar.date_input("End date (for charts)", value=datetime.now().date())
refresh = st.sidebar.button("Refresh data")

st.header("Smart Workload Balancer — Dashboard")
st.subheader("Operational Forecasts • Explainability • Alerts")
if model is None:
    st.error("Model artifact not found. Place xgb_model.pkl in the project root or run training first.")
    st.stop()

df_hourly = build_hourly_features(pe, ss, tl, om)
if df_hourly.empty:
    st.warning("No hourly features could be built from data. Check CSVs.")
    st.stop()

# filter by date range
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df_range = df_hourly[(df_hourly["hour"] >= start_dt) & (df_hourly["hour"] <= end_dt)].copy()
if df_range.empty:
    st.warning("No data in selected date range.")
    st.stop()

# main layout
left_col, mid_col, right_col = st.columns([2, 1, 1])

with left_col:
    st.markdown("### Forecast (Next hours)")
    last_row = df_hourly.iloc[-1]
    # build forecast series: we show next N hours by iteratively predicting using arrivals shifts (simple sim)
    forecast_hours = hours_back
    forecast_rows = []
    temp_df = df_hourly.copy()
    for i in range(forecast_hours):
        base = temp_df.iloc[[-1]].copy().reset_index(drop=True)
        # shift arrivals to previous predicted if exists
        features = feature_vector_from_row(base.iloc[0], feature_cols, scaler)
        pred = model.predict(features.values)[0]
        next_hour = base["hour"].iloc[0] + pd.Timedelta(hours=1)
        new_row = base.copy()
        new_row["hour"] = next_hour
        new_row["arrivals"] = pred
        # update rolling features simply by appending and re-calculating rolls
        temp_df = pd.concat([temp_df, new_row], ignore_index=True)
        temp_df = temp_df.set_index("hour").sort_index()
        for w in [1,3,6,12,24]:
            temp_df[f"arrivals_roll_{w}h"] = temp_df["arrivals"].rolling(window=w, min_periods=1).mean()
        temp_df = temp_df.reset_index()
        forecast_rows.append((next_hour, float(pred)))
    fc_df = pd.DataFrame(forecast_rows, columns=["hour", "predicted_arrivals"])
    chart_df = pd.concat([
        df_range[["hour", "arrivals"]].rename(columns={"arrivals": "value"}).assign(series="observed"),
        fc_df.rename(columns={"predicted_arrivals": "value"}).assign(series="predicted")
    ])
    fig = px.line(chart_df, x="hour", y="value", color="series", title="Observed arrivals and short-term forecast", markers=True)
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=35, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # metrics row
    latest_pred = float(fc_df["predicted_arrivals"].iloc[-1])
    latest_staff = float(last_row.get("staff_on_duty", 0))
    latest_capacity_per = float(last_row.get("avg_capacity_per_staff", 1))
    capacity = latest_staff * latest_capacity_per
    utilization = None if capacity == 0 else round(latest_pred / capacity, 3)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest observed (hour)", int(last_row["arrivals"]))
    col2.metric("Predicted (next hour)", f"{latest_pred:.1f}")
    col3.metric("Staff on duty", int(latest_staff))
    col4.metric("Utilization", "N/A" if utilization is None else f"{utilization*100:.1f}%")

with mid_col:
    st.markdown("### Staff Heatmap (by hour × role)")
    heatmap = compute_staff_heatmap(ss, start_dt, end_dt)
    if heatmap.empty:
        st.info("No staff schedule details to show.")
    else:
        fig2 = px.imshow(heatmap.T, labels=dict(x="Hour of day", y="Role", color="Count"),
                         x=heatmap.index, y=heatmap.columns,
                         title="Staff on duty heatmap")
        fig2.update_layout(height=420, margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Latest Operational Metrics")
    om_display = last_row[[c for c in last_row.index if c not in ["hour", "arrivals", "tasks_created", "staff_on_duty", "avg_acuity", "avg_proc_time", "hour_of_day", "day_of_week", "is_weekend"]]].to_dict()
    if om_display:
        st.json({k: (float(v) if pd.notna(v) else None) for k, v in om_display.items()})
    else:
        st.info("No operational metrics available in latest row.")

with right_col:
    st.markdown("### Alerts")
    # simple alerting for the forecasted next hour (last prediction)
    alert = alert_engine(latest_pred, latest_staff, latest_capacity_per, slack_threshold=0.8)
    status = alert["status"]
    if status == "CRITICAL":
        st.error(f"CRITICAL — Predicted load {alert['prediction']:.1f} exceeds capacity {alert['capacity']:.1f}")
    elif status == "HIGH":
        st.warning(f"HIGH — Predicted load {alert['prediction']:.1f} near capacity {alert['capacity']:.1f}")
    elif status == "WARN":
        st.info(f"WARN — Predicted load {alert['prediction']:.1f}, utilization {alert['utilization']:.2f}")
    elif status == "NO STAFF":
        st.error("NO STAFF — No staffing scheduled for next hour")
    else:
        st.success("OK — Predicted load within capacity")

    st.markdown("### Alert details")
    st.table(pd.DataFrame([alert]))

    st.markdown("### What-if simulator")
    extra_arrivals = st.number_input("Simulate extra arrivals next hour (+)", min_value=0, max_value=500, value=0, step=1)
    absent_staff = st.number_input("Simulate absent staff next hour (-)", min_value=0, max_value=int(latest_staff) if latest_staff>0 else 0, value=0, step=1)
    sim_pred = latest_pred + extra_arrivals
    sim_staff = max(0, latest_staff - absent_staff)
    sim_capacity = sim_staff * latest_capacity_per
    sim_util = None if sim_capacity == 0 else round(sim_pred / sim_capacity, 3)
    st.markdown(f"**Simulated predicted arrivals:** {sim_pred:.1f}")
    st.markdown(f"**Simulated staff on duty:** {sim_staff}")
    st.markdown(f"**Simulated utilization:** {'N/A' if sim_util is None else f'{sim_util*100:.1f}%'}")
    if sim_util is not None and sim_util >= 1.2:
        st.error("Simulation: CRITICAL — immediate action recommended")
    elif sim_util is not None and sim_util >= 0.9:
        st.warning("Simulation: HIGH — consider reassigning resources")
    else:
        st.success("Simulation: OK")

# ---------- Explainability section ----------
st.markdown("---")
st.subheader("Explainability — SHAP")
left, right = st.columns([2, 1])
with left:
    st.markdown("#### Select timestamp to explain")
    available_hours = df_hourly["hour"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    selected = st.selectbox("Hour (row) to explain", options=available_hours, index=len(available_hours)-1)
    sel_dt = pd.to_datetime(selected)
    sel_row = df_hourly[df_hourly["hour"] == sel_dt]
    if sel_row.empty:
        st.warning("Selected row not available.")
    else:
        sel_row = sel_row.iloc[0]
        shap_df = compute_shap_local(model, scaler, sel_row, feature_cols)
        st.markdown("##### Top SHAP contributors")
        st.table(shap_df[["feature", "value", "shap"]].head(10).reset_index(drop=True))

        # bar chart of top contributors
        topn = shap_df.head(15)
        fig3 = px.bar(topn.sort_values("shap"), x="shap", y="feature", orientation="h", title="SHAP contributions (signed)")
        fig3.update_layout(height=480, margin=dict(l=0, r=0, t=35, b=0))
        st.plotly_chart(fig3, use_container_width=True)

with right:
    st.markdown("#### Prediction & Contribution")
    pred_val = predict_row(model, scaler, sel_row, feature_cols)
    st.metric("Predicted arrivals next hour", f"{pred_val:.2f}")
    st.markdown("#### Feature snapshot")
    snap = pd.DataFrame({"feature": feature_cols, "value": sel_row[feature_cols].astype(float).values})
    st.dataframe(snap.style.format({"value": "{:.2f}"}), height=360)

# ---------- Data & exports ----------
st.markdown("---")
st.subheader("Data Explorer & Export")
tabs = st.tabs(["Hourly features", "Patient events (sample)", "Staff schedule", "Operational metrics"])
with tabs[0]:
    st.dataframe(df_range.sort_values("hour").reset_index(drop=True), height=420)
    if st.button("Export hourly features CSV"):
        out = "hourly_features_export.csv"
        df_range.to_csv(out, index=False)
        st.success(f"Exported to {out}")

with tabs[1]:
    if not pe.empty:
        st.dataframe(pe.sort_values("timestamp", ascending=False).head(500))
    else:
        st.info("No patient events CSV found.")

with tabs[2]:
    if not ss.empty:
        st.dataframe(ss.head(200))
    else:
        st.info("No staff schedule CSV found.")

with tabs[3]:
    if not om.empty:
        st.dataframe(om.head(500))
    else:
        st.info("No operational metrics CSV found.")

st.markdown("---")
st.caption("Smart Workload Balancer • Forecasts powered by XGBoost • Explainability via SHAP")
