import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_patient_events(start_date, days=14, per_day_mean=180):
    rows = []
    for d in range(days):
        base = start_date + timedelta(days=d)
        n = np.random.poisson(per_day_mean)
        for i in range(n):
            ts = base + timedelta(minutes=int(np.random.uniform(0, 1440)))
            acuity = np.random.choice([1,2,3,4], p=[0.1,0.2,0.4,0.3])
            proc = max(5, np.random.normal(30 - acuity*3, 8))
            rows.append([
                f"p_{d}_{i}", ts, "admit", "ED",
                int(acuity), int(proc), None
            ])
    return pd.DataFrame(rows, columns=[
        "patient_id","timestamp","event_type","dept",
        "acuity_level","processing_time_minutes","assigned_staff_id"
    ])

def generate_staff_schedule(num_staff=40, days=14):
    rows = []
    roles = ["Nurse","Doctor","Technician","Support"]
    for i in range(num_staff):
        role = np.random.choice(roles, p=[0.55,0.20,0.15,0.10])
        for d in range(days):
            start = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0) + timedelta(days=d)
            end = start + timedelta(hours=10)
            rows.append([
                f"s_{i}", role, start, end,
                np.random.randint(10,25), round(np.random.uniform(1,15),1)
            ])
    return pd.DataFrame(rows, columns=[
        "staff_id","role","shift_start","shift_end",
        "baseline_capacity_per_hour","experience_years"
    ])

def generate_task_log(staff_ids, days=14, avg_tasks=300):
    rows = []
    task_types = ["Checkup","Vitals","Medication","Procedure","Transport"]
    for d in range(days):
        base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(14-d))
        n = np.random.poisson(avg_tasks)
        for i in range(n):
            created = base + timedelta(minutes=int(np.random.uniform(0,1440)))
            duration = abs(np.random.normal(20,7))
            completed = created + timedelta(minutes=duration)
            rows.append([
                f"t_{d}_{i}",
                created,
                completed,
                np.random.choice(task_types),
                np.random.randint(1,4),
                np.random.choice(staff_ids),
                "completed"
            ])
    return pd.DataFrame(rows, columns=[
        "task_id","created_at","completed_at","task_type",
        "priority","assigned_staff_id","status"
    ])

def generate_operational_metrics(days=14):
    rows = []
    metrics = ["avg_wait_time","bed_occupancy","resource_utilization"]
    for d in range(days):
        base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=(14-d))
        for h in range(24):
            ts = base + timedelta(hours=h)
            rows.append([metrics[0], ts, np.random.uniform(5,45), "ED"])
            rows.append([metrics[1], ts, np.random.uniform(60,100), "Hospital"])
            rows.append([metrics[2], ts, np.random.uniform(40,95), "Hospital"])
    return pd.DataFrame(rows, columns=["metric_name","timestamp","value","location"])

if __name__ == "__main__":
    start = datetime.now() - timedelta(days=14)

    patient_df = generate_patient_events(start)
    staff_df = generate_staff_schedule()
    task_df = generate_task_log(staff_df["staff_id"].unique())
    op_df = generate_operational_metrics()

    patient_df.to_csv("patient_events.csv", index=False)
    staff_df.to_csv("staff_schedule.csv", index=False)
    task_df.to_csv("task_log.csv", index=False)
    op_df.to_csv("operational_metrics.csv", index=False)

    print("Synthetic data generated:")
    print(len(patient_df), "patient events")
    print(len(staff_df), "staff schedule rows")
    print(len(task_df), "task logs")
    print(len(op_df), "operational metric rows")
