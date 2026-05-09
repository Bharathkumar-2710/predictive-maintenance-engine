"""
Flask REST API — Predictive Maintenance Backend
Endpoints: sensor ingestion, machine status, RUL predictions, alerts, dashboard data.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, jsonify, request
from datetime import datetime

from backend.sensors  import SensorSuite, AlertSeverity
from backend.database import get_db
from models.rul_model import get_trained_model
from data.data_generator import generate_dataset, get_latest_snapshot, DEGRADATION_RATES, START_CYCLES

import pandas as pd
import numpy as np

app = Flask(__name__)

# ──────────────────────────────────────────────
# Startup — seed DB and train model
# ──────────────────────────────────────────────

_sensor_suites: dict[str, SensorSuite] = {}

def initialize():
    db = get_db()
    model, df = get_trained_model()

    # Register machines
    for machine_id in DEGRADATION_RATES:
        db.upsert_machine(machine_id,
                          max_cycle=int(df[df.machine_id == machine_id]["max_cycle"].iloc[0]),
                          degradation_rate=DEGRADATION_RATES[machine_id])
        _sensor_suites[machine_id] = SensorSuite(machine_id)

    # Seed sensor logs (skip if already populated)
    stats = db.summary_stats()
    if stats["total_readings"] == 0:
        print("Seeding database with simulated sensor data...")
        n = db.bulk_insert_sensor_logs(df)
        print(f"  Inserted {n} sensor log rows.")

    # Seed health snapshots
    snap = get_latest_snapshot()
    predictions = model.predict_batch_snapshot(snap, df)
    for _, row in predictions.iterrows():
        db.upsert_health_snapshot({
            "machine_id":     row["machine_id"],
            "snapshot_at":    datetime.utcnow().isoformat(),
            "cycle":          int(row["current_cycle"]),
            "health_score":   float(row["health_score"]),
            "predicted_rul":  int(row["predicted_rul"]),
            "actual_rul":     int(row["actual_rul"]),
            "alert_level":    row["alert_level"],
            "days_to_failure": float(row["days_to_failure"]),
        })
    print("Initialization complete.")


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _jsonify_df(df: pd.DataFrame):
    return jsonify(df.replace({np.nan: None}).to_dict(orient="records"))


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})


@app.route("/api/dashboard")
def dashboard():
    """All data needed to render the full dashboard in one call."""
    db = get_db()
    model, df = get_trained_model()
    snap = get_latest_snapshot()
    predictions = model.predict_batch_snapshot(snap, df)

    at_risk = predictions[predictions["health_score"] < 0.5].to_dict(orient="records")
    metrics  = model.evaluate()
    stats    = db.summary_stats()
    alerts   = db.get_recent_alerts(limit=20)

    # Alert level distribution
    alert_dist = predictions["alert_level"].value_counts().to_dict()

    # Fleet overview
    fleet = predictions.to_dict(orient="records")

    return jsonify({
        "fleet":         fleet,
        "at_risk":       at_risk,
        "model_metrics": metrics,
        "db_stats":      stats,
        "alert_dist":    alert_dist,
        "recent_alerts": alerts,
    })


@app.route("/api/machines")
def list_machines():
    db = get_db()
    return jsonify(db.list_machines())


@app.route("/api/machines/<machine_id>")
def machine_detail(machine_id: str):
    db = get_db()
    model, df = get_trained_model()

    history = db.get_machine_history(machine_id, limit=50)
    trend   = db.get_health_trend(machine_id)
    snap    = get_latest_snapshot()
    machine_snap = snap[snap.machine_id == machine_id]
    if machine_snap.empty:
        return jsonify({"error": "Machine not found"}), 404

    preds = model.predict_batch_snapshot(machine_snap, df)
    pred_row = preds.iloc[0].to_dict() if not preds.empty else {}

    # Sensor suite snapshot
    suite = _sensor_suites.get(machine_id)
    sensor_info = suite.snapshot() if suite else {}

    return jsonify({
        "machine_id":    machine_id,
        "prediction":    pred_row,
        "history":       history[-20:],
        "health_trend":  trend,
        "sensor_info":   sensor_info,
    })


@app.route("/api/ingest", methods=["POST"])
def ingest_sensor_data():
    """
    Receive live sensor readings for a machine.
    Body: { machine_id, cycle, temperature, vibration, pressure, rpm, oil_viscosity, current_draw }
    """
    data = request.get_json(force=True)
    machine_id = data.get("machine_id")
    if not machine_id:
        return jsonify({"error": "machine_id required"}), 400

    suite = _sensor_suites.get(machine_id)
    if not suite:
        suite = SensorSuite(machine_id)
        _sensor_suites[machine_id] = suite

    # Record readings & collect alerts
    sensor_readings = {k: float(v) for k, v in data.items()
                       if k in ("temperature", "vibration", "pressure",
                                "rpm", "oil_viscosity", "current_draw")}
    alerts = suite.ingest(sensor_readings)

    # Persist alerts
    db = get_db()
    for alert in alerts:
        db.insert_alert(alert.to_dict())

    # Persist sensor log
    row = {
        "machine_id":   machine_id,
        "timestamp":    datetime.utcnow().isoformat(),
        "cycle":        int(data.get("cycle", 0)),
        "health_score": float(data.get("health_score", 1.0)),
        "rul":          int(data.get("rul", 999)),
        "alert_level":  data.get("alert_level", "NORMAL"),
        **sensor_readings,
    }
    db.insert_sensor_log(row)

    return jsonify({
        "status":  "ok",
        "alerts":  [a.to_dict() for a in alerts],
        "machine": machine_id,
    })


@app.route("/api/predictions")
def predictions():
    model, df = get_trained_model()
    snap = get_latest_snapshot()
    preds = model.predict_batch_snapshot(snap, df)
    return _jsonify_df(preds)


@app.route("/api/alerts")
def recent_alerts():
    db = get_db()
    limit = int(request.args.get("limit", 50))
    return jsonify(db.get_recent_alerts(limit=limit))


@app.route("/api/feature-importance")
def feature_importance():
    model, _ = get_trained_model()
    fi = model.feature_importances()
    return _jsonify_df(fi)


@app.route("/api/model-metrics")
def model_metrics():
    model, _ = get_trained_model()
    return jsonify(model.evaluate())


if __name__ == "__main__":
    initialize()
    app.run(debug=False, port=5050)
