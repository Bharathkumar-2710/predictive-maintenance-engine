"""
API route handlers for the Predictive Maintenance System.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import pandas as pd
import numpy as np

from src.core.database import get_db
from src.models.rul_model import get_trained_model
from src.services.data_generator import get_latest_snapshot, DEGRADATION_RATES
from src.services.sensor_service import SensorSuite, SENSOR_REGISTRY
from src.schemas.schemas import (
    SensorReading,
    DashboardResponse,
    HealthResponse,
    ErrorResponse,
)

router = APIRouter()

# Global sensor suites
_sensor_suites: dict[str, SensorSuite] = {}


def _jsonify_df(df: pd.DataFrame):
    return df.replace({np.nan: None}).to_dict(orient="records")


def initialize():
    """Initialize database and model on startup."""
    db = get_db()
    model, df = get_trained_model()

    # Register machines
    for machine_id in DEGRADATION_RATES:
        db.upsert_machine(
            machine_id,
            max_cycle=int(df[df.machine_id == machine_id]["max_cycle"].iloc[0]),
            degradation_rate=DEGRADATION_RATES[machine_id],
        )
        _sensor_suites[machine_id] = SensorSuite(machine_id)

    # Seed sensor logs
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
            "machine_id": row["machine_id"],
            "snapshot_at": datetime.utcnow().isoformat(),
            "cycle": int(row["current_cycle"]),
            "health_score": float(row["health_score"]),
            "predicted_rul": int(row["predicted_rul"]),
            "actual_rul": int(row["actual_rul"]),
            "alert_level": row["alert_level"],
            "days_to_failure": float(row["days_to_failure"]),
        })
    print("Initialization complete.")


@router.get("/health", response_model=HealthResponse)
def api_health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/dashboard")
def dashboard():
    """Full dashboard data in one call."""
    db = get_db()
    model, df = get_trained_model()
    snap = get_latest_snapshot()
    predictions = model.predict_batch_snapshot(snap, df)

    at_risk = predictions[predictions["health_score"] < 0.5].to_dict(orient="records")
    metrics = model.evaluate()
    stats = db.summary_stats()
    alerts = db.get_recent_alerts(limit=20)
    alert_dist = predictions["alert_level"].value_counts().to_dict()
    fleet = predictions.to_dict(orient="records")

    return {
        "fleet": fleet,
        "at_risk": at_risk,
        "model_metrics": metrics,
        "db_stats": stats,
        "alert_dist": alert_dist,
        "recent_alerts": alerts,
    }


@router.get("/machines")
def list_machines():
    """List all registered machines."""
    db = get_db()
    return db.list_machines()


@router.get("/machines/{machine_id}")
def machine_detail(machine_id: str):
    """Get detailed information for a specific machine."""
    db = get_db()
    model, df = get_trained_model()

    history = db.get_machine_history(machine_id, limit=50)
    trend = db.get_health_trend(machine_id)
    snap = get_latest_snapshot()
    machine_snap = snap[snap.machine_id == machine_id]
    if machine_snap.empty:
        raise HTTPException(status_code=404, detail="Machine not found")

    preds = model.predict_batch_snapshot(machine_snap, df)
    pred_row = preds.iloc[0].to_dict() if not preds.empty else {}

    suite = _sensor_suites.get(machine_id)
    sensor_info = suite.snapshot() if suite else {}

    return {
        "machine_id": machine_id,
        "prediction": pred_row,
        "history": history[-20:],
        "health_trend": trend,
        "sensor_info": sensor_info,
    }


@router.post("/ingest")
def ingest_sensor_data(data: SensorReading):
    """Ingest live sensor readings for a machine."""
    db = get_db()
    machine_id = data.machine_id

    suite = _sensor_suites.get(machine_id)
    if not suite:
        suite = SensorSuite(machine_id)
        _sensor_suites[machine_id] = suite

    # Record readings & collect alerts
    sensor_readings = {k: float(v) for k, v in data.model_dump().items()
                       if k in SENSOR_REGISTRY}
    alerts = suite.ingest(sensor_readings)

    # Persist alerts
    for alert in alerts:
        db.insert_alert(alert.to_dict())

    # Persist sensor log
    row = {
        "machine_id": machine_id,
        "timestamp": datetime.utcnow().isoformat(),
        "cycle": data.cycle,
        "health_score": data.health_score,
        "rul": data.rul,
        "alert_level": data.alert_level,
        **sensor_readings,
    }
    db.insert_sensor_log(row)

    return {
        "status": "ok",
        "alerts": [a.to_dict() for a in alerts],
        "machine": machine_id,
    }


@router.get("/predictions")
def predictions():
    """Get RUL predictions for all machines."""
    model, df = get_trained_model()
    snap = get_latest_snapshot()
    preds = model.predict_batch_snapshot(snap, df)
    return _jsonify_df(preds)


@router.get("/alerts")
def recent_alerts(limit: int = 50):
    """Get recent alerts."""
    db = get_db()
    return db.get_recent_alerts(limit=limit)


@router.get("/feature-importance")
def feature_importance():
    """Get feature importance scores."""
    model, _ = get_trained_model()
    fi = model.feature_importances()
    return _jsonify_df(fi)


@router.get("/model-metrics")
def model_metrics():
    """Get model performance metrics."""
    model, _ = get_trained_model()
    return model.evaluate()


# Initialize on module load
initialize()