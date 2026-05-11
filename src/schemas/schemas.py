"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class MachineBase(BaseModel):
    machine_id: str


class MachineResponse(MachineBase):
    registered_at: str
    max_cycle: int
    degradation_rate: float
    is_active: int


class SensorReading(BaseModel):
    machine_id: str
    cycle: int
    temperature: Optional[float] = None
    vibration: Optional[float] = None
    pressure: Optional[float] = None
    rpm: Optional[float] = None
    oil_viscosity: Optional[float] = None
    current_draw: Optional[float] = None
    health_score: Optional[float] = 1.0
    rul: Optional[int] = 999
    alert_level: Optional[str] = "NORMAL"


class AlertResponse(BaseModel):
    machine_id: str
    severity: str
    color: str
    message: str
    sensor_name: str
    value: float
    threshold: float
    timestamp: float


class PredictionResponse(BaseModel):
    machine_id: str
    current_cycle: int
    actual_rul: int
    predicted_rul: int
    health_score: float
    alert_level: str
    hours_to_failure: float
    days_to_failure: float


class DashboardResponse(BaseModel):
    fleet: list
    at_risk: list
    model_metrics: dict
    db_stats: dict
    alert_dist: dict
    recent_alerts: list


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ErrorResponse(BaseModel):
    error: str