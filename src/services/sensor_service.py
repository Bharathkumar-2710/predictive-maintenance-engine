"""
OOP sensor class hierarchy and alert system for the Predictive Maintenance System.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class AlertSeverity(Enum):
    NORMAL = 1
    WATCH = 2
    WARNING = 3
    CRITICAL = 4

    @classmethod
    def from_health(cls, health_score: float) -> "AlertSeverity":
        if health_score >= 0.75:
            return cls.NORMAL
        if health_score >= 0.50:
            return cls.WATCH
        if health_score >= 0.25:
            return cls.WARNING
        return cls.CRITICAL

    @property
    def color(self) -> str:
        return {
            AlertSeverity.NORMAL: "#22c55e",
            AlertSeverity.WATCH: "#eab308",
            AlertSeverity.WARNING: "#f97316",
            AlertSeverity.CRITICAL: "#ef4444",
        }[self]

    @property
    def label(self) -> str:
        return self.name


@dataclass
class Alert:
    """Represents a triggered alert for a machine."""
    machine_id: str
    severity: AlertSeverity
    message: str
    sensor_name: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "severity": self.severity.label,
            "color": self.severity.color,
            "message": self.message,
            "sensor_name": self.sensor_name,
            "value": round(self.value, 4),
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


class Sensor(ABC):
    """Abstract base class for all industrial sensors."""

    def __init__(self, sensor_id: str, machine_id: str, unit: str):
        self.sensor_id = sensor_id
        self.machine_id = machine_id
        self.unit = unit
        self._history: list[tuple[float, float]] = []
        self._calibration_offset: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def warning_threshold(self) -> float:
        ...

    @property
    @abstractmethod
    def critical_threshold(self) -> float:
        ...

    @abstractmethod
    def validate(self, value: float) -> bool:
        ...

    def calibrate(self, offset: float) -> None:
        self._calibration_offset = offset

    def record(self, value: float) -> float:
        calibrated = value + self._calibration_offset
        self._history.append((time.time(), calibrated))
        if len(self._history) > 500:
            self._history.pop(0)
        return calibrated

    def latest(self) -> Optional[float]:
        return self._history[-1][1] if self._history else None

    def moving_average(self, window: int = 5) -> Optional[float]:
        if len(self._history) < window:
            return None
        vals = [v for _, v in self._history[-window:]]
        return sum(vals) / len(vals)

    def check_alert(self, value: float) -> Optional[Alert]:
        if not self.validate(value):
            return Alert(
                machine_id=self.machine_id,
                severity=AlertSeverity.WARNING,
                message=f"{self.name} reading {value} {self.unit} is out of physical range",
                sensor_name=self.name,
                value=value,
                threshold=self.warning_threshold,
            )
        if value >= self.critical_threshold or value <= -self.critical_threshold * 0.5:
            return Alert(
                machine_id=self.machine_id,
                severity=AlertSeverity.CRITICAL,
                message=f"{self.name} CRITICAL: {value:.2f} {self.unit} exceeded {self.critical_threshold}",
                sensor_name=self.name,
                value=value,
                threshold=self.critical_threshold,
            )
        if value >= self.warning_threshold:
            return Alert(
                machine_id=self.machine_id,
                severity=AlertSeverity.WARNING,
                message=f"{self.name} WARNING: {value:.2f} {self.unit} exceeded {self.warning_threshold}",
                sensor_name=self.name,
                value=value,
                threshold=self.warning_threshold,
            )
        return None

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.name,
            "machine_id": self.machine_id,
            "unit": self.unit,
            "latest_value": self.latest(),
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }


# Concrete Sensor Implementations
class TemperatureSensor(Sensor):
    def __init__(self, sensor_id: str, machine_id: str):
        super().__init__(sensor_id, machine_id, unit="°F")

    @property
    def name(self) -> str:
        return "temperature"

    @property
    def warning_threshold(self) -> float:
        return 490.0

    @property
    def critical_threshold(self) -> float:
        return 530.0

    def validate(self, value: float) -> bool:
        return -40.0 <= value <= 900.0


class VibrationSensor(Sensor):
    def __init__(self, sensor_id: str, machine_id: str):
        super().__init__(sensor_id, machine_id, unit="g")

    @property
    def name(self) -> str:
        return "vibration"

    @property
    def warning_threshold(self) -> float:
        return 0.45

    @property
    def critical_threshold(self) -> float:
        return 0.60

    def validate(self, value: float) -> bool:
        return 0.0 <= value <= 10.0


class PressureSensor(Sensor):
    def __init__(self, sensor_id: str, machine_id: str):
        super().__init__(sensor_id, machine_id, unit="PSI")

    @property
    def name(self) -> str:
        return "pressure"

    @property
    def warning_threshold(self) -> float:
        return 10.0

    @property
    def critical_threshold(self) -> float:
        return 7.5

    def validate(self, value: float) -> bool:
        return 0.0 <= value <= 200.0

    def check_alert(self, value: float) -> Optional[Alert]:
        if not self.validate(value):
            return Alert(self.machine_id, AlertSeverity.WARNING,
                         f"Pressure {value} PSI out of range", self.name, value, self.warning_threshold)
        if value <= self.critical_threshold:
            return Alert(self.machine_id, AlertSeverity.CRITICAL,
                         f"Pressure CRITICAL LOW: {value:.2f} PSI", self.name, value, self.critical_threshold)
        if value <= self.warning_threshold:
            return Alert(self.machine_id, AlertSeverity.WARNING,
                         f"Pressure WARNING LOW: {value:.2f} PSI", self.name, value, self.warning_threshold)
        return None


class RPMSensor(Sensor):
    def __init__(self, sensor_id: str, machine_id: str):
        super().__init__(sensor_id, machine_id, unit="RPM")

    @property
    def name(self) -> str:
        return "rpm"

    @property
    def warning_threshold(self) -> float:
        return 1650.0

    @property
    def critical_threshold(self) -> float:
        return 1500.0

    def validate(self, value: float) -> bool:
        return 0.0 <= value <= 5000.0

    def check_alert(self, value: float) -> Optional[Alert]:
        if not self.validate(value):
            return Alert(self.machine_id, AlertSeverity.WARNING,
                         f"RPM {value} out of range", self.name, value, self.warning_threshold)
        if value <= self.critical_threshold:
            return Alert(self.machine_id, AlertSeverity.CRITICAL,
                         f"RPM CRITICAL LOW: {value:.0f} RPM", self.name, value, self.critical_threshold)
        if value <= self.warning_threshold:
            return Alert(self.machine_id, AlertSeverity.WARNING,
                         f"RPM WARNING LOW: {value:.0f} RPM", self.name, value, self.warning_threshold)
        return None


class OilViscositySensor(Sensor):
    def __init__(self, sensor_id: str, machine_id: str):
        super().__init__(sensor_id, machine_id, unit="cSt")

    @property
    def name(self) -> str:
        return "oil_viscosity"

    @property
    def warning_threshold(self) -> float:
        return 36.0

    @property
    def critical_threshold(self) -> float:
        return 30.0

    def validate(self, value: float) -> bool:
        return 0.0 <= value <= 200.0

    def check_alert(self, value: float) -> Optional[Alert]:
        if not self.validate(value):
            return Alert(self.machine_id, AlertSeverity.WARNING,
                         f"Oil viscosity {value} cSt out of range", self.name, value, self.warning_threshold)
        if value <= self.critical_threshold:
            return Alert(self.machine_id, AlertSeverity.CRITICAL,
                         f"Oil viscosity CRITICAL LOW: {value:.1f} cSt", self.name, value, self.critical_threshold)
        if value <= self.warning_threshold:
            return Alert(self.machine_id, AlertSeverity.WARNING,
                         f"Oil viscosity LOW: {value:.1f} cSt", self.name, value, self.warning_threshold)
        return None


class CurrentDrawSensor(Sensor):
    def __init__(self, sensor_id: str, machine_id: str):
        super().__init__(sensor_id, machine_id, unit="A")

    @property
    def name(self) -> str:
        return "current_draw"

    @property
    def warning_threshold(self) -> float:
        return 15.5

    @property
    def critical_threshold(self) -> float:
        return 17.0

    def validate(self, value: float) -> bool:
        return 0.0 <= value <= 50.0


# Sensor Factory
SENSOR_REGISTRY = {
    "temperature": TemperatureSensor,
    "vibration": VibrationSensor,
    "pressure": PressureSensor,
    "rpm": RPMSensor,
    "oil_viscosity": OilViscositySensor,
    "current_draw": CurrentDrawSensor,
}


def create_sensor(sensor_type: str, machine_id: str) -> Sensor:
    """Factory function to instantiate the correct Sensor subclass."""
    cls = SENSOR_REGISTRY.get(sensor_type)
    if cls is None:
        raise ValueError(f"Unknown sensor type: {sensor_type!r}. "
                         f"Valid types: {list(SENSOR_REGISTRY.keys())}")
    sensor_id = f"{machine_id}_{sensor_type.upper()}"
    return cls(sensor_id=sensor_id, machine_id=machine_id)


class SensorSuite:
    """Manages a complete set of sensors for one machine."""

    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.sensors: dict[str, Sensor] = {
            name: create_sensor(name, machine_id)
            for name in SENSOR_REGISTRY
        }

    def ingest(self, readings: dict[str, float]) -> list[Alert]:
        """Record readings for all sensors; return any triggered alerts."""
        alerts = []
        for sensor_name, value in readings.items():
            if sensor_name not in self.sensors:
                continue
            sensor = self.sensors[sensor_name]
            sensor.record(value)
            alert = sensor.check_alert(value)
            if alert:
                alerts.append(alert)
        return alerts

    def snapshot(self) -> dict:
        return {name: s.to_dict() for name, s in self.sensors.items()}