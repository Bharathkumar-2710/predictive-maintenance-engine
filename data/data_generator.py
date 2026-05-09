"""
NASA Turbofan-inspired sensor data generator.
Simulates CMAPSS (Commercial Modular Aero-Propulsion System Simulation) style data
with temperature, vibration, pressure, and RPM sensors across multiple machines.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
NUM_MACHINES   = 10
MAX_CYCLES     = 300   # max operational cycles before forced failure
SENSORS        = ["temperature", "vibration", "pressure", "rpm", "oil_viscosity", "current_draw"]

# Degradation slopes per machine (how fast each degrades)
DEGRADATION_RATES = {
    f"MACH-{i:03d}": round(random.uniform(0.003, 0.012), 4)
    for i in range(1, NUM_MACHINES + 1)
}

# Starting cycles (so machines are at different lifecycle stages)
START_CYCLES = {
    f"MACH-{i:03d}": random.randint(0, 200)
    for i in range(1, NUM_MACHINES + 1)
}

# ──────────────────────────────────────────────
# Sensor baseline & noise config
# ──────────────────────────────────────────────
SENSOR_CONFIG = {
    "temperature":   {"baseline": 450.0,  "noise": 5.0,  "degrade_factor": 80.0},
    "vibration":     {"baseline": 0.02,   "noise": 0.005,"degrade_factor": 0.6},
    "pressure":      {"baseline": 14.7,   "noise": 0.3,  "degrade_factor": -3.0},  # drops with wear
    "rpm":           {"baseline": 1800.0, "noise": 20.0, "degrade_factor": -100.0},
    "oil_viscosity": {"baseline": 45.0,   "noise": 1.0,  "degrade_factor": -12.0},
    "current_draw":  {"baseline": 12.0,   "noise": 0.4,  "degrade_factor": 5.0},
}


def compute_health_score(cycle: int, max_cycle: int, degradation_rate: float) -> float:
    """
    Nonlinear health degradation: stays near 1.0 early, drops sharply near end.
    Formula inspired by NASA CMAPSS RUL modeling literature.
    """
    progress = cycle / max_cycle
    health = 1.0 - (progress ** 2.2) * (1 + degradation_rate * 10)
    return float(np.clip(health, 0.0, 1.0))


def compute_rul(cycle: int, max_cycle: int) -> int:
    """Remaining Useful Life in cycles."""
    return max(0, max_cycle - cycle)


def simulate_sensor_reading(sensor: str, health: float) -> float:
    """
    Generate a single sensor reading given current machine health.
    Degradation shifts the signal from baseline as health decreases.
    """
    cfg = SENSOR_CONFIG[sensor]
    degradation_signal = cfg["degrade_factor"] * (1.0 - health)
    noise = np.random.normal(0, cfg["noise"])
    value = cfg["baseline"] + degradation_signal + noise
    return round(float(value), 4)


def generate_alert_level(health: float) -> str:
    """Map health score to alert tier."""
    if health >= 0.75:
        return "NORMAL"
    elif health >= 0.50:
        return "WATCH"
    elif health >= 0.25:
        return "WARNING"
    else:
        return "CRITICAL"


def generate_dataset(readings_per_machine: int = 50) -> pd.DataFrame:
    """
    Generate a full dataset simulating N machines over time,
    each with multiple sensor readings per cycle sample.
    """
    records = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for machine_id, deg_rate in DEGRADATION_RATES.items():
        start_cycle = START_CYCLES[machine_id]
        # Random max life between 200-350 cycles
        max_cycle = random.randint(200, MAX_CYCLES)

        for i in range(readings_per_machine):
            cycle = start_cycle + i
            if cycle > max_cycle:
                break

            health = compute_health_score(cycle, max_cycle, deg_rate)
            rul    = compute_rul(cycle, max_cycle)
            alert  = generate_alert_level(health)
            ts     = base_time + timedelta(hours=i * 4, minutes=random.randint(0, 59))

            row = {
                "machine_id":        machine_id,
                "timestamp":         ts.isoformat(),
                "cycle":             cycle,
                "max_cycle":         max_cycle,
                "health_score":      round(health, 4),
                "rul":               rul,
                "alert_level":       alert,
                "degradation_rate":  deg_rate,
            }
            for sensor in SENSORS:
                row[sensor] = simulate_sensor_reading(sensor, health)

            records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values(["machine_id", "cycle"]).reset_index(drop=True)
    return df


def get_latest_snapshot() -> pd.DataFrame:
    """Return the most recent reading per machine (current state)."""
    df = generate_dataset(readings_per_machine=50)
    return df.groupby("machine_id").last().reset_index()


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(20).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nAlert distribution:\n{df['alert_level'].value_counts()}")
    print(f"\nMachine health snapshot:")
    snap = get_latest_snapshot()[["machine_id", "cycle", "health_score", "rul", "alert_level"]]
    print(snap.to_string(index=False))
