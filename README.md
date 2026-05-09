# Predictive Maintenance System

NASA Turbofan-inspired industrial IoT simulation with ML-powered Remaining Useful Life (RUL) prediction.

---

## Architecture

```
predictive_maintenance/
├── data/
│   └── data_generator.py      # NASA CMAPSS-style sensor data simulation
├── backend/
│   ├── sensors.py             # OOP sensor class hierarchy (6 sensor types)
│   ├── database.py            # SQLite/MySQL database manager (Repository pattern)
│   └── app.py                 # Flask REST API
├── models/
│   └── rul_model.py           # Gradient Boosting RUL regression pipeline
├── requirements.txt
└── README.md
```

---

## Components

### 1. Data Generation (`data/data_generator.py`)
- Simulates 10 machines with realistic sensor degradation curves
- Sensors: temperature, vibration, pressure, RPM, oil viscosity, current draw
- Nonlinear health degradation: `health = 1 - (progress^2.2) * (1 + rate*10)`
- Each machine has randomized lifecycle (200–350 cycles) and degradation rate
- Outputs labeled dataset with health score, RUL, and alert level per reading

### 2. OOP Sensor Hierarchy (`backend/sensors.py`)
```
Sensor (ABC)
├── TemperatureSensor   — thermocouple, °F, warns at 490, critical at 530
├── VibrationSensor     — accelerometer, g-force, warns at 0.45g
├── PressureSensor      — inverted logic (low = bad), warns below 10 PSI
├── RPMSensor           — inverted logic, warns below 1650 RPM
├── OilViscositySensor  — inverted logic, warns below 36 cSt
└── CurrentDrawSensor   — clamp ammeter, warns above 15.5 A
```
- `AlertSeverity` enum: NORMAL / WATCH / WARNING / CRITICAL
- `SensorSuite` — manages full 6-sensor set per machine, ingests readings, returns alerts
- Factory function `create_sensor(type, machine_id)` for dynamic instantiation

### 3. RUL Regression Model (`models/rul_model.py`)
- **Algorithm**: Gradient Boosting Regressor (sklearn)
- **Feature engineering**: Rolling mean/std (window=5), lag-1 deltas, composite cross-features
- **Pipeline**: `StandardScaler → GBR(n_estimators=200, max_depth=4, lr=0.05)`
- **Results**: MAE = 3.6 cycles, R² = 0.995
- Top features: `current_draw_roll_mean`, `oil_viscosity_roll_mean`, `cycle_normalized`

### 4. Database (`backend/database.py`)
**Tables** (MySQL-compatible DDL):
- `machines` — machine registry (machine_id, max_cycle, degradation_rate)
- `sensor_logs` — timestamped raw sensor readings (indexed on machine_id + cycle)
- `health_snapshots` — periodic health scores + RUL predictions
- `alerts` — triggered alert history with severity, sensor, value, threshold

**Pattern**: Repository-style `DatabaseManager` class with context-managed connections.

### 5. Flask REST API (`backend/app.py`)
| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/dashboard` | GET | Full dashboard data bundle |
| `/api/machines` | GET | List all machines |
| `/api/machines/<id>` | GET | Machine detail + sensor suite + trend |
| `/api/ingest` | POST | Receive live sensor readings |
| `/api/predictions` | GET | RUL predictions for all machines |
| `/api/alerts` | GET | Recent alert feed |
| `/api/feature-importance` | GET | Model feature importances |
| `/api/model-metrics` | GET | MAE, R², train/test counts |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test data generator
python data/data_generator.py

# Test RUL model
python models/rul_model.py

# Run Flask API
cd backend
python app.py
# → http://localhost:5050/api/dashboard
```

## Switching to MySQL

In `backend/database.py`, replace SQLite connection with:
```python
import mysql.connector

def _conn(self):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="...", database="predictive_maintenance"
    )
    yield conn
    conn.commit()
    conn.close()
```

Replace `?` placeholders with `%s` in all SQL statements.

---

## Alert Logic Summary

| Sensor | Warning | Critical | Direction |
|---|---|---|---|
| Temperature | > 490 °F | > 530 °F | High = bad |
| Vibration | > 0.45 g | > 0.60 g | High = bad |
| Pressure | < 10 PSI | < 7.5 PSI | Low = bad |
| RPM | < 1650 rpm | < 1500 rpm | Low = bad |
| Oil viscosity | < 36 cSt | < 30 cSt | Low = bad |
| Current draw | > 15.5 A | > 17 A | High = bad |
