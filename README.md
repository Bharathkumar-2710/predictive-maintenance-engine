# Industrial IoT Predictive Maintenance System

**Role:** Lead Python Developer & Data Analyst  
**Tech Stack:** Python (FastAPI), PostgreSQL, NumPy, Scikit-learn, Docker, Power BI

---

## Project Overview

Developed a high-performance end-to-end monitoring system designed to predict mechanical failures in industrial machinery before they occur. The system simulates real-time sensor data (vibration, temperature, and RPM), processes these signals through a physics-informed machine learning model, and provides actionable "Remaining Useful Life" (RUL) estimates via a web dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Predictive Maintenance System            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Data        │    │  ML Model    │    │  REST API    │  │
│  │  Generator   │───▶│  (GBR)       │───▶│  (FastAPI)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  PostgreSQL  │    │  Metrics     │    │  Dashboard   │  │
│  │  Database    │    │  Evaluation  │    │  (JSON)      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

- **Real-time Sensor Simulation**: Temperature, vibration, pressure, RPM, oil viscosity, current draw
- **RUL Prediction**: Gradient Boosting Regressor with feature engineering
- **Alert System**: Automatic alerts when sensor thresholds exceeded
- **REST API**: FastAPI-based endpoints for predictions, alerts, machine status
- **Docker Support**: Containerized deployment
- **PostgreSQL Support**: Production-ready database (SQLite for development)

---

## Project Structure

```
predictive-maintenance/
├── src/
│   ├── main.py                 # FastAPI application entry point
│   ├── api/
│   │   ├── routes.py           # API route handlers
│   │   └── deps.py             # Dependency injection
│   ├── core/
│   │   ├── config.py           # Configuration settings
│   │   └── database.py         # Database manager
│   ├── models/
│   │   └── rul_model.py        # ML RUL prediction model
│   ├── services/
│   │   ├── sensor_service.py  # Sensor management
│   │   └── data_generator.py  # Simulated sensor data
│   └── schemas/
│       └── schemas.py          # Pydantic data models
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API server
python -m src.main

# Access API docs: http://localhost:8000/docs
```

### Docker

```bash
# Build and run
docker-compose up --build

# Access API docs: http://localhost:8000/docs
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/health` | GET | Health check |
| `/api/dashboard` | GET | Full dashboard data |
| `/api/machines` | GET | List all machines |
| `/api/machines/{id}` | GET | Machine details |
| `/api/predictions` | GET | RUL predictions |
| `/api/alerts` | GET | Recent alerts |
| `/api/ingest` | POST | Ingest sensor data |
| `/api/model-metrics` | GET | Model performance metrics |
| `/api/feature-importance` | GET | Feature importance scores |

---

## Alert Thresholds

| Sensor | Warning | Critical | Direction |
|--------|---------|----------|-----------|
| Temperature | > 490 °F | > 530 °F | High = bad |
| Vibration | > 0.45 g | > 0.60 g | High = bad |
| Pressure | < 10 PSI | < 7.5 PSI | Low = bad |
| RPM | < 1650 rpm | < 1500 rpm | Low = bad |
| Oil viscosity | < 36 cSt | < 30 cSt | Low = bad |
| Current draw | > 15.5 A | > 17 A | High = bad |

---

## Model Performance

- **MAE**: ~3.6 cycles
- **R²**: 0.99
- **Algorithm**: Gradient Boosting Regressor with StandardScaler pipeline
- **Features**: Rolling mean/std, lag features, composite features

---

## License

MIT