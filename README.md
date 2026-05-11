# Industrial IoT Predictive Maintenance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

**Lead Python Developer & Data Analyst**

*A high-performance end-to-end monitoring system for predicting mechanical failures in industrial machinery before they occur.*

</div>

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Project Structure](#project-structure)
5. [Tech Stack](#tech-stack)
6. [Quick Start](#quick-start)
7. [Web Interface](#web-interface)
8. [API Endpoints](#api-endpoints)
9. [Sensor Specifications](#sensor-specifications)
10. [Model Performance](#model-performance)
11. [Docker Deployment](#docker-deployment)
12. [License](#license)

---

## Project Overview

This Industrial IoT Predictive Maintenance System is designed to predict mechanical failures in industrial machinery before they occur. The system simulates real-time sensor data (temperature, vibration, pressure, RPM, oil viscosity, and current draw), processes these signals through a physics-informed machine learning model, and provides actionable "Remaining Useful Life" (RUL) estimates via a modern web dashboard.

### Key Capabilities

- **Real-time Health Monitoring**: Continuous tracking of 10 industrial machines
- **RUL Prediction**: Machine learning model predicting remaining useful life in cycles
- **Automatic Alerting**: Threshold-based alerts for NORMAL, WATCH, WARNING, and CRITICAL states
- **Interactive Dashboard**: Modern web UI for fleet overview and individual machine details
- **Sensor Data Input**: Manual input form for predicting RUL with custom sensor values

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    Predictive Maintenance System                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────┐ │
│  │  Sensor   │───▶│  Database  │───▶│    ML    │───▶│  REST  │ │
│  │  Data     │    │  (SQLite)  │    │  Model   │    │  API   │ │
│  │ Generator │    │           │    │  (GBR)   │    │ Flask  │ │
│  └────────────┘    └────────────┘    └────────────┘    └────────┘ │
│                                              │                     │
│                                              ▼                     │
│                                    ┌────────────────────┐         │
│                                    │    Web Dashboard   │         │
│                                    │     (HTML/CSS)     │         │
│                                    └────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Sensor Simulation** | Generates realistic industrial sensor data with degradation patterns |
| **RUL Prediction** | Gradient Boosting Regressor predicts remaining useful life |
| **6 Sensor Types** | Temperature, Vibration, Pressure, RPM, Oil Viscosity, Current Draw |
| **Alert System** | Automatic alerts when sensors exceed warning/critical thresholds |
| **Multi-page UI** | Dashboard, Machines, Alerts, Predictions, Input pages |
| **Search & Filter** | Filter machines by status, search by machine ID |
| **Health Visualization** | Visual health bars showing percentage degradation |
| **Feature Engineering** | Rolling statistics, lag features, composite features |
| **REST API** | Full API for integration with other systems |

---

## Project Structure

```
predictive-maintenance/
├── backend/
│   ├── app.py              # Flask REST API server
│   ├── database.py         # SQLite database manager
│   └── sensors.py         # OOP sensor class hierarchy
├── models/
│   └── rul_model.py       # Gradient Boosting RUL regressor
├── data/
│   └── data_generator.py # NASA CMAPSS-style data simulator
├── static/
│   ├── index.html        # Home/redirect page
│   ├── dashboard.html   # Main fleet dashboard
│   ├── machines.html   # Machine listing with search/filter
│   ├── alerts.html     # Alert history and summary
│   ├── predictions.html # Model metrics and predictions table
│   └── input.html     # Manual sensor input form
├── tests/              # Test files
├── Dockerfile          # Docker container definition
├── docker-compose.yml  # Docker Compose configuration
├── requirements.txt  # Python dependencies
├── run.bat           # Quick start script (Windows)
└── README.md        # This file
```

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core programming language |
| **Flask** | REST API web framework |
| **SQLite** | Embedded database (development) |
| **PostgreSQL** | Production database (optional) |
| **NumPy** | Numerical computing |
| **Pandas** | Data manipulation |
| **Scikit-learn** | Machine learning (Gradient Boosting) |
| **HTML/CSS/JS** | Modern responsive web interface |
| **Docker** | Containerized deployment |
| **Power BI** | Optional data visualization |

---

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

```bash
# Clone the repository
cd predictive-maintenance-engine

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```powershell
# Windows
run.bat

# Or manually
python backend/app.py
```

The server will start on **http://localhost:5050**

---

## Web Interface

### Available Pages

| URL | Page | Description |
|-----|------|-------------|
| `/` | Home | Redirects to dashboard |
| `/dashboard.html` | **Dashboard** | Fleet overview with stats, alerts, and health bars |
| `/machines.html` | **Machines** | All machines with search & filter functionality |
| `/alerts.html` | **Alerts** | Alert summary and history |
| `/predictions.html` | **Predictions** | Model metrics, feature importance, predictions table |
| `/input.html` | **Input** | Manual sensor data input for RUL prediction |

### Dashboard Preview

The main dashboard displays:
- Total machines and readings count
- At-risk machine count
- Model performance metrics (MAE, R²)
- Alert distribution (CRITICAL, WARNING, WATCH, NORMAL)
- At-risk machines section
- Fleet status cards with health bars

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/dashboard` | GET | Full dashboard data bundle |
| `/api/machines` | GET | List all machines |
| `/api/machines/<id>` | GET | Single machine details |
| `/api/predictions` | GET | RUL predictions for all machines |
| `/api/alerts` | GET | Recent alerts |
| `/api/ingest` | POST | Submit sensor readings |
| `/api/model-metrics` | GET | Model performance metrics |
| `/api/feature-importance` | GET | Top feature importances |

### Example API Response

```json
{
  "fleet": [
    {
      "machine_id": "MACH-001",
      "current_cycle": 104,
      "predicted_rul": 164,
      "health_score": 0.8788,
      "alert_level": "NORMAL",
      "days_to_failure": 27.3
    }
  ],
  "model_metrics": {
    "mae": 3.6,
    "r2": 0.99
  }
}
```

---

## Sensor Specifications

### Supported Sensors

| Sensor | Unit | Warning Threshold | Critical Threshold | Direction |
|--------|------|-------------------|-------------------|-----------|
| Temperature | °F | > 490 | > 530 | High = Bad |
| Vibration | g | > 0.45 | > 0.60 | High = Bad |
| Pressure | PSI | < 10 | < 7.5 | Low = Bad |
| RPM | rpm | < 1650 | < 1500 | Low = Bad |
| Oil Viscosity | cSt | < 36 | < 30 | Low = Bad |
| Current Draw | A | > 15.5 | > 17 | High = Bad |

### Alert Levels

| Level | Health Score | Color | Action Required |
|-------|--------------|-------|---------------|
| **NORMAL** | ≥ 75% | Green | Routine monitoring |
| **WATCH** | 50-74% | Yellow | Schedule maintenance |
| **WARNING** | 25-49% | Orange | Plan for maintenance |
| **CRITICAL** | < 25% | Red | Immediate action required |

---

## Model Performance

The RUL prediction model is built using **Gradient Boosting Regressor** with comprehensive feature engineering.

### Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **MAE** | ~3.6 cycles | Mean Absolute Error |
| **R²** | 0.99 | Coefficient of Determination |
| **Training Samples** | ~400 | Training data size |
| **Test Samples** | ~100 | Test data size |

### Feature Engineering

The model uses these engineered features:
- Rolling mean (window=5)
- Rolling standard deviation
- Lag-1 values and deltas
- Composite features (e.g., vibration × temperature)
- Cycle normalization

### Top Features

1. `current_draw_roll_mean`
2. `oil_viscosity_roll_mean`
3. `cycle_normalized`
4. `temperature_roll_mean`
5. `rpm_roll_mean`

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t predictive-maintenance .

# Run the container
docker run -p 5050:5050 predictive-maintenance
```

### Docker Compose

```bash
# Start all services
docker-compose up --build

# Stop services
docker-compose down
```

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Lead Python Developer & Data Analyst**

*Building intelligent systems for predictive maintenance and industrial automation.*