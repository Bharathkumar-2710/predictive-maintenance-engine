"""
Configuration settings for the Predictive Maintenance System.
"""

import os
from typing import Optional


class Settings:
    """Application settings."""

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./predictive_maintenance.db"
    )

    # API
    API_PREFIX: str = "/api"
    PROJECT_NAME: str = "Industrial IoT Predictive Maintenance System"

    # Model
    MODEL_TYPE: str = "gbr"  # "gbr" or "ridge"
    FEATURE_WINDOW: int = 5

    # Simulation
    NUM_MACHINES: int = 10
    MAX_CYCLES: int = 300
    READINGS_PER_MACHINE: int = 50


settings = Settings()