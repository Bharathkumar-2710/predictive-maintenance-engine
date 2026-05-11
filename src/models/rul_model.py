"""
Remaining Useful Life (RUL) Regression Model.
Uses Gradient Boosting Regressor with engineered features for prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

from src.services.data_generator import generate_dataset, get_latest_snapshot, SENSORS

# Feature engineering constants
SENSOR_COLS = SENSORS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix with rolling statistics and lag features."""
    df = df.sort_values(["machine_id", "cycle"]).copy()
    feature_frames = []

    for machine_id, group in df.groupby("machine_id"):
        g = group.copy().reset_index(drop=True)

        # Rolling window stats (window = 5 cycles)
        for col in SENSOR_COLS:
            g[f"{col}_roll_mean"] = g[col].rolling(5, min_periods=1).mean()
            g[f"{col}_roll_std"] = g[col].rolling(5, min_periods=1).std().fillna(0)
            g[f"{col}_lag1"] = g[col].shift(1).fillna(g[col])
            g[f"{col}_delta"] = g[col] - g[f"{col}_lag1"]

        # Derived composite features
        g["vibration_x_temp"] = g["vibration"] * g["temperature"]
        g["rpm_pressure_ratio"] = g["rpm"] / (g["pressure"].clip(lower=0.1))
        g["cycle_normalized"] = g["cycle"] / g["max_cycle"]

        feature_frames.append(g)

    return pd.concat(feature_frames, ignore_index=True)


def build_feature_matrix(df: pd.DataFrame):
    """Return (X, y) arrays ready for sklearn."""
    df_feat = engineer_features(df)

    drop_cols = ["machine_id", "timestamp", "alert_level", "rul",
                 "health_score", "max_cycle", "degradation_rate"]
    feature_cols = [c for c in df_feat.columns if c not in drop_cols]

    X = df_feat[feature_cols].fillna(0).values
    y = df_feat["rul"].values
    return X, y, feature_cols


class RULPredictor:
    """
    Gradient Boosting pipeline for Remaining Useful Life regression.
    """

    def __init__(self, model_type: str = "gbr"):
        if model_type == "gbr":
            estimator = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            estimator = Ridge(alpha=1.0)

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])
        self.feature_cols_: list[str] = []
        self.trained_: bool = False
        self.metrics_: dict = {}

    def fit(self, df: pd.DataFrame) -> "RULPredictor":
        X, y, self.feature_cols_ = build_feature_matrix(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.pipeline.fit(X_train, y_train)
        self.trained_ = True

        y_pred = self.pipeline.predict(X_test)
        self.metrics_ = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "r2": round(float(r2_score(y_test, y_pred)), 2),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained_:
            raise RuntimeError("Model not trained. Call fit() first.")
        return np.clip(self.pipeline.predict(X), 0, None)

    def predict_from_row(self, row: pd.Series, df_history: pd.DataFrame) -> int:
        """Predict RUL for a single machine."""
        machine_id = row["machine_id"]
        history = df_history[df_history["machine_id"] == machine_id].copy()

        if row["cycle"] not in history["cycle"].values:
            history = pd.concat([history, row.to_frame().T], ignore_index=True)

        df_feat = engineer_features(history)
        latest_feat = df_feat[df_feat["cycle"] == row["cycle"]].iloc[-1]

        feature_vals = latest_feat[self.feature_cols_].fillna(0).values.reshape(1, -1)
        pred = self.predict(feature_vals)[0]
        return int(round(pred))

    def predict_batch_snapshot(self, snapshot_df: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
        """Predict RUL for all machines in snapshot."""
        results = []
        for _, row in snapshot_df.iterrows():
            predicted_rul = self.predict_from_row(row, full_df)
            hours_to_failure = predicted_rul * 4
            results.append({
                "machine_id": row["machine_id"],
                "current_cycle": int(row["cycle"]),
                "actual_rul": int(row["rul"]),
                "predicted_rul": predicted_rul,
                "health_score": round(float(row["health_score"]), 4),
                "alert_level": row["alert_level"],
                "hours_to_failure": hours_to_failure,
                "days_to_failure": round(hours_to_failure / 24, 1),
            })

        return pd.DataFrame(results).sort_values("predicted_rul")

    def evaluate(self) -> dict:
        if not self.trained_:
            raise RuntimeError("Model not trained.")
        return self.metrics_

    def feature_importances(self) -> pd.DataFrame:
        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "feature_importances_"):
            return pd.DataFrame()
        imp = model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols_, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(15)
        )


# Singleton model instance
_model_instance: RULPredictor | None = None
_training_df: pd.DataFrame | None = None


def get_trained_model() -> tuple[RULPredictor, pd.DataFrame]:
    global _model_instance, _training_df
    if _model_instance is None:
        print("Training RUL model on simulated dataset...")
        _training_df = generate_dataset(readings_per_machine=50)
        _model_instance = RULPredictor(model_type="gbr").fit(_training_df)
        m = _model_instance.metrics_
        print(f"  MAE = {m['mae']} cycles | R² = {m['r2']} | "
              f"Train/Test = {m['n_train']}/{m['n_test']}")
    return _model_instance, _training_df


if __name__ == "__main__":
    model, df = get_trained_model()
    print("\nMetrics:", model.evaluate())
    snap = df.groupby("machine_id").last().reset_index()
    preds = model.predict_batch_snapshot(snap, df)
    print("\nRUL Predictions:")
    print(preds.to_string(index=False))