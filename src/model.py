from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_FILE = Path(
    "data/mumbai_real_heat_data.csv"
)
MODEL_FILE = Path("data/mumbai_heat_risk_model.pkl")
PREDICTIONS_FILE = Path("data/mumbai_model_predictions.csv")


FEATURE_COLUMNS = [
    "vegetation_pct",
    "building_density_pct",
    "impervious_surface_pct",
    "population_density",
    "humidity_pct",
    "average_temperature_c",
]

TARGET_COLUMN = "heat_risk_score"


def train_heat_risk_model():
    """Train and evaluate the neighborhood heat-risk model."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Run data_generator.py first."
        )

    data = pd.read_csv(DATA_FILE)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)

    test_results = data.loc[X_test.index].copy()
    test_results["predicted_heat_risk"] = predictions
    test_results.to_csv(PREDICTIONS_FILE, index=False)

    with open(MODEL_FILE, "wb") as model_file:
        pickle.dump(model, model_file)

    print("Model training complete.")
    print(f"Training neighborhoods: {len(X_train)}")
    print(f"Testing neighborhoods: {len(X_test)}")
    print(f"R² score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")

    if r2 >= 0.75:
        print("Target achieved: R² is at least 0.75.")
    else:
        print("R² target not reached yet. We will tune the model next.")

    return model, r2, rmse, mae


if __name__ == "__main__":
    train_heat_risk_model()