from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Loading test data and trained model...")
    X_test = pd.read_csv(data_dir / "X_test_scaled.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
    model = joblib.load(models_dir / "trained_model.pkl")

    print("Making predictions...")
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })
    predictions.to_csv(data_dir / "predictions.csv", index=False)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    scores = {
        "mse": mse,
        "mae": mae,
        "r2": r2
    }

    with open(metrics_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Scores:", scores)
    print("Predictions and scores saved successfully.")


if __name__ == "__main__":
    main()
