from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")

    print("Loading training data and best parameters...")
    X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    best_params = joblib.load(models_dir / "best_params.pkl")

    print("Training final model...")
    model = RandomForestRegressor(
        random_state=42,
        **best_params
    )
    model.fit(X_train, y_train)

    joblib.dump(model, models_dir / "trained_model.pkl")

    print("Trained model saved successfully.")


if __name__ == "__main__":
    main()
