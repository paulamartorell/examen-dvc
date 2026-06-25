from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def main():
    data_dir = Path("data/processed_data")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scaled training data...")
    X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()

    model = RandomForestRegressor(random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
    }

    print("Running GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best CV R2 score:", grid_search.best_score_)

    joblib.dump(grid_search.best_params_, models_dir / "best_params.pkl")

    print("Best parameters saved successfully.")


if __name__ == "__main__":
    main()
