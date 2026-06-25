# The exam statement indicates that all columns are float64.
# The downloaded dataset contains an additional string column ('date'),
# which is excluded from modeling.

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    input_path = Path("data/raw_data/raw.csv")
    output_dir = Path("data/processed_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(input_path)

    print(f"Dataset shape: {df.shape}")
    
    # Remove non-numerical date column
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    X = df.drop(columns=["silica_concentrate"])
    y = df["silica_concentrate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)

    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Train-test split saved successfully.")


if __name__ == "__main__":
    main()
