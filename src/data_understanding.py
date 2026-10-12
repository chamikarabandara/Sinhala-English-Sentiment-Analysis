
import pandas as pd

from src.config import DATA_PATH


def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    main()
