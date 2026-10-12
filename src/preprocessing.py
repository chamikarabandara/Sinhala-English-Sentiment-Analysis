
import pandas as pd
import re

from src.config import DATA_PATH, PROCESSED_DATA_DIR


def normalize_text(text):
    """Normalize text without removing Singlish words."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # Remove rows with missing text or sentiment
    df = df.dropna(subset=["Comments", "Sentiment"]).copy()

    # Normalize text
    df["Comments"] = df["Comments"].apply(normalize_text)

    # Remove empty comments
    df = df[df["Comments"] != ""]

    # Remove duplicate text-label pairs
    df = df.drop_duplicates(subset=["Comments", "Sentiment"])

    # Save cleaned dataset
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "singlish_cleaned.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Cleaned dataset shape:", df.shape)
    print("Saved cleaned dataset to:", output_path)
    print("\nSentiment distribution after cleaning:")
    print(df["Sentiment"].value_counts())


if __name__ == "__main__":
    main()
