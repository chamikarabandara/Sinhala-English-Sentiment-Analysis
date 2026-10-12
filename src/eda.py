
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATA_PATH, FIGURES_DIR


def main():
    df = pd.read_csv(DATA_PATH)

    # Count each sentiment class
    sentiment_counts = df["Sentiment"].value_counts()

    print("Sentiment Distribution")
    print("-" * 30)
    print(sentiment_counts)

    print("\nSentiment Percentages")
    print("-" * 30)
    print((df["Sentiment"].value_counts(normalize=True) * 100).round(2))

    # Create output directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Plot sentiment distribution
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        data=df,
        x="Sentiment",
        order=["Pos", "Neg", "Neu"]
    )

    ax.set_title("Sentiment Class Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Comments")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "sentiment_distribution.png",
        dpi=300
    )
    plt.show()


    # Analyze comment lengths
    df["Comment_Length"] = df["Comments"].astype(str).str.split().str.len()

    print("\nComment Length Statistics")
    print("-" * 30)
    print(df["Comment_Length"].describe().round(2))

    # Average comment length by sentiment
    print("\nAverage Comment Length by Sentiment")
    print("-" * 30)
    print(
        df.groupby("Sentiment")["Comment_Length"]
        .mean()
        .round(2)
    )

    # Save comment length chart
    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="Sentiment",
        y="Comment_Length",
        order=["Pos", "Neg", "Neu"]
    )

    plt.title("Comment Length by Sentiment")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Words")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "comment_length_by_sentiment.png",
        dpi=300
    )
    plt.show()


if __name__ == "__main__":
    main()
