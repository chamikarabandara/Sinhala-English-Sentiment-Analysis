# Sinhala-English Code-Mixed Sentiment Analysis

## Project Overview

This project develops a machine learning system to classify Sinhala-English code-mixed text (Singlish) into three sentiment categories: Positive, Negative, and Neutral.

The system uses Natural Language Processing (NLP), TF-IDF feature extraction, and machine learning algorithms to identify sentiment in informal Sinhala-English comments. A Streamlit web application allows users to enter a comment and receive a predicted sentiment.

## Objectives

- Explore and understand the Singlish sentiment dataset.
- Clean and preprocess text data.
- Perform Exploratory Data Analysis (EDA).
- Train and compare different machine learning models.
- Evaluate models using appropriate classification metrics.
- Develop a web application for sentiment prediction.

## Dataset

The dataset contains Sinhala-English code-mixed comments with sentiment labels.

- **Text column:** `Comments`
- **Target column:** `Sentiment`
- **Sentiment classes:** `Pos`, `Neg`, and `Neu`
- **Cleaned dataset size:** 7,831 comments

The dataset is divided into training and testing sets using a stratified 80:20 split, with `random_state=42`.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Linear Support Vector Machine (Linear SVM)
- Matplotlib
- Jupyter Notebook
- Streamlit
- Joblib
- Git and GitHub

## Project Structure

```text
Singlish_Sentiment_Analysis/
├── app/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── Singlish Dataset.csv
│   └── processed/
│       └── singlish_cleaned.csv
├── models/
│   └── best_sentiment_model.joblib
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_text_preprocessing.ipynb
│   ├── 04_model_experiments.ipynb
│   └── 05_model_evaluation.ipynb
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── config.py
│   ├── data_understanding.py
│   ├── eda.py
│   └── preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Model Development

Several machine learning models were evaluated, including:

- Linear SVM with word-level TF-IDF features
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM with word and character TF-IDF features
- Linear SVM with character-only TF-IDF features

The current best-performing model is **Character-only TF-IDF with Linear SVM**.

Character-level TF-IDF features can help represent spelling variations and informal word patterns commonly found in Sinhala-English code-mixed text.

## Model Evaluation

The best model achieved the following results on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 75.05% |
| Macro Precision | 70.37% |
| Macro Recall | 68.52% |
| Macro F1-score | 69.27% |

### Classification Report

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 71.71% | 67.65% | 69.63% |
| Neutral | 57.48% | 49.27% | 53.06% |
| Positive | 81.91% | 88.63% | 85.14% |

The model performs best on Positive comments. Neutral sentiment remains the most challenging class and is an area for future improvement.

## Web Application

A Streamlit application provides an interface where users can enter a Sinhala-English mixed comment and receive a sentiment prediction.

### Run the Application Locally

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a Python virtual environment.
4. Install the required dependencies.
5. Run the Streamlit application.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application from the project root:

```bash
streamlit run app/app.py
```

The application will open in a browser.

**Note:** The trained model file must be available at `models/best_sentiment_model.joblib`.

## Limitations

- Sinhala-English code-mixed text contains informal spellings and variations.
- Neutral comments are more difficult to classify correctly.
- The model may misclassify new comments that differ from the training data.
- Evaluation results represent the selected test split and may differ on unseen real-world data.

## Future Improvements

- Expand the dataset with diverse and representative Singlish comments.
- Review ambiguous or potentially incorrect sentiment labels.
- Improve Neutral sentiment classification.
- Experiment with additional feature extraction methods and machine learning models.
- Evaluate generalization using additional validation strategies.
- Improve the Streamlit interface and prediction feedback.

## Author

Developed as a machine learning capstone project on Sinhala-English code-mixed sentiment analysis.
