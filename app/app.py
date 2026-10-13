
import streamlit as st
from pathlib import Path
import joblib

st.set_page_config(
    page_title="Singlish Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_sentiment_model.joblib"

st.title("💬 Singlish Sentiment Analysis")
st.write(
    "Enter a Sinhala-English mixed comment to predict "
    "whether its sentiment is Positive, Negative, or Neutral."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

comment = st.text_area(
    "Enter your comment",
    placeholder="Example: me phone eka godak hodai"
)

if st.button("Predict Sentiment"):
    if not comment.strip():
        st.warning("Please enter a comment first.")
    elif model is None:
        st.error(
            "The trained model has not been saved yet. "
            "Please train and save the model first."
        )
    else:
        prediction = model.predict([comment])[0]

        sentiment_names = {
            "Pos": "Positive 😊",
            "Neg": "Negative 😞",
            "Neu": "Neutral 😐"
        }

        st.subheader(
            sentiment_names.get(prediction, str(prediction))
        )
