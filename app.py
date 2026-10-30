import streamlit as st
import joblib
import re
from pathlib import Path


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="YouTube Spam Detector",
    page_icon="🚨",
    layout="centered"
)


# ==========================================
# FIND PROJECT ROOT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# HEADER
# ==========================================

st.title("🚨 YouTube Comment Spam Detector")

st.write(
    "Enter a YouTube comment below to check "
    "whether it is spam or not."
)

st.divider()


# ==========================================
# COMMENT INPUT
# ==========================================

comment = st.text_area(
    "💬 Enter YouTube Comment",
    placeholder="Example: Congratulations! You won a free iPhone. Click here!",
    height=150
)


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "🔍 Detect Spam",
    use_container_width=True
):

    if not comment.strip():

        st.warning(
            "⚠️ Please enter a comment first."
        )

    else:

        # Clean comment
        cleaned_comment = clean_text(comment)

        # Convert text into TF-IDF
        comment_vector = vectorizer.transform(
            [cleaned_comment]
        )

        # Prediction
        prediction = model.predict(
            comment_vector
        )[0]

        # Probability
        probabilities = model.predict_proba(
            comment_vector
        )[0]

        confidence = max(probabilities) * 100


        # ==================================
        # RESULT
        # ==================================

        st.divider()

        if prediction == 1:

            st.error(
                "🚨 SPAM COMMENT"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "The model classified this comment "
                "as **Spam**."
            )

        else:

            st.success(
                "✅ NOT SPAM"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.write(
                "The model classified this comment "
                "as **Not Spam**."
            )


        # ==================================
        # ANALYSIS
        # ==================================

        with st.expander("🔎 View Analysis"):

            st.write(
                "**Original Comment:**"
            )

            st.write(comment)

            st.write(
                "**Cleaned Comment:**"
            )

            st.write(cleaned_comment)

            st.write(
                "**Prediction:**"
            )

            if prediction == 1:
                st.write("🚨 Spam")
            else:
                st.write("✅ Not Spam")

            st.write(
                f"**Confidence:** {confidence:.2f}%"
            )