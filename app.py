import streamlit as st
import pickle
import re
import string


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main application */
    .stApp {
        background: linear-gradient(135deg, #f5f9ff 0%, #eef4ff 50%, #ffffff 100%);
    }

    /* Remove top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 25px 20px 10px 20px;
    }

    .main-header h1 {
        color: #123B6D;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .main-header p {
        color: #5D6B7A;
        font-size: 17px;
        margin-top: 5px;
    }

    /* Subtitle badge */
    .badge {
        display: inline-block;
        background: #DCEBFF;
        color: #1769AA;
        padding: 7px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* Input card */
    .input-card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #DCE6F2;
        box-shadow: 0 5px 20px rgba(30, 80, 130, 0.08);
        margin-top: 20px;
    }

    .section-title {
        color: #123B6D;
        font-size: 22px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .section-description {
        color: #6B7785;
        font-size: 14px;
        margin-bottom: 18px;
    }

    /* Text area */
    textarea {
        border-radius: 12px !important;
        border: 2px solid #D6E3F0 !important;
        font-size: 16px !important;
    }

    textarea:focus {
        border-color: #4A90E2 !important;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1) !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #1769AA, #3182CE);
        color: white;
        font-size: 17px;
        font-weight: 700;
        padding: 13px 20px;
        transition: 0.3s;
        margin-top: 10px;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(23, 105, 170, 0.25);
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 35px 25px;
        text-align: center;
        border: 1px solid #DCE6F2;
        box-shadow: 0 7px 25px rgba(30, 80, 130, 0.10);
        margin-top: 20px;
        min-height: 300px;
    }

    .result-label {
        color: #718096;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .emotion-emoji {
        font-size: 65px;
        margin: 15px 0 5px 0;
    }

    .emotion-name {
        color: #123B6D;
        font-size: 34px;
        font-weight: 800;
        margin: 5px 0;
    }

    .result-message {
        color: #66788A;
        font-size: 15px;
        margin-top: 10px;
    }

    /* Confidence */
    .confidence-box {
        background: #F1F7FF;
        border-radius: 12px;
        padding: 12px;
        margin-top: 18px;
    }

    .confidence-title {
        color: #54708C;
        font-size: 13px;
        font-weight: 600;
    }

    .confidence-value {
        color: #1769AA;
        font-size: 23px;
        font-weight: 800;
    }

    # /* Info cards */
    # .info-card {
    #     background: white;
    #     padding: 20px;
    #     border-radius: 15px;
    #     border: 1px solid #DCE6F2;
    #     box-shadow: 0 4px 15px rgba(30, 80, 130, 0.06);
    #     text-align: center;
    #     height: 120px;
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123B6D 0%, #0B2A4A 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .sidebar-brand {
        text-align: center;
        padding: 15px 5px 25px 5px;
    }

    .sidebar-brand .icon {
        font-size: 50px;
    }

    .sidebar-brand h2 {
        margin: 5px 0;
        font-size: 24px;
    }

    .sidebar-brand p {
        font-size: 13px;
        opacity: 0.75;
    }

    .sidebar-box {
        background: rgba(255,255,255,0.10);
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
    }

    .sidebar-box-title {
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .sidebar-box-text {
        font-size: 13px;
        line-height: 1.6;
        opacity: 0.85;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        padding: 35px 10px 10px 10px;
    }

    /* Hide Streamlit menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():

    try:
        with open("emotion_model.pkl", "rb") as file:
            model = pickle.load(file)

        with open("bow_vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        with open("emotion_mapping.pkl", "rb") as file:
            emotion_mapping = pickle.load(file)

        return model, vectorizer, emotion_mapping

    except FileNotFoundError:
        return None, None, None


model, vectorizer, emotion_mapping = load_model()


# ============================================================
# EMOTION INFORMATION
# ============================================================

emotion_info = {

    "sadness": {
        "emoji": "😢",
        "message": "The text appears to express sadness or disappointment."
    },

    "anger": {
        "emoji": "😡",
        "message": "The text appears to express anger, frustration, or irritation."
    },

    "love": {
        "emoji": "❤️",
        "message": "The text appears to express love, affection, or strong positive feelings."
    },

    "surprise": {
        "emoji": "😮",
        "message": "The text appears to express surprise, shock, or unexpected feelings."
    },

    "fear": {
        "emoji": "😨",
        "message": "The text appears to express fear, worry, or anxiety."
    },

    "joy": {
        "emoji": "😊",
        "message": "The text appears to express happiness, excitement, or joy."
    }
}


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">
        <div class="icon">💭</div>
        <h2>EmotionAI</h2>
        <p>NLP Emotion Classification</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-box">
        <div class="sidebar-box-title">Model</div>
        <div class="sidebar-box-text">
            Logistic Regression
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <div class="sidebar-box-title">NLP Technique</div>
        <div class="sidebar-box-text">
            Bag of Words<br>
            CountVectorizer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <div class="sidebar-box-title">Model Accuracy</div>
        <div class="sidebar-box-text">
            88.97%
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-box">
        <div class="sidebar-box-title">Emotions</div>
        <div class="sidebar-box-text">
            😢 Sadness<br>
            😡 Anger<br>
            ❤️ Love<br>
            😮 Surprise<br>
            😨 Fear<br>
            😊 Joy
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>AI Emotion Detector</h1>
    <p>
        Understand the emotion behind your text using
        Natural Language Processing and Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL ERROR CHECK
# ============================================================

if model is None:
    st.error(
        "⚠️ Model files were not found. "
        "Please make sure emotion_model.pkl, "
        "bow_vectorizer.pkl and emotion_mapping.pkl "
        "are in the same folder as app.py."
    )
    st.stop()


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1.15, 0.85],
    gap="large"
)


# ============================================================
# LEFT COLUMN — INPUT
# ============================================================

with left_column:
    st.markdown("""
    <div class="input-card">
        <div class="section-title">
            ✍️ Enter Your Text
        </div>
        <div class="section-description">
            Write a sentence and let the AI identify the emotion.
        </div>
    </div>
    """, unsafe_allow_html=True)

    user_text = st.text_area(
        "Text",
        placeholder="Example: I am so happy today because I got my dream job!",
        height=180,
        label_visibility="collapsed"
    )

    # Character counter
    if user_text:
        st.caption(f"Characters: {len(user_text)}")
    predict_button = st.button(
        "Detect Emotion",
        use_container_width=True
    )


# ============================================================
# RIGHT COLUMN — RESULT
# ============================================================

with right_column:
    if predict_button:
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            cleaned_text = preprocess_text(user_text)
            text_vector = vectorizer.transform(
                [cleaned_text]
            )
            prediction = model.predict(text_vector)[0]
            emotion = emotion_mapping[prediction]
            confidence = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    text_vector
                )[0]
                confidence = max(probabilities) * 100

            # --------------------------------------------
            # EMOTION INFORMATION
            # --------------------------------------------

            info = emotion_info.get(
                emotion,
                {
                    "emoji": "💭",
                    "message": "Emotion detected from the text."
                }
            )

            emoji = info["emoji"]
            message = info["message"]

            # --------------------------------------------
            # RESULT CARD
            # --------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Detected Emotion
                    </div>
                    <div class="emotion-emoji">
                        {emoji}
                    </div>
                    <div class="emotion-name">
                        {emotion.upper()}
                    </div>
                    <div class="result-message">
                        {message}
                    </div>

                """,
                unsafe_allow_html=True
            )
            if confidence is not None:
                st.markdown(
                    f"""
                    <div class="confidence-box">
                        <div class="confidence-title">
                            MODEL CONFIDENCE
                        </div>
                        <div class="confidence-value">
                            {confidence:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-label">
                Prediction Result
            </div>
            <div class="emotion-emoji">
                💭
            </div>
            <div class="emotion-name">
                Ready
            </div>
            <div class="result-message">
                Enter your text and click
                <b>Detect Emotion</b> to get a prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)




# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    <b>💭 EmotionAI</b> • NLP Emotion Classification Project
    <br>
    Built with Python, Scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)