import streamlit as st
import joblib
import re
import base64
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie

# 🎨 Gradient background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f8f8ff, #e0c3fc, #8ec5fc);
    }
    </style>
""", unsafe_allow_html=True)

# 🔤 Bricolage Grotesque font + custom styles
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;600&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Bricolage Grotesque', sans-serif !important;
        }
        textarea {
            background-color: #F8F8FF !important;
            color: #000000 !important;
            font-size: 18px !important;
            border: 2px solid #5A189A !important;
            border-radius: 12px !important;
            padding: 10px !important;
            box-shadow: 2px 2px 8px rgba(90, 24, 154, 0.2) !important;
        }
        .review-label {
            font-size: 26px !important;
            font-weight: 600 !important;
            color: #3a0ca3;
            margin-top: 4px !important;
}

        div.stButton > button:first-child {
            background-color: #5A189A;
            color: white;
            font-size: 18px;
            border-radius: 12px;
            height: 50px;
            width: 200px;
        }
    </style>
""", unsafe_allow_html=True)

# 🪄 Optional: Lottie animation (requires internet)
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

animation = load_lottie_url("https://lottie.host/3acb0cb2-6f82-45b3-84f2-e73dd3e4dc0d/BYlvG7ZNo7.json")
if animation:
    st_lottie(animation, height=180)

# 📦 Load trained model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# 🔧 Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# 🧠 Title and Instructions
st.markdown("""
    <h1 style='
        text-align: center;
        color: #5A189A;
        font-size: 36px;
        margin-top: 10px;
        margin-bottom: 5px;
    '>
        🍽️ Sentiment Analysis - Restaurant Reviews
    </h1>
""", unsafe_allow_html=True)




# ✏️ Input Area
st.markdown("### <div class='review-label'>✍ Enter your review below:</div>", unsafe_allow_html=True)
review_input = st.text_area("", placeholder="Type your review...", key="review", height=150)

# 🚀 Prediction
if st.button("🔍 Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        cleaned = preprocess_text(review_input)
        vectorized = vectorizer.transform([cleaned]).toarray()
        prediction = model.predict(vectorized)

        if prediction[0] == 1:
            st.success("😍 Positive Review")
        else:
            st.error("😠 Negative Review")


# 🧾 Footer
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Made with ❤️ by <b>Sebi Verma</b><br>
United College of Engineering and Research
</p>
""", unsafe_allow_html=True)
