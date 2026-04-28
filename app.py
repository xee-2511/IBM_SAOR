import streamlit as st
import joblib
import re
import base64
import requests
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from streamlit_lottie import st_lottie
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk

# Download NLTK resources
nltk.download('stopwords')

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

# 🔧 Preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# 🏋️ Train model function (runs only once)
@st.cache_resource
def train_model():
    try:
        # Load both datasets
        df1 = pd.read_csv('amazon_reviews_dataset1.csv')
        df2 = pd.read_csv('amazon_reviews_dataset2.csv')
        df = pd.concat([df1, df2]).drop_duplicates()
        
        # Check if columns exist
        if 'review' not in df.columns or 'liked' not in df.columns:
            st.error("CSV files must contain 'review' and 'liked' columns")
            return None, None
            
        df['cleaned_review'] = df['review'].apply(preprocess_text)
        
        # Vectorize and train
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['cleaned_review'])
        y = df['liked']
        
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X, y)
        
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None, None

# 📦 Load trained model & vectorizer
model, vectorizer = train_model()

# 🧠 Title and Instructions
st.markdown("""
    <h1 style='
        text-align: center;
        color: #5A189A;
        font-size: 36px;
        margin-top: 10px;
        margin-bottom: 5px;
    '>
         Sentiment Analysis of Amazon Product Reviews
    </h1>
""", unsafe_allow_html=True)

# ✏️ Input Area
st.markdown("### <div class='review-label'>✍ Enter your review below:</div>", unsafe_allow_html=True)
review_input = st.text_area("", placeholder="Type your review...", key="review", height=150)

# 🚀 Prediction
if st.button("🔍 Predict Sentiment"):
    if review_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    elif model is None or vectorizer is None:
        st.error("Model failed to load. Check your dataset files.")
    else:
        cleaned = preprocess_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        
        # Get probability for better insight
        proba = model.predict_proba(vectorized)[0][1]
        prediction = 1 if proba > 0.45 else 0
        
        if prediction == 1:
            st.success(f"😍 Positive Review ({proba:.0%} confidence)")
        else:
            st.error(f"😠 Negative Review ({(1-proba):.0%} confidence)")

# 🧾 Footer
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Made with ❤️ by <b>Sebi Verma</b><br>
United College of Engineering and Research
</p>
""", unsafe_allow_html=True)

