
# ✨ Sentiment Analysis - Amazon Product Reviews

A stylish Streamlit web application that predicts whether an Amazon Product review is **Positive 😍** or **Negative 😠** using the **Naive Bayes classifier** and **TF-IDF vectorization**.

---

## 📌 Project Overview

This app reads an Amazon Product review, processes the text using NLP techniques, converts it into numeric form using **TF-IDF**, and predicts sentiment using a trained **Multinomial Naive Bayes model**.

---

## 👩‍💻 Developed By

**Sebi Verma**  
United College of Engineering and Research

---

## 🧠 Technologies & Libraries

- Python 🐍
- Streamlit (for UI)
- NLTK (for NLP preprocessing)
- scikit-learn (for TF-IDF and Naive Bayes)
- joblib (for saving model/vectorizer)
- streamlit-lottie (for animation)

---

## 🧪 Dataset Used

- amazon_reviews_dataset1.csv
- amazon_reviews_dataset2.csv

These datasets contain Amazon Product reviews labeled as **Liked (1)** or **Not Liked (0)**.

---

## ⚙️ How It Works

1. **Text Cleaning**  
   - Remove punctuation, lowercase text, remove stopwords, and apply stemming

2. **Vectorization**  
   - TF-IDF converts cleaned text into numerical feature vectors

3. **Model**  
   - Multinomial Naive Bayes classifier trained on labeled data

4. **Prediction**  
   - Input review is preprocessed, vectorized, and predicted as Positive or Negative

---

## 💻 How to Run the App

### 1. Clone the repository or copy the code

```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📊 Evaluation Metrics

- **Accuracy**: 89.76%
- **Precision**: 85.29%
- **Recall**: 87.18%
- **F1-Score**: 90.01%

---

## 🧾 Folder Structure

```
IBM_SAOR/
│
├── app.py                 ← Main Streamlit app
├── sentiment_model.pkl    ← Trained Naive Bayes model
├── vectorizer.pkl         ← TF-IDF vectorizer
├── requirements.txt       ← Project dependencies
├── README.md              ← You're reading this!
```

---

## 🌟 Features

- 🎨 Lavender gradient UI with custom fonts and icons
- ✏️ Styled input box and prediction button
- 🧠 Accurate sentiment prediction using NLP
- 💖 Animated Lottie support
- 📱 Ready for local or cloud deployment

---

## 📌 License

This project is for educational purposes and is open to enhancement.
