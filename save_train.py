import pandas as pd
import re
import nltk
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# 1. Load datasets
df1 = pd.read_csv("a1_RestaurantReviews_HistoricDump.tsv", sep="\t")
df2 = pd.read_csv("a2_RestaurantReviews_FreshDump.tsv", sep="\t")
df = pd.concat([df1, df2]).dropna()

# 2. Preprocess text
ps = PorterStemmer()
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-letters
    text = text.lower()                   # Convert to lowercase
    words = text.split()                  # Tokenize
    stop_words = set(stopwords.words('english'))
    words = [ps.stem(word) for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(words)

df['cleaned_review'] = df['Review'].apply(preprocess_text)

# 3. Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['Liked']

# 4. Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# 5. Save model and vectorizer using joblib
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
