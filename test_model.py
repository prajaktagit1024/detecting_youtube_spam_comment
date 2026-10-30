import joblib

print("Loading model...")

model = joblib.load("models/spam_model.pkl")

print("Model loaded!")

print("Loading vectorizer...")

vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

print("Vectorizer loaded!")