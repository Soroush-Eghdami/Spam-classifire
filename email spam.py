import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import joblib
import random

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df[['Category', 'Message']]
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['Cleaned'] = df['Message'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Cleaned'])
y = df['Category']

# === 60-20-20 Data Split ===
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_reinforce, y_train, y_reinforce = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 25% of 80% = 20%

print(f"Train size: {X_train.shape[0]}")
print(f"Reinforcement size: {X_reinforce.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# === MODEL SELECTION BASED ON CROSS-VALIDATION ===
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'Linear SVM': LinearSVC(),
    'Random Forest': RandomForestClassifier()
}

print("\nEvaluating models with 5-fold cross-validation...")
best_model = None
best_score = 0
results = {}

for name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    mean_score = np.mean(scores)
    results[name] = mean_score
    print(f"{name}: Accuracy = {mean_score:.4f} ± {np.std(scores):.4f}")
    if mean_score > best_score:
        best_model = name
        best_score = mean_score

print(f"\n✅ Best model selected: {best_model} (Accuracy: {best_score:.4f})")

# === INITIAL TRAINING WITH REPEATED FITTING (3 rounds) ===
selected_model = models[best_model]

print("\nTraining model on 60% data (3 rounds)...")
for i in range(3):
    selected_model.fit(X_train, y_train)
    print(f"Round {i+1}/3 completed.")

# === INITIAL EVALUATION ===
y_pred_initial = selected_model.predict(X_test)
print("\nInitial Evaluation on Test Set (After 60% Training):")
print(f"R² Score: {r2_score(y_test, y_pred_initial):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_initial):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_initial):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_initial))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_initial))

# === REINFORCEMENT LEARNING (extra 20%) ===
print("\nReinforcing model with additional 20% data...")

X_train_combined = np.vstack((X_train.toarray(), X_reinforce.toarray()))
y_train_combined = np.concatenate((y_train, y_reinforce))

# === Add noise to a small fraction of reinforcement data ===
num_noisy = int(0.05 * len(y_reinforce))  # 5% noise
noise_indices = random.sample(range(len(y_train_combined)), num_noisy)
for idx in noise_indices:
    y_train_combined[idx] = 1 - y_train_combined[idx]  # flip label

selected_model.fit(X_train_combined, y_train_combined)

# === FINAL EVALUATION ===
y_pred_final = selected_model.predict(X_test)
print("\nFinal Evaluation on Test Set (After Reinforcement with 20%):")
print(f"R² Score: {r2_score(y_test, y_pred_final):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred_final):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_final))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_final))

# === SAVE FINAL MODEL ===
print("\nSaving final model and vectorizer...")
joblib.dump(selected_model, "spam_classifier_learning.pkl")
joblib.dump(vectorizer, "spam_vectorizer.pkl")