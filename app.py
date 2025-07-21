import pandas as pd
import streamlit as st
import joblib

# === Load trained model and vectorizer ===
@st.cache_resource
def load_model():
    # Load the pre-trained model and vectorizer
    model = joblib.load("spam_classifier_learning.pkl")
    vectorizer = joblib.load("spam_vectorizer.pkl")
    return model, vectorizer

# === Load model and vectorizer ===
model, vectorizer = load_model()

# === File Uploader ===
uploaded_file = st.file_uploader("📁 Choose a CSV file", type=["csv"])

# === User Input Text Area ===
user_email = st.text_area("✉️ Enter your email content for prediction:")

# === Handle CSV Input ===
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded CSV Columns:", df.columns.tolist())

    possible_columns = [col for col in df.columns if df[col].dtype == "object"]
    
    if 'email_content' in df.columns:
        content_column = 'email_content'
    elif possible_columns:
        content_column = possible_columns[0]
        st.warning(f"'email_content' not found. Using '{content_column}' instead.")
    else:
        st.error("❌ No suitable text column found in the CSV.")
        content_column = None

    if content_column:
        X = vectorizer.transform(df[content_column].astype(str))
        df['prediction'] = model.predict(X)
        label_map = {0: "Not Spam", 1: "Spam"}
        df['prediction'] = df['prediction'].map(label_map)
        st.success("✅ Predictions generated from uploaded file.")
        st.write(df)

# === Handle User Text Input ===
if user_email:
    X_input = vectorizer.transform([user_email])
    prediction = model.predict(X_input)[0]
    label_map = {0: "Not Spam", 1: "Spam"}
    prediction_label = label_map[prediction]
    st.info(f"🧠 The prediction for your email content is: **{prediction_label}**")
