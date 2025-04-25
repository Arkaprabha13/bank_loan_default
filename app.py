import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Define file paths (you can upload files through Streamlit UI later)
csv_file_path = "Loan_default.csv"  # Allow users to upload their dataset
preprocessor_path = "loan_default_preprocessor.pkl"  # Path to the preprocessor file
model_options = {
    "ANN": "ann_loan_default_model.h5",
    "MLP": "mlp_loan_default_model.h5",
    "Best Model": "best_loan_default_model.h5"
}

# Load the models and preprocessor once
models = {name: load_model(path) for name, path in model_options.items()}
preprocessor = joblib.load(preprocessor_path)

# Streamlit UI
st.title("Loan Default Prediction")

# Upload CSV file (Loan data)
uploaded_file = st.file_uploader("Upload Loan Default Data", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(f"Loaded Data: {data.shape[0]} rows and {data.shape[1]} columns.")
    
    # Display data preview
    st.dataframe(data.head())
    
    # Drop the 'LoanID' column from the data
    data = data.drop(columns=["LoanID"])
    
    # Separate features and labels
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Preprocess the data using the preprocessor
    X_scaled = preprocessor.transform(X)

    # Initialize results container
    results = []

    # Evaluate models
    for model_name, model in models.items():
        st.write(f"Evaluating {model_name} model...")

        # Predict the outputs (class labels)
        y_pred = model.predict(X_scaled)

        # Convert predictions to binary labels if needed
        if y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = (y_pred > 0.5).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y, y_pred)
        class_report = classification_report(y, y_pred)

        # Store the results for display
        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Classification Report": class_report
        })

    # Display results in a dataframe
    results_df = pd.DataFrame(results)
    st.write("Comparison of Models:", results_df)

# Instructions for using the app
else:
    st.write("Please upload a CSV file containing the loan data.")
