import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib
import os
import re
import requests # To download the file

# --- Configuration ---
DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
LOCAL_FILENAME = 'Concrete_Data.xls'
MODEL_FILENAME = 'concrete_strength_model.joblib'

# --- Function to Download Dataset ---
def download_file(url, filename):
    """Downloads a file from a URL if it doesn't exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading dataset from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Dataset saved as {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return False
    else:
        print(f"Dataset file '{filename}' already exists.")
    return True

# --- Function to Clean Column Names ---
def clean_col_name(name):
    """Strips whitespace and replaces multiple spaces with one."""
    name = str(name).strip() # Ensure it's a string and strip whitespace
    name = re.sub(r'\s+', ' ', name) # Replace multiple spaces with single space
    return name

# --- Main Training Logic ---
if __name__ == "__main__":
    print("--- Starting Model Training ---")

    # 1. Download Dataset
    if not download_file(DATASET_URL, LOCAL_FILENAME):
        exit("Failed to download dataset. Exiting.") # Stop if download fails

    # 2. Load Data
    print(f"\nLoading data from {LOCAL_FILENAME}...")
    try:
        # Need openpyxl installed (should be in requirements.txt)
        df = pd.read_excel(LOCAL_FILENAME)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print("Make sure 'openpyxl' is installed (`pip install openpyxl`)")
        exit("Failed to load data. Exiting.")

    # 3. Clean Column Names
    print("\nCleaning column names...")
    original_columns = df.columns.tolist()
    df.columns = [clean_col_name(col) for col in df.columns]
    cleaned_columns = df.columns.tolist()
    print("Cleaned columns:", cleaned_columns)

    # 4. Prepare Features (X) and Target (y)
    try:
        target_column_name = cleaned_columns[-1] # Assume target is last column
        print(f"Using target column: '{target_column_name}'")
        X = df.drop(columns=[target_column_name])
        y = df[target_column_name]
        feature_names_clean = X.columns.tolist()
        print("Features prepared.")
    except Exception as e:
        print(f"Error preparing features/target: {e}")
        exit("Failed data preparation. Exiting.")

    # 5. Split Data
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {X_train.shape[0]} train / {X_test.shape[0]} test samples.")

    # 6. Train Model
    print("\nTraining Linear Regression model...")
    try:
        model = LinearRegression()
        # Fit using the DataFrame with cleaned column names
        model.fit(X_train, y_train)
        print("Model training complete.")
        # Check stored names
        print(f"Model stored feature names: {model.feature_names_in_.tolist()}")
    except Exception as e:
        print(f"Error during model training: {e}")
        exit("Failed model training. Exiting.")

    # 7. Evaluate Model (Optional but good practice)
    print("\nEvaluating model performance on test set...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R-squared (R²): {r2:.2f}")

    # 8. Save Model
    print(f"\nSaving trained model to {MODEL_FILENAME}...")
    try:
        joblib.dump(model, MODEL_FILENAME)
        if os.path.exists(MODEL_FILENAME):
            print(f"✅ Model saved successfully.")
        else:
            raise IOError("Model file not found after saving.")
    except Exception as e:
        print(f"Error saving model: {e}")
        exit("Failed model saving. Exiting.")

    print("\n--- Model Training Script Finished ---")
