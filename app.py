import streamlit as st
import pandas as pd
import joblib
import os
import re # Needed for cleaning check, though not strictly required if names are guaranteed clean

# --- Configuration ---
MODEL_FILENAME = 'concrete_strength_model.joblib'

# --- Load the Trained Model ---
if not os.path.exists(MODEL_FILENAME):
    st.error(f"Error: Model file '{MODEL_FILENAME}' not found in the current directory.")
    st.error("Please run the training script first (e.g., `python train_model.py`) to generate the model file.")
    st.stop()

try:
    model = joblib.load(MODEL_FILENAME)
    # Get feature names as the model expects them (should be the cleaned names)
    model_feature_names = model.feature_names_in_.tolist()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")
st.title('Concrete Compressive Strength Prediction')
st.write('Enter the concrete component values below to predict its compressive strength (MPa).')
st.markdown("---")

# --- Input Fields for Features ---
# Use the feature names learned by the loaded model
col1, col2 = st.columns(2)
input_data = {}

default_values = [281.1, 20.0, 0.0, 180.0, 5.0, 970.0, 780.0, 28] # Example defaults
for i, feature in enumerate(model_feature_names):
    target_col = col1 if i < len(model_feature_names) / 2 else col2
    value = default_values[i] if i < len(default_values) else 0.0
    # Basic check for 'Age' to set format/step correctly
    is_age = 'age' in feature.lower() and 'day' in feature.lower()
    format_str = "%d" if is_age else "%.2f"
    step = 1 if is_age else 0.1
    min_val = 1 if is_age else 0.0

    with target_col:
        input_data[feature] = st.number_input(
            label=f'Enter {feature}', # Display the cleaned name from the model
            value=value,
            min_value=min_val,
            step=step,
            format=format_str
        )

st.markdown("---")

# --- Prediction Button ---
if st.button('Predict Strength', type="primary"):
    try:
        # Create DataFrame from input, ensuring correct column order
        input_df = pd.DataFrame([input_data])
        # Reorder columns to exactly match the order the model expects
        input_df = input_df[model_feature_names]

        # Make prediction
        prediction = model.predict(input_df)
        predicted_strength = prediction[0]

        st.subheader('Prediction Result')
        st.success(f'Predicted Concrete Compressive Strength: **{predicted_strength:.2f} MPa**')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        # Adding more detail for debugging if it still fails
        st.error(f"Model expects features: {model.feature_names_in_.tolist()}")
        st.error(f"Input DataFrame columns: {input_df.columns.tolist()}")


st.markdown("---")
st.write("Model based on UCI Concrete Compressive Strength Dataset.")
