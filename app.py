import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load the trained model
model = joblib.load('bagging_model.pkl')

# Load the encoder (you'll need to save this as well)
encoder = joblib.load('encoder.pkl') # Make sure you have saved the encoder

# Function to preprocess user input
def preprocess_input(input_data, encoder):
    input_df = pd.DataFrame([input_data])

    # Convert object columns to category type
    for col in input_df.columns:
        if input_df[col].dtype == 'object':
            input_df[col] = input_df[col].astype('category')

    # One-hot encode the categorical features
    encoded_input = pd.DataFrame(encoder.transform(input_df.select_dtypes(include=['category'])))
    encoded_input.columns = encoder.get_feature_names_out(input_df.select_dtypes(include=['category']).columns)

    # Concatenate with numerical features
    processed_input = pd.concat([input_df.select_dtypes(exclude=['category']), encoded_input], axis=1)

    return processed_input

# Streamlit app
st.title('Heart Disease Prediction')

# Input features (replace with your actual feature names and types)
age = st.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])  # Example
# ... other input fields (adjust to match your features)


# Example of additional features
chest_pain_type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
resting_blood_pressure = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=120)

# Create a dictionary to hold the input data
input_data = {
    'age': age,
    'sex': sex,
    'chest_pain_type': chest_pain_type,
    'resting_blood_pressure' : resting_blood_pressure,
    # Add other input values here...
}


# Make prediction when the user clicks the button
if st.button('Predict'):
    try:
        processed_input = preprocess_input(input_data, encoder)

        # Ensure that the input to the model matches the features the model was trained on
        # If you get an error like "ValueError: Number of features of the model must match the input"
        # Carefully check for missing or extra features.
        prediction = model.predict(processed_input)
        st.write(f'Prediction: {prediction[0]}')  # Display the prediction

    except ValueError as e:
        st.error(f"Error during prediction: {e}. Please check your input data.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
