import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import BaggingClassifier
import joblib

# Load the trained model
model = joblib.load('bagging_model.pkl')

# Load the data (replace with your actual data loading)
try:
    df = pd.read_csv('/content/Heart_Disease_and_Hospitals.csv')
    columns_to_remove = ['full_name', 'country', 'state', 'first_name', 'last_name', 'hospital', 'treatment', 'treatment_date']
    df = df.drop(columns=columns_to_remove, errors='ignore')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    y = df['heart_disease']
    X = df.drop('heart_disease', axis=1)

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_X = pd.DataFrame(encoder.fit_transform(X.select_dtypes(include=['category'])))
    encoded_X.columns = encoder.get_feature_names_out(X.select_dtypes(include=['category']).columns)
    X_encoded = pd.concat([X.select_dtypes(exclude=['category']), encoded_X], axis=1)

except FileNotFoundError:
    st.error("Error: Data file not found. Please make sure 'Heart_Disease_and_Hospitals.csv' is in the correct location.")
    st.stop() # Stop execution if file not found
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()


# Streamlit app
st.title('Heart Disease Prediction')

# Input features (replace with your actual features)
# Example: Numerical input
age = st.number_input('Age', min_value=0, max_value=120, value=50)

# Example: Categorical Input (needs to match the one-hot encoded features)
sex_options = ["Male", "Female"]
sex_index = st.selectbox("Sex", sex_options)

# Create a dictionary to hold the input data in a format that matches X_encoded
input_data = {
    'age': age,
}
if sex_index == 'Male':
  input_data['sex_Male'] = 1
  input_data['sex_Female'] = 0
else:
  input_data['sex_Male'] = 0
  input_data['sex_Female'] = 1
# ... (add other input fields for remaining features)


# Create a DataFrame from the user input
input_df = pd.DataFrame([input_data])


# Preprocess input
try:
    encoded_input = pd.DataFrame(encoder.transform(input_df.select_dtypes(include=['category'])))
    encoded_input.columns = encoder.get_feature_names_out(input_df.select_dtypes(include=['category']).columns)
    input_df_encoded = pd.concat([input_df.select_dtypes(exclude=['category']), encoded_input], axis=1)
    # Align columns for prediction
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_df_encoded)

    # Display prediction
    st.write(f'Prediction: {prediction[0]}')
except ValueError as e:
    st.error(f"Error during prediction: {e}")