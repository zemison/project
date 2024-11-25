import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Load training data
train = pd.read_csv('X_train.csv')

# Streamlit App
st.title("Calorie Prediction App")
st.write("Predict calorie burn based on personal and exercise details.")

# Display default data
if st.checkbox("Show Training Data Preview"):
    st.write("### Training Data Preview")
    st.dataframe(train.head())

# Sidebar for user input
st.sidebar.header("Input Features")
Gender = st.sidebar.selectbox("Gender", options=["Male", "Female"], index=0)
Age = st.sidebar.number_input("Age (years)", min_value=1, max_value=100, value=25)
Height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
Weight = st.sidebar.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
Duration = st.sidebar.number_input("Duration (minutes)", min_value=1, max_value=500, value=30)
Heart_Rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=75)
Body_Temp = st.sidebar.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)

# Convert Gender to numeric
Gender_numeric = 1 if Gender == "Male" else 0

# Predict button
if st.button("Predict"):
    # Prepare the input features
    features = np.array([[Gender_numeric, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]])
    
    # Make the prediction
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Calorie Burn: {prediction[0]:.2f} calories")
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
