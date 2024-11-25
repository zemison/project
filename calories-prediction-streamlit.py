import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Calories Burned Prediction", layout="wide")

# App Header
st.write("## Calories Burned Prediction")
st.image(
    "https://assets.considerable.com/wp-content/uploads/2019/07/03093250/ExerciseRegimenPano.jpg", 
    use_column_width=True
)
st.write("""
In this WebApp, you can predict the calories burned during exercise by providing parameters such as Age, Gender, BMI, etc.
""")

# Sidebar input parameters
st.sidebar.header("User Input Parameters : ")

def user_input_features():
    age = st.sidebar.slider("Age : ", 10, 100, 30)  # Integer slider
    bmi = st.sidebar.slider("BMI : ", 15, 40, 20)  # Integer slider
    duration = st.sidebar.slider("Duration (min) : ", 0, 60, 30)  # Integer slider
    heart_rate = st.sidebar.slider("Heart Rate : ", 60, 200, 100)  # Integer slider
    body_temp = st.sidebar.slider("Body Temperature (°C) : ", 36.0, 42.0, 37.5, step=0.1)  # Float slider
    gender = st.sidebar.radio("Gender : ", ("Male", "Female"))  # Radio button for gender selection
    gender_encoded = 1 if gender == "Male" else 0  # Encoding gender as 1 for Male and 0 for Female

    # Dataframes for display and model input
    data_display = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart Rate": heart_rate,
        "Body Temp (°C)": body_temp,
        "Gender": gender
    }

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_encoded
    }

    return pd.DataFrame(data_display, index=[0]), pd.DataFrame(data_model, index=[0])

data_display, data_model = user_input_features()

# Show user input parameters with progress bar
st.write("---")
st.header("Your Parameters : ")
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(data_display)

# Load and preprocess data
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    df["BMI"] = round(df["Weight"] / ((df["Height"] / 100) ** 2), 2)
    return df

exercise_df = load_data()

# Prepare training and test sets
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# One-hot encode categorical variables
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the Random Forest model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=1)
random_reg.fit(X_train, y_train)

# Ensure user input features align with training columns
expected_columns = X_train.columns
for col in expected_columns:
    if col not in data_model.columns:
        data_model[col] = 0  # Add missing columns

# Predict calories burned with progress bar
st.write("---")
st.header("Prediction : ")
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

prediction = random_reg.predict(data_model[expected_columns])
st.write(f"### {round(prediction[0], 2)} **kilocalories**")

# Find similar results with progress bar
st.write("---")
st.header("Similar Results : ")
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

range_low, range_high = prediction[0] - 10, prediction[0] + 10
similar_results = exercise_df[
    (exercise_df["Calories"] >= range_low) & (exercise_df["Calories"] <= range_high)
]
if not similar_results.empty:
    st.write(similar_results.sample(min(5, len(similar_results))))
else:
    st.write("No similar results found.")

# Additional analysis with progress bar
st.write("---")
st.header("General Information : ")
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

comparison_stats = {
    "Age": (exercise_df["Age"] < data_model["Age"][0]).mean(),
    "Duration": (exercise_df["Duration"] < data_model["Duration"][0]).mean(),
    "Heart Rate": (exercise_df["Heart_Rate"] < data_model["Heart_Rate"][0]).mean(),
    "Body Temp": (exercise_df["Body_Temp"] < data_model["Body_Temp"][0]).mean()
}

st.write(f"You are older than {comparison_stats['Age'] * 100:.2f}% of people in the dataset.")
st.write(f"You exercised longer than {comparison_stats['Duration'] * 100:.2f}% of people in the dataset.")
st.write(f"Your heart rate was higher than {comparison_stats['Heart Rate'] * 100:.2f}% of people in the dataset.")
st.write(f"Your body temperature was higher than {comparison_stats['Body Temp'] * 100:.2f}% of people in the dataset.")
