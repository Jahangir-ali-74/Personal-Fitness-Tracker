import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

# -----------------------------
# Page Configuration (LinkedIn-friendly metadata)
# -----------------------------
st.set_page_config(
    page_title="Personal Fitness Tracker | AI-Powered Calorie Predictor",
    page_icon="🏋️‍♂️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# SEO + Metadata (helps LinkedIn & search engines read your app info)
st.markdown(
    """
    <meta name="title" content="Personal Fitness Tracker | AI-Powered Calorie Predictor">
    <meta name="description" content="An intelligent fitness tracker that predicts calories burned based on age, BMI, gender, and heart rate using Machine Learning. Created by Sayed Jahangir Ali.">
    <meta property="og:title" content="Personal Fitness Tracker | AI-Powered Calorie Predictor">
    <meta property="og:description" content="Predict your calories burned with AI. Built using Streamlit, Python, and Random Forest Regression.">
    <meta property="og:image" content="https://meridianfitness.in/wp-content/uploads/2019/07/collage-strength.jpg">
    <meta property="og:url" content="https://personal-fitness-tracker-jahangir.streamlit.app/">
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Custom Background Styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                          url("https://meridianfitness.in/wp-content/uploads/2019/07/collage-strength.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }

    /* Sidebar styling */
    .css-18e3th9 {
        background-color: rgba(0, 0, 0, 0.6);
    }

    /* Headings and text */
    .css-1v0mbdj, .css-10trblm, .css-1d391kg {
        color: white !important;
    }

    /* Sliders */
    .stSlider > div {
        background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App Title and Description
# -----------------------------
st.title("🏃 Personal Fitness Tracker")
st.markdown(
    "Predict the **calories burned** based on your physical parameters such as age, BMI, gender, heart rate, etc."
)

# -----------------------------
# Sidebar - User Inputs
# -----------------------------
st.sidebar.header("User Input Parameters")

def get_user_input():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.0)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 38.0)
    gender = st.sidebar.radio("Gender", ("Male", "Female"))

    gender_val = 1 if gender == "Male" else 0

    return pd.DataFrame({
        "Age": [age],
        "BMI": [bmi],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
        "Gender_male": [gender_val]
    })

user_data = get_user_input()

# -----------------------------
# Load and Preprocess Data
# -----------------------------
@st.cache_data
def load_and_train_model():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")

    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = round(df["Weight"] / ((df["Height"] / 100) ** 2), 2)

    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Calories", axis=1)
    y = df["Calories"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
    model.fit(X_train, y_train)

    return model, df, X_train.columns

model, full_df, model_columns = load_and_train_model()

# Align user input with model columns
user_data = user_data.reindex(columns=model_columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Your Inputs")
st.write(user_data)

st.subheader("Prediction")
with st.spinner("Calculating calories burned..."):
    time.sleep(1)
    prediction = model.predict(user_data)[0]
    st.success(f"🔥 Estimated Calories Burned: **{round(prediction, 2)} kcal**")

# -----------------------------
# Similar Cases
# -----------------------------
st.subheader("People With Similar Results")
calorie_range = (prediction - 10, prediction + 10)
similar = full_df[(full_df["Calories"] >= calorie_range[0]) & (full_df["Calories"] <= calorie_range[1])]
st.dataframe(similar.sample(min(5, len(similar))))

# -----------------------------
# General Comparison Stats
# -----------------------------
st.subheader("How You Compare to Others")
st.write(f"You're older than **{(full_df['Age'] < user_data['Age'].values[0]).mean() * 100:.2f}%** of others.")
st.write(f"Your exercise duration is longer than **{(full_df['Duration'] < user_data['Duration'].values[0]).mean() * 100:.2f}%** of others.")
st.write(f"Your heart rate is higher than **{(full_df['Heart_Rate'] < user_data['Heart_Rate'].values[0]).mean() * 100:.2f}%** of others.")
st.write(f"Your body temperature is higher than **{(full_df['Body_Temp'] < user_data['Body_Temp'].values[0]).mean() * 100:.2f}%** of others.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <div style='
        background-color: rgba(0, 0, 0, 0.7);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: #fff;
        font-size: 1rem;
        margin-top: 3rem;
    '>
        © 2025 <strong>Sayed Jahangir Ali</strong>. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)