import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
current_dir = os.path.dirname(__file__)
model = joblib.load(
    os.path.join(
        current_dir,
        "models",
        "random_forest_regressor.pkl"
    )
)
scaler = joblib.load(
    os.path.join(
        current_dir,
        "models",
        "scaler.pkl"
    )
)
feature_names = joblib.load(
    os.path.join(
        current_dir,
        "models",
        "feature_names.pkl"
    )
)
st.set_page_config(
    page_title="Random Forest Regressor",
    layout="wide"
)
st.title("Salary Prediction using Random Forest Regressor")
df = pd.read_csv(
    os.path.join(
        current_dir,
        "data",
        "salary.csv"
    )
)
st.subheader("Dataset")
st.dataframe(df.head())
X = df[["YearsExperience"]]
y = df["Salary"]
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
mae = mean_absolute_error(
    y,
    predictions
)
mse = mean_squared_error(
    y,
    predictions
)
r2 = r2_score(
    y,
    predictions
)
st.subheader("Model Evaluation")
st.success(f"MAE: {mae:.2f}")
st.success(f"MSE: {mse:.2f}")
st.success(f"R2 Score: {r2:.2f}")
st.subheader("Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(
    y,
    predictions
)
ax1.set_xlabel("Actual Salary")
ax1.set_ylabel("Predicted Salary")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)
st.subheader("Salary Distribution")
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.histplot(
    y,
    kde=True,
    label="Actual",
    ax=ax2
)
sns.histplot(
    predictions,
    kde=True,
    label="Predicted",
    ax=ax2
)
ax2.legend()
st.pyplot(fig2)
st.subheader("Manual Prediction")
years_experience = st.number_input(
    "Enter YearsExperience",
    min_value=0.0,
    value=2.0
)
if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        "YearsExperience":[years_experience]
    })
    input_scaled = scaler.transform(
        input_df
    )
    prediction = model.predict(
        input_scaled
    )[0]
    st.success(
        f"Predicted Salary: {prediction:.2f}"
    )