import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
current_dir = os.path.dirname(__file__)
model = joblib.load(
    os.path.join(
        current_dir,
        "models",
        "decision_tree_model.pkl"
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
    page_title="Decision Tree Classifier",
    layout="wide"
)
st.title("Heart Disease Prediction using Decision Tree")
df = pd.read_csv(
    os.path.join(
        current_dir,
        "data",
        "heart.csv"
    )
)
st.subheader("Dataset")
st.dataframe(df.head())
X = df.drop("target", axis=1)
y = df["target"]
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
accuracy = accuracy_score(
    y,
    predictions
)
st.subheader("Accuracy")
st.success(
    f"Accuracy: {accuracy:.4f}"
)
cm = confusion_matrix(
    y,
    predictions
)
st.subheader("Confusion Matrix")
fig1, ax1 = plt.subplots(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax1
)
ax1.set_title("Confusion Matrix")
st.pyplot(fig1)
st.subheader("Classification Report")
report = classification_report(
    y,
    predictions,
    output_dict=True
)
st.dataframe(
    pd.DataFrame(report).transpose()
)
st.subheader("Feature Distribution")
selected_feature = st.selectbox(
    "Select Feature",
    feature_names
)
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.histplot(
    df[selected_feature],
    kde=True,
    ax=ax2
)
ax2.set_title(
    f"Distribution of {selected_feature}"
)
st.pyplot(fig2)
st.subheader("Manual Prediction")
age = st.number_input(
    "Age",
    value=50
)
sex = st.number_input(
    "Sex",
    value=1
)
cp = st.number_input(
    "Chest Pain Type",
    value=0
)
trestbps = st.number_input(
    "Resting Blood Pressure",
    value=120
)
chol = st.number_input(
    "Cholesterol",
    value=200
)
fbs = st.number_input(
    "Fasting Blood Sugar",
    value=0
)
restecg = st.number_input(
    "Rest ECG",
    value=1
)
thalach = st.number_input(
    "Maximum Heart Rate",
    value=150
)
exang = st.number_input(
    "Exercise Induced Angina",
    value=0
)
oldpeak = st.number_input(
    "Oldpeak",
    value=1.0
)
slope = st.number_input(
    "Slope",
    value=1
)
ca = st.number_input(
    "CA",
    value=0
)
thal = st.number_input(
    "Thal",
    value=2
)
if st.button("Predict"):
    input_data = pd.DataFrame({
        "age":[age],
        "sex":[sex],
        "cp":[cp],
        "trestbps":[trestbps],
        "chol":[chol],
        "fbs":[fbs],
        "restecg":[restecg],
        "thalach":[thalach],
        "exang":[exang],
        "oldpeak":[oldpeak],
        "slope":[slope],
        "ca":[ca],
        "thal":[thal]
    })
    input_scaled = scaler.transform(
        input_data
    )
    prediction = model.predict(
        input_scaled
    )[0]
    probability = np.max(
        model.predict_proba(
            input_scaled
        )
    )
    if prediction == 1:
        st.error(
            "Heart Disease Detected"
        )
    else:
        st.success(
            "No Heart Disease"
        )
    st.info(
        f"Confidence: {probability:.2f}"
    )