import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
st.set_page_config(
    page_title="Random Forest Regressor",
    layout="wide"
)
st.title("Salary Prediction using Random Forest Regressor")
current_dir = os.path.dirname(__file__)
data_path = os.path.join(
    current_dir,
    "data",
    "salary.csv"
)
df = pd.read_csv(data_path)
st.subheader("Dataset")
st.dataframe(df.head())
X = df[["YearsExperience"]]
y = df["Salary"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
param_grid = {
    "n_estimators":[50,100,150],
    "max_depth":[2,4,6,None],
    "min_samples_split":[2,5,10]
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=42
    ),
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_search.fit(
    X_train,
    y_train
)
model = grid_search.best_estimator_
st.subheader("Best Parameters")
st.write(
    grid_search.best_params_
)
y_pred = model.predict(X_test)
mae = mean_absolute_error(
    y_test,
    y_pred
)
mse = mean_squared_error(
    y_test,
    y_pred
)
r2 = r2_score(
    y_test,
    y_pred
)
st.subheader("Model Evaluation")
st.success(f"MAE: {mae:.2f}")
st.success(f"MSE: {mse:.2f}")
st.success(f"R2 Score: {r2:.2f}")
st.subheader("Actual vs Predicted")
fig1, ax1 = plt.subplots(figsize=(6,4))
ax1.scatter(
    y_test,
    y_pred
)
ax1.set_xlabel("Actual Salary")
ax1.set_ylabel("Predicted Salary")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)
st.subheader("Prediction Distribution")
fig2, ax2 = plt.subplots(figsize=(6,4))
sns.histplot(
    y_test,
    kde=True,
    label="Actual",
    ax=ax2
)
sns.histplot(
    y_pred,
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
