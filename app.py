import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

st.title("Random Forest Regressor")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset")

    st.dataframe(df.head())

    st.subheader("Dataset Information")

    st.write(df.shape)

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    target_column = st.selectbox(
        "Select Target Column",
        numeric_columns
    )

    feature_columns = st.multiselect(
        "Select Feature Columns",
        [col for col in numeric_columns if col != target_column],
        default=[
            col for col in numeric_columns
            if col != target_column
        ][:1]
    )

    if len(feature_columns) > 0:

        X = df[feature_columns]

        y = df[target_column]

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=0.2,
            random_state=42
        )

        param_grid = {
            "n_estimators":[50,100],
            "max_depth":[2,4,None],
            "min_samples_split":[2,5]
        }

        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(
                random_state=42
            ),
            param_grid=param_grid,
            cv=3,
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

        ax1.set_xlabel("Actual")

        ax1.set_ylabel("Predicted")

        ax1.set_title("Actual vs Predicted")

        st.pyplot(fig1)

        st.subheader("Feature Distribution")

        selected_feature = st.selectbox(
            "Select Feature",
            feature_columns
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

        input_data = {}

        for col in feature_columns:

            input_data[col] = st.number_input(
                f"Enter {col}",
                value=float(df[col].mean())
            )

        if st.button("Predict"):

            input_df = pd.DataFrame(
                [input_data]
            )

            input_scaled = scaler.transform(
                input_df
            )

            prediction = model.predict(
                input_scaled
            )[0]

            st.success(
                f"Predicted Value: {prediction:.2f}"
            )

else:

    st.info(
        "Please upload a CSV file to continue"
    )
