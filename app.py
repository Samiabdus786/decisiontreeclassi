import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ================= Page Config =================

st.set_page_config(
    page_title="KNN Regression Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 California Housing Price Prediction")

# ================= Create Folders Automatically =================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================= Load/Create Dataset =================

csv_path = "data/california_housing.csv"

if not os.path.exists(csv_path):

    housing = fetch_california_housing()

    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["PRICE"] = housing.target

    df.to_csv(csv_path, index=False)

else:

    df = pd.read_csv(csv_path)

# ================= Features & Target =================

X = df.drop("PRICE", axis=1)
y = df["PRICE"]

# ================= Train-Test Split =================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================= Scaling =================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================= Sidebar =================

st.sidebar.header("⚙️ KNN Hyperparameters")

k = st.sidebar.slider(
    "Number of Neighbors (K)",
    1,
    20,
    5
)

weights = st.sidebar.selectbox(
    "Weights",
    ["uniform", "distance"]
)

metric = st.sidebar.selectbox(
    "Distance Metric",
    ["minkowski", "euclidean", "manhattan"]
)

# ================= Train Model =================

model = KNeighborsRegressor(
    n_neighbors=k,
    weights=weights,
    metric=metric
)

model.fit(X_train_scaled, y_train)

# ================= Save Model =================

joblib.dump(model, "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ================= Predictions =================

y_pred = model.predict(X_test_scaled)

# ================= Metrics =================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ================= Display Metrics =================

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MSE", f"{mse:.4f}")

with col2:
    st.metric("RMSE", f"{rmse:.4f}")

with col3:
    st.metric("MAE", f"{mae:.4f}")

with col4:
    st.metric("R² Score", f"{r2:.4f}")

# ================= Dataset Preview =================

st.header("📂 Dataset Preview")

st.dataframe(df.head())

# ================= Statistical Summary =================

st.header("📋 Dataset Statistics")

st.dataframe(df.describe())

# ================= Correlation Heatmap =================

st.header("📈 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))

corr = df.corr()

heatmap = ax.imshow(corr, cmap="coolwarm")

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

plt.colorbar(heatmap)

st.pyplot(fig)

# ================= Actual vs Predicted =================

st.header("📉 Actual vs Predicted")

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(y_test, y_pred)

ax2.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

ax2.set_xlabel("Actual Values")
ax2.set_ylabel("Predicted Values")
ax2.set_title("Actual vs Predicted")

st.pyplot(fig2)

# ================= Error Distribution =================

st.header("📊 Error Distribution")

errors = y_test - y_pred

fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.hist(errors, bins=30)

ax3.set_xlabel("Prediction Error")
ax3.set_ylabel("Frequency")
ax3.set_title("Distribution of Errors")

st.pyplot(fig3)

# ================= Custom Prediction =================

st.header("🏠 Predict House Price")

input_data = {}

for column in X.columns:

    input_data[column] = st.number_input(
        f"Enter {column}",
        value=float(X[column].mean())
    )

if st.button("Predict"):

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted House Price: {prediction[0]:.4f}"
    )

# ================= Footer =================

st.success("🎉 KNN Regression Dashboard Completed Successfully")