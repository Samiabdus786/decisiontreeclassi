import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ================= Page Config =================

st.set_page_config(
    page_title="KNN Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Breast Cancer Prediction using KNN Classification")

# ================= Create Folders =================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ================= Load/Create Dataset =================

csv_path = "data/breast_cancer.csv"

if not os.path.exists(csv_path):

    cancer = load_breast_cancer()

    df = pd.DataFrame(
        cancer.data,
        columns=cancer.feature_names
    )

    df["target"] = cancer.target

    df.to_csv(csv_path, index=False)

else:

    df = pd.read_csv(csv_path)

# ================= Dataset Info =================

st.header("📂 Dataset Preview")

st.dataframe(df.head())

st.header("📋 Dataset Shape")

st.write(df.shape)

st.header("📊 Class Distribution")

st.bar_chart(df["target"].value_counts())

# ================= Features & Target =================

X = df.drop("target", axis=1)
y = df["target"]

# ================= Train-Test Split =================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
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

model = KNeighborsClassifier(
    n_neighbors=k,
    weights=weights,
    metric=metric
)

model.fit(X_train_scaled, y_train)

# ================= Save Model =================

joblib.dump(model, "models/knn_classifier.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ================= Predictions =================

y_pred = model.predict(X_test_scaled)

# ================= Metrics =================

accuracy = accuracy_score(y_test, y_pred)

st.header("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col2:
    st.metric(
        "Correct Predictions",
        f"{(y_test == y_pred).sum()}"
    )

# ================= Confusion Matrix =================

st.header("📈 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

heatmap = ax.imshow(cm, cmap="Blues")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

plt.colorbar(heatmap)

st.pyplot(fig)

# ================= Classification Report =================

st.header("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ================= Correlation Heatmap =================

st.header("📊 Correlation Heatmap")

fig2, ax2 = plt.subplots(figsize=(12, 8))

corr = df.corr()

heatmap2 = ax2.imshow(corr, cmap="coolwarm")

ax2.set_xticks(range(len(corr.columns)))
ax2.set_yticks(range(len(corr.columns)))

ax2.set_xticklabels(
    corr.columns,
    rotation=90,
    fontsize=7
)

ax2.set_yticklabels(
    corr.columns,
    fontsize=7
)

plt.colorbar(heatmap2)

st.pyplot(fig2)

# ================= Custom Prediction =================

st.header("🔍 Predict Cancer Type")

input_data = {}

for column in X.columns[:10]:

    input_data[column] = st.number_input(
        f"Enter {column}",
        value=float(X[column].mean())
    )

remaining_features = X.columns[10:]

for column in remaining_features:

    input_data[column] = float(X[column].mean())

if st.button("Predict Cancer"):

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Prediction: Benign Tumor")
    else:
        st.error("Prediction: Malignant Tumor")

# ================= Footer =================

st.success("🎉 KNN Classification Dashboard Completed Successfully")