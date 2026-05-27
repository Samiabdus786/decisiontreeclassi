import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Medical Insurance Classification")
st.markdown("Interactive Decision Tree Classification Dashboard")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

data = load_data()

# ---------------------------------------------------
# Create Classification Target
# ---------------------------------------------------
median_charge = data["charges"].median()

data["insurance_category"] = np.where(
    data["charges"] >= median_charge,
    1,
    0
)

# ---------------------------------------------------
# Encode Categorical Columns
# ---------------------------------------------------
categorical_cols = data.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    data[col] = pd.factorize(data[col])[0]

# ---------------------------------------------------
# Create Models Folder
# ---------------------------------------------------
os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/decision_tree_classifier.pkl"

# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------
X = data.drop(["charges", "insurance_category"], axis=1)

y = data["insurance_category"]

# ---------------------------------------------------
# Train Model if PKL Doesn't Exist
# ---------------------------------------------------
if not os.path.exists(MODEL_PATH):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    params = {
        "max_depth": [3, 5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": [
            "gini",
            "entropy"
        ]
    }

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        params,
        cv=5,
        scoring="accuracy"
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    joblib.dump(model, MODEL_PATH)

else:
    model = joblib.load(MODEL_PATH)

# ---------------------------------------------------
# Dataset Overview
# ---------------------------------------------------
st.subheader("📂 Dataset Overview")

col1, col2 = st.columns([2,1])

with col1:
    st.dataframe(data.head())

with col2:
    st.info(f"Rows: {data.shape[0]}")
    st.info(f"Columns: {data.shape[1]}")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

show_heatmap = st.sidebar.checkbox(
    "Show Correlation Heatmap",
    value=True
)

show_distribution = st.sidebar.checkbox(
    "Show Insurance Category Distribution",
    value=True
)

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------
if show_heatmap:

    st.subheader("📊 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,7))

    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------------------------
# Distribution Plot
# ---------------------------------------------------
if show_distribution:

    st.subheader("📈 Insurance Category Distribution")

    fig = px.histogram(
        data,
        x="insurance_category",
        color="insurance_category",
        title="Insurance Category Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions)
recall = recall_score(y, predictions)
f1 = f1_score(y, predictions)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
st.subheader("📌 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{accuracy:.4f}")
m2.metric("Precision", f"{precision:.4f}")
m3.metric("Recall", f"{recall:.4f}")
m4.metric("F1 Score", f"{f1:.4f}")

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------
st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig = px.bar(
    importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------
st.subheader("📄 Classification Report")

report = classification_report(
    y,
    predictions,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ---------------------------------------------------
# Model Information
# ---------------------------------------------------
st.subheader("🧠 Model Information")

st.success("""
Algorithm Used: Decision Tree Classifier

Target:
- 0 = Low Insurance Charges
- 1 = High Insurance Charges

Hyperparameter Tuning:
- GridSearchCV
- Cross Validation = 5
- Optimized Tree Depth
- Optimized Split Criteria
""")

# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
st.subheader("🩺 Insurance Category Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Age", 18, 70, 30)
    sex = st.selectbox("Sex", [0,1])
    bmi = st.slider("BMI", 10.0, 50.0, 25.0)

with col2:
    children = st.slider("Children", 0, 5, 1)
    smoker = st.selectbox("Smoker", [0,1])
    region = st.selectbox("Region", [0,1,2,3])

# ---------------------------------------------------
# Input Data
# ---------------------------------------------------
input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region
}])

# ---------------------------------------------------
# Match Columns
# ---------------------------------------------------
input_data = input_data.reindex(
    columns=X.columns,
    fill_value=0
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Insurance Category"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success("High Insurance Charges Category")
    else:
        st.info("Low Insurance Charges Category")

    st.write(f"Probability of Low Charges: {probability[0]:.2f}")
    st.write(f"Probability of High Charges: {probability[1]:.2f}")

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    "Made with ❤️ using Streamlit and Scikit-Learn"
)
