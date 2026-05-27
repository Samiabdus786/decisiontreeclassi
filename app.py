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
    confusion_matrix
)

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Classifier",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction Dashboard")
st.markdown("Interactive Decision Tree Classification System")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

data = load_data()

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

MODEL_PATH = "models/decision_tree_model.pkl"

# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------
X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# ---------------------------------------------------
# Train Model if PKL Doesn't Exist
# ---------------------------------------------------
if not os.path.exists(MODEL_PATH):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    params = {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"]
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
    "Show Target Distribution",
    value=True
)

# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------
if show_heatmap:

    st.subheader("📊 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12,8))

    sns.heatmap(
        data.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# ---------------------------------------------------
# Target Distribution
# ---------------------------------------------------
if show_distribution:

    st.subheader("📈 Heart Disease Distribution")

    fig = px.histogram(
        data,
        x="HeartDisease",
        color="HeartDisease",
        title="Target Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
predictions = model.predict(X)

acc = accuracy_score(y, predictions)
prec = precision_score(y, predictions)
rec = recall_score(y, predictions)
f1 = f1_score(y, predictions)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
st.subheader("📌 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{acc:.4f}")
m2.metric("Precision", f"{prec:.4f}")
m3.metric("Recall", f"{rec:.4f}")
m4.metric("F1 Score", f"{f1:.4f}")

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------
st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y, predictions)

fig, ax = plt.subplots(figsize=(5,5))

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
# Model Information
# ---------------------------------------------------
st.subheader("🧠 Model Information")

st.success("""
Algorithm Used: Decision Tree Classifier

Hyperparameter Tuning:
- GridSearchCV
- Cross Validation = 5
- Criterion = Gini / Entropy
- Optimized Tree Depth
""")

# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
st.subheader("🩺 Patient Details Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.slider("Age", 20, 80, 45)
    Sex = st.selectbox("Sex", [0, 1])
    ChestPainType = st.slider("Chest Pain Type", 0, 3, 1)
    RestingBP = st.slider("Resting BP", 80, 200, 120)

with col2:
    Cholesterol = st.slider("Cholesterol", 0, 600, 200)
    FastingBS = st.selectbox("Fasting Blood Sugar", [0,1])
    RestingECG = st.slider("Resting ECG", 0, 2, 1)
    MaxHR = st.slider("Max Heart Rate", 60, 220, 150)

with col3:
    ExerciseAngina = st.selectbox("Exercise Angina", [0,1])
    Oldpeak = st.slider("Old Peak", 0.0, 6.0, 1.0)
    ST_Slope = st.slider("ST Slope", 0, 2, 1)

# ---------------------------------------------------
# Input Data
# ---------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "Sex": Sex,
    "ChestPainType": ChestPainType,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
    "RestingECG": RestingECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "Oldpeak": Oldpeak,
    "ST_Slope": ST_Slope
}])

# ---------------------------------------------------
# Match Columns
# ---------------------------------------------------
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Heart Disease"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0]

    confidence = np.max(probability) * 100

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠️ High Risk of Heart Disease\n\nConfidence: {confidence:.2f}%"
        )

    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\nConfidence: {confidence:.2f}%"
        )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    "Made with ❤️ using Streamlit and Scikit-Learn"
)