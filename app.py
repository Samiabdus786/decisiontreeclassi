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

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Medical Insurance Cost Prediction")
st.markdown("Interactive Decision Tree Regression Dashboard")

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

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

MODEL_PATH = "models/decision_tree_regressor.pkl"

# ---------------------------------------------------
# Features & Target
# ---------------------------------------------------
X = data.drop("charges", axis=1)
y = data["charges"]

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
        "max_depth": [3, 5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": [
            "squared_error",
            "friedman_mse"
        ]
    }

    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        params,
        cv=5,
        scoring="r2"
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
    "Show Insurance Charges Distribution",
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

    st.subheader("📈 Insurance Charges Distribution")

    fig = px.histogram(
        data,
        x="charges",
        nbins=30,
        title="Insurance Charges Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# Predictions
# ---------------------------------------------------
predictions = model.predict(X)

mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
st.subheader("📌 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("MAE", f"{mae:.2f}")
m2.metric("MSE", f"{mse:.2f}")
m3.metric("RMSE", f"{rmse:.2f}")
m4.metric("R² Score", f"{r2:.4f}")

# ---------------------------------------------------
# Actual vs Predicted
# ---------------------------------------------------
st.subheader("📉 Actual vs Predicted Charges")

comparison = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

fig = px.scatter(
    comparison,
    x="Actual",
    y="Predicted",
    title="Actual vs Predicted Charges"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

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
Algorithm Used: Decision Tree Regressor

Hyperparameter Tuning:
- GridSearchCV
- Cross Validation = 5
- Optimized Tree Depth
- Optimized Split Criteria
""")

# ---------------------------------------------------
# User Input Section
# ---------------------------------------------------
st.subheader("🩺 Insurance Cost Prediction")

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
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("Predict Insurance Cost"):

    prediction = model.predict(input_data)[0]

    st.subheader("💰 Estimated Insurance Charges")

    st.success(
        f"Predicted Insurance Cost: ₹ {prediction:,.2f}"
    )

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")

st.markdown(
    "Made with ❤️ using Streamlit and Scikit-Learn"
)