"""
House Price Prediction System
A simple ML web app built with Streamlit + scikit-learn.

Run locally:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

# ----------------------------------------------------------------------
# Data + model (cached so it only runs once per session)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Generates a realistic synthetic housing dataset.
    Replace this with pd.read_csv("your_data.csv") to use real data.
    """
    rng = np.random.default_rng(42)
    n = 1500

    area = rng.integers(500, 5000, n)                       # sq. ft.
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    stories = rng.integers(1, 4, n)
    age = rng.integers(0, 40, n)                             # years old
    location_score = rng.integers(1, 10, n)                  # 1=poor, 10=prime
    parking = rng.integers(0, 3, n)

    price = (
        area * 120
        + bedrooms * 50000
        + bathrooms * 35000
        + stories * 20000
        - age * 1500
        + location_score * 40000
        + parking * 15000
        + rng.normal(0, 50000, n)                             # noise
    )
    price = np.clip(price, 50000, None)

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "age": age,
        "location_score": location_score,
        "parking": parking,
        "price": price.astype(int),
    })
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, mae, r2, X.columns.tolist()


df = load_data()
model, mae, r2, feature_names = train_model(df)

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write(
    "Enter the details of a house below and get an estimated market price "
    "from a Random Forest model trained on housing data."
)

with st.expander("ℹ️ Model performance (on held-out test data)"):
    c1, c2 = st.columns(2)
    c1.metric("Mean Absolute Error", f"₹{mae:,.0f}")
    c2.metric("R² Score", f"{r2:.3f}")
    st.caption("This demo uses synthetic data — swap in `load_data()` with your own CSV for real predictions.")

st.subheader("Enter house details")

col1, col2 = st.columns(2)
with col1:
    area = st.slider("Area (sq. ft.)", 500, 5000, 1500, step=50)
    bedrooms = st.slider("Bedrooms", 1, 5, 3)
    bathrooms = st.slider("Bathrooms", 1, 3, 2)
    stories = st.slider("Stories", 1, 3, 1)

with col2:
    age = st.slider("Age of property (years)", 0, 40, 5)
    location_score = st.slider("Location score (1=poor, 10=prime)", 1, 10, 6)
    parking = st.slider("Parking spaces", 0, 2, 1)

input_df = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "age": age,
    "location_score": location_score,
    "parking": parking,
}])[feature_names]

st.divider()

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    prediction = model.predict(input_df)[0]
    st.success(f"### Estimated Price: ₹{prediction:,.0f}")

    st.subheader("What's driving this prediction?")
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)
    st.bar_chart(importances)

st.divider()
with st.expander("📊 View training data sample"):
    st.dataframe(df.head(20), use_container_width=True)
