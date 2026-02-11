import streamlit as st
import requests

# FastAPI endpoint
API_URL = "https://ccid-f6qq.onrender.com/predict"

st.set_page_config(
    page_title="House Price Predictor",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write("Predict house price using a Linear Regression model")

# User inputs
area = st.number_input(
    "Area (sqft)",
    min_value=300.0,
    max_value=5000.0,
    value=1200.0
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

if st.button("Predict Price"):
    payload = {
        "area": area,
        "bedrooms": int(bedrooms)   # ✅ important
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()   # ✅ fixed typo
            price = result["predicted_price"]

            st.success(f"💰 Estimated Price: Rs. {price:,.2f}")

        else:
            st.error("❌ Prediction failed. Check API.")

    except requests.exceptions.ConnectionError:
        st.error("🚨 Cannot connect to FastAPI backend.")
