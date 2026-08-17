from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from housing_model import build_model_pipeline, load_and_prepare_data, make_features


@st.cache_data(show_spinner=False)
def get_model_and_data():
    df = load_and_prepare_data()
    X, y = make_features(df)
    model = build_model_pipeline(LinearRegression())
    model.fit(X, y)
    return model


st.set_page_config(page_title="Sydney Housing Predictor", page_icon="🏠", layout="wide")
st.title("Sydney Housing Price Prediction and Decision Support")
st.caption("Prototype ML app using a simple linear model trained on the collected Sydney housing dataset.")

model = get_model_and_data()

suburb = st.selectbox("Suburb", options=["Campsie", "Manly", "Parramatta"])
property_type = st.selectbox("Property type", options=["Apartment", "House", "Unit", "Townhouse", "Villa", "Block of units"])
bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3)
bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
parking_spaces = st.number_input("Parking spaces", min_value=0, max_value=6, value=1)
land_size_sqm = st.number_input("Land size (sqm)", min_value=0.0, max_value=2000.0, value=360.0, step=10.0)
floor_area_sqm = st.number_input("Floor area (sqm)", min_value=0.0, max_value=500.0, value=120.0, step=5.0)
sale_month = st.slider("Sale month", min_value=1, max_value=12, value=7)

if floor_area_sqm <= 0:
    st.warning("Floor area should be greater than zero for a meaningful prediction.")
    st.stop()

price_per_sqm = (bedrooms * 300000 + bathrooms * 200000 + parking_spaces * 75000) / max(floor_area_sqm, 1)

row = pd.DataFrame(
    [{
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spaces": parking_spaces,
        "land_size_sqm": land_size_sqm,
        "floor_area_sqm": floor_area_sqm,
        "sale_month": sale_month,
        "price_per_sqm": price_per_sqm,
        "bed_bath_ratio": bedrooms / max(bathrooms, 1),
        "parking_to_bed_ratio": parking_spaces / max(bedrooms, 1),
    }]
)

predicted_price = model.predict(row)[0]

st.subheader("Estimated sale price")
st.metric("Predicted Price", f"${predicted_price:,.0f}")

st.markdown("### Inputs used")
st.dataframe(row, width="stretch")

st.markdown(
    """
    This prototype is meant for decision support and should not replace a professional valuation.
    It is best used as a quick benchmark in the context of Sydney’s diversified housing markets.
    """
)
