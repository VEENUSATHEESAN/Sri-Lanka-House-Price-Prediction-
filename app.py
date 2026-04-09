import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load saved objects
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
X_columns = pickle.load(open("X_columns.pkl", "rb"))

# -----------------------------
# District → Area Mapping
# -----------------------------
district_areas = {
    "Ampara": ["Ampara Central"],

    "Anuradhapura": [
        "New Town", "Nuwaragam Palatha", "Madawachchiya"
    ],

    "Badulla": [
        "Badulla Town", "Bandarawela", "Hali Ela"
    ],

    "Batticaloa": [
        "Batticaloa Town", "Eravur", "Kallady"
    ],

    "Colombo": [
        "Narahenpita", "Nugegoda", "Wellawatte", "Borella",
        "Bambalapitiya", "Rajagiriya", "Mount Lavinia",
        "Dehiwala", "Kollupitiya"
    ],

    "Galle": [
        "Unawatuna", "Karapitiya", "Hikkaduwa", "Galle Fort"
    ],

    "Gampaha": [
        "Ja-Ela", "Ragama", "Wattala",
        "Gampaha Town", "Kadawatha", "Negombo"
    ],

    "Hambantota": [
        "Hambantota Town", "Tangalle", "Ambalantota"
    ],

    "Jaffna": [
        "Chunnakam", "Jaffna Town", "Kokuvil", "Nallur"
    ],

    "Kalutara": [
        "Beruwala", "Panadura", "Wadduwa", "Kalutara North"
    ],

    "Kandy": [
        "Kandy City", "Peradeniya", "Katugastota",
        "Gatambe", "Tennekumbura"
    ],

    "Kegalle": ["Kegalle Central"],

    "Kilinochchi": ["Kilinochchi Central"],

    "Kurunegala": [
        "Melsiripura", "Pannala",
        "Kurunegala Town", "Polgahawela"
    ],

    "Mannar": ["Mannar Central"],

    "Matale": ["Matale Central"],

    "Matara": [
        "Matara Town", "Weligama", "Nupe", "Akurugoda"
    ],

    "Monaragala": ["Monaragala Central"],

    "Mullaitivu": ["Mullaitivu Central"],

    "Nuwara Eliya": ["Nuwara Eliya Central"],

    "Polonnaruwa": ["Polonnaruwa Central"],

    "Puttalam": ["Puttalam Central"],

    "Ratnapura": [
        "Kuruwita", "Ratnapura Town", "Pelmadulla"
    ],

    "Trincomalee": [
        "Nilaveli", "Uppuveli", "China Bay"
    ],

    "Vavuniya": ["Vavuniya Central"]
}


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Sri Lanka House Price Prediction", layout="centered")

st.title("🏠 Sri Lanka House Price Prediction")
st.write("Enter house details to predict the estimated price (LKR)")

# -----------------------------
# User Inputs
# -----------------------------
district = st.selectbox("District", list(district_areas.keys()))

area = st.selectbox(
    "Area",
    district_areas[district]
)

perch = st.number_input("Land Size (Perch)", min_value=1, max_value=100, value=10)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
kitchen_area_sqft = st.number_input("Kitchen Area (sqft)", min_value=30, max_value=300, value=90)
parking_spots = st.number_input("Parking Spots", min_value=0, max_value=5, value=1)
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
year_built = st.number_input("Year Built", min_value=1980, max_value=2025, value=2015)

has_garden = st.selectbox("Garden Available", [True, False])
has_ac = st.selectbox("Air Conditioning", [True, False])
water_supply = st.selectbox("Water Supply", ["Pipe-borne", "Well"])
electricity = st.selectbox("Electricity Type", ["Single phase", "Three phase"])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict House Price"):

    input_data = {
        "district": district,
        "area": area,
        "perch": perch,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "kitchen_area_sqft": kitchen_area_sqft,
        "parking_spots": parking_spots,
        "has_garden": int(has_garden),
        "has_ac": int(has_ac),
        "water_supply": water_supply,
        "electricity": electricity,
        "floors": floors,
        "year_built": year_built
    }

    df_input = pd.DataFrame([input_data])

    # One-hot encoding
    df_input = pd.get_dummies(df_input)

    # Align columns with training data
    df_input = df_input.reindex(columns=X_columns, fill_value=0)

    # Scale numerical features
    numerical_features = [
        "perch", "bedrooms", "bathrooms",
        "kitchen_area_sqft", "parking_spots",
        "floors", "year_built"
    ]

    df_input[numerical_features] = scaler.transform(
        df_input[numerical_features]
    )

    # Predict
    prediction = model.predict(df_input)

    st.success(f"💰 Predicted House Price: {prediction[0]:,.2f} LKR")
