import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load The model
model = joblib.load('rent_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.image("house.png", caption="My Image", use_container_width=True)

#Page configuration
st.set_page_config(page_title="House Rent Prediction", page_icon=":house:" , layout="centered")
st.title("House Rent Prediction App")
st.write("Predict monthly house rent based on property details")

#input fields
bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
size = st.number_input("Size (sq.ft)", min_value=300, max_value=10000, value=2500)
area_type = st.selectbox("Area Type",["Super built-up  Area", "Built-up  Area", "Carpet  Area"])
city = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
furnishing = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
tenant = st.selectbox("Tenant Preferred",["Family", "Bachelors", "Bachelors/Family"])
bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
contact = st.selectbox("Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])

#Predict button
if st.button("🔍 Predict Rent"):

    input_data = pd.DataFrame({
        'BHK': [bhk],
        'Size': [size],
        'Area Type': [area_type],
        'City': [city],
        'Furnishing Status': [furnishing],
        'Tenant Preferred': [tenant],
        'Bathroom': [bathroom],
        'Point of Contact': [contact]
    })

# Encode & scale
encoded_data = encoder.transform(input_data)
scaled_data = scaler.transform(encoded_data)
prediction = model.predict(scaled_data)
st.success(f"💰 Predicted Monthly Rent: ₹ {prediction[0]:,.2f}")
