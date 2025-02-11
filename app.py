import streamlit as st
import pandas as pd
import numpy as np
import pickle

# App title
st.title('Car Price Prediction App')
st.write('This app predicts the **selling price of used cars** based on the input features.')

# Load the trained model and feature columns
with open('m-columns.pkl', 'rb') as f:
    columns = pickle.load(f)
with open('m-lr.pkl', 'rb') as f:
    model = pickle.load(f)

# Input fields for user
st.sidebar.header('Input Features')

brand = st.sidebar.selectbox('Car Brand', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Others'])
year = st.sidebar.number_input('Year of Manufacture', min_value=1990, max_value=2025, value=2015, step=1)
km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000)
fuel = st.sidebar.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
owner = st.sidebar.selectbox('Number of Previous Owners', ['First', 'Second', 'Third', 'Fourth & Above'])

# Feature engineering
car_age = 2025 - year

# Convert user input to a DataFrame
user_data = pd.DataFrame({
    'brand': [brand],
    'km_driven': [np.log1p(km_driven)],  # Apply log transformation
    'fuel': [fuel],
    'owner': [owner],
    'car_age': [car_age]
})
