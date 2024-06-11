import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# Load the saved model and preprocessing pipeline
model = joblib.load('house_price_model.pkl')
preprocessing_pipeline = joblib.load('preprocessing_pipeline.pkl')

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Predict"])

if page == "Home":
    st.title("Welcome to the House Price Prediction App")
    st.image("img.jpg")  # Add an image (ensure the image file is in the correct path)
    st.write("""
    This app predicts the price of a house based on several features. 
    Use the Predict page to input the features of the house and get the predicted price.
    """)

elif page == "About":
    st.title("About")
    st.write("""
    This app is developed to predict house prices using machine learning. 
    It takes into account various features of a house to provide an accurate price estimate.
    
    **Contact Details:**
    - **Name:** MYM.SHARAF
    - **Email:** mymsharaff@gmail.com
    - **LinkedIn:** [Your LinkedIn](https://www.linkedin.com/in/mohammed-sharaf-921079256/)
    - **GitHub:** [Your GitHub](https://github.com/MrMYMSharaf)
    """)

elif page == "Predict":
    st.title("House Price Prediction")

    st.write("""
    # Predict the price of a house based on the following features:
    """)

    # Create input fields for user to enter data
    total_sqft = st.number_input('Total Square Feet', value=2680)
    age = st.number_input('Age of the House', value=59)
    date = st.date_input('Date', value=datetime(2014, 5, 2))
    location = st.text_input('Location', value='Shoreline, WA 98133')
    total_rooms = st.number_input('Total Rooms', value=4)
    condition = st.number_input('Condition (1-5)', value=3, min_value=1, max_value=5)
    view = st.number_input('View (0-4)', value=0, min_value=0, max_value=4)
    floors = st.number_input('Number of Floors', value=2)
    renovated = st.number_input('Renovated (0 or 1)', value=1, min_value=0, max_value=1)

    # Convert the date to ordinal format
    date_ordinal = pd.to_datetime(date).toordinal()

    # Create a dataframe from the input data
    new_data = pd.DataFrame({
        'total_sqft': [total_sqft],
        'age': [age],
        'date': [date_ordinal],
        'location': [location],
        'total_rooms': [total_rooms],
        'condition': [condition],
        'view': [view],
        'floors': [floors],
        'renovated': [renovated]
    })

    # Debugging: Print the new data before preprocessing
    st.write("Input Data:")
    st.write(new_data)

    # Preprocess and predict
    try:
        # Transform the new data using the entire pipeline (preprocessing + model)
        predicted_log_price = model.predict(new_data)
        predicted_price = np.exp(predicted_log_price)  # Convert log price to actual price

        st.write(f"## Predicted House Price: ${predicted_price[0]:,.2f}")
    except Exception as e:
        st.write("Error during preprocessing or prediction:")
        st.write(e)
