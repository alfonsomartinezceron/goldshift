import streamlit as st
import requests
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

# Function to interact with FastAPI backend
def get_prediction():
    url = f"https://goldshift-usqqma72oq-ew.a.run.app/predict"
    response = requests.get(url)
    return response.json()

st.title("Goldshift future predictions")

num_days = st.number_input("Select the number of days you want predictions for:", min_value=1, step=1, max_value=10)

if st.button("Get prediction"):
    result = get_prediction(num_days)
    st.write(result)

# Streamlit UI
st.title("Future Predictions")
st.write("Enter the number of days you want predictions for:")

days = st.number_input("Number of days", min_value=1, max_value=365, value=1)

if st.button("Predict"):
    future_dates, predictions = predict_days(days)
    st.write("Predictions for the next {} days:".format(days))
    for date, prediction in zip(future_dates, predictions):
        st.write(f"{date.date()}: {prediction}")
