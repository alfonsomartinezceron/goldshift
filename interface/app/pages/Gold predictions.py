import streamlit as st
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

st.title("Gold price prediction for tomorrow")
st.write("Click on the button below to see tomorrow's predicted price of gold")
if st.button("Get prediction"):
    result = get_prediction()['day 1']
    st.write(f"Tomorrow, gold will be valued at: {result:.2f}")

#List the features that are used for our predictions
st.text("This prediction uses the following features:\nOil\nNatural gas\nInterest rate (USD)\nYuan\nBrazilian Real\nRussian ruble")
