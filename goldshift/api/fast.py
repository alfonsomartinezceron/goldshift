import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from goldshift.ml_logic.model import *
# from main import load_model
from contextlib import asynccontextmanager
#   from goldshift.ml_logic.registry import load_model
#   from goldshift.ml_logic.preprocessor import preprocess_features

app = FastAPI()
app.state.model = load_model()  # When is this line of code started?

# app.state.model = load_model()  # Load only once.

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(num_days=7):  # number of days starting at 01.07.2024
    """
    Make a prediction for the next number 'num_days' of days.
    """
    result_dict = dict()
    # Index check:
    limit = int(num_days)
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000
    test_input = np.random.random((limit, 10))
    y_pred = app.state.model.predict(test_input)
    b = y_pred.flatten().tolist()
    for i, v in enumerate(b):
        result_dict[f"day {i+1}"] = v
    return result_dict  # Current value of the gold today as dummy.


# What will happen when website 127.0.0.1/8000 is opened? (127.0.0.1 is
# localhost)
@app.get("/")
def root():
    return {'greeting': 'Hello'}
