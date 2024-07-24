import pandas as pd
import numpy as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from goldshift.ml_logic.model import *
from goldshift.ml_logic.data import *
# from main import load_model
from contextlib import asynccontextmanager
#   from goldshift.ml_logic.registry import load_model
#   from goldshift.ml_logic.preprocessor import preprocess_features

app = FastAPI()
app.state.model = load_model()  # When is this line of code started?
app.state.df = load_data()

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
def predict(num_days=1):  # number of days starting at 01.07.2024
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

    #x_new = tf.expand_dims(tf.convert_to_tensor([2324.4,
    #    2324.3,
    #    2351.6,
    #    2335.1,
    #    2328.8,
    #    2325.1,
    #    2299.7,
    #    2323.6,
    #    2330.9,
    #    2329.1]),-1)

    #x_new = tf.expand_dims(x_new,axis=0)
    #y_prediction = app.state.model.predict(x_new)
    #b = y_prediction.flatten().tolist()

    # test_input = np.random.random((limit, 10))
    # y_pred = app.state.model.predict(test_input)

    last_ten = app.store.df['gold_price'].iloc[-10:]
    y_last_ten = np.array(app.store.df.drop(columns= ['gold_price']).iloc[-10:])

    y_new = tf.convert_to_tensor(y_last_ten)
    y_new = tf.expand_dims(y_new,axis=0)
    y_prediction = app.state.model.predict(y_new)
    b = last_ten[-1]+last_ten[-1] *(y_prediction[0][0]/100)

    for i, v in enumerate(b):
        result_dict[f"The prediction for next day is {i+1}"] = v
    return result_dict  # Current value of the gold today as dummy.


# What will happen when website 127.0.0.1/8000 is opened? (127.0.0.1 is
# localhost)
@app.get("/")
def root():
    return {'greeting': 'Hello'}
