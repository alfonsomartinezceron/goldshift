import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from goldshift.ml_logic.registry import load_model
from goldshift.ml_logic.preprocessor import preprocess_features

app = FastAPI()

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
def predict(
        from_datetime: str,  # 2014-07-06 19:18:00
        to_datetime: str  # 2014-07-06 19:18:00
    ):      # 1
    """
    Make a prediction.
    Assumes `from_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `from_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """
    # Preprocess the features and convert the arguments to a dataframe.
  # X_prep = preprocess_features(pd.DataFrame.from_dict(
  #     dict([ ('from_datetime', [pd.Timestamp(pickup_datetime, tz='UTC')]),
  #            ('to_datetime', [pd.Timestamp(pickup_datetime, tz='UTC')]) ] )))
  # model = load_model()
  # # model = app.state.model  # Load model only one time. (tbd)
  # # assert model is not None
  # y_pred = model.predict(X_prep)
  # print()
  # print(y_pred)
  # print()
    # return dict(fare= float(y_pred))  # FastAPI does need a dict().
    return dict(estimate="Hello, World")  # FastAPI does need a dict().



# What will happen when website 127.0.0.1/8000 is opened? (127.0.0.1 is
# localhost)
@app.get("/")
def root():
    return {'greeting': 'Hello'}
