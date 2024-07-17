import numpy as np
import time
import statsmodels.api as sm
from datetime import datetime, timedelta
from colorama import Fore, Style
from goldshift.ml_logic.data import clean_data

def initialize_model():

    # Reading the Excel file
    df = pd.read_excel(../raw_data/'gold_price.xlsx')
    df = clean_data(df)

    # Assuming that there is no seasonal order - Dummy hypothesis
    model = sm.tsa.SARIMAX(X, order =(0,1,0), seasonal_order = (0,0,0,0))
    model = model.fit()

    print("✅ Model initialized")

    return model

def prediction_model(start: timestamp,
                     end: timestamp) -> pd.DataFrame:
    model = initialize_model()
    predictions = model.get_prediction(start = start, end = end, dynamic = False)
    predictions_conf = predictions.conf_init()

    print("✅ Predictions calculated")

    return predictions_conf
