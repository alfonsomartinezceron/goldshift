import numpy as np
import time
import statsmodels.api as sm
from datetime import datetime, timedelta
from colorama import Fore, Style

def initialize_model(X: np.series) -> Model:

    # Assuming that there is no seasonal order - dUMMY HYPOTHESIS
    model = sm.tsa.SARIMAX(X, order =(0,1,0), seasonal_order = (0,0,0,0))
    model = model.fit()

    print("✅ Model initialized")

    return model

def prediction_model(model: Model, start: timestamp,
                     end: timestamp) -> pd.DataFrame:
    predictions = model.get_prediction(start = start, end = end, dynamic = False)
    predictions_conf = predictions.conf_init()

    print("✅ Predictions calculated")

    return predictions_conf
