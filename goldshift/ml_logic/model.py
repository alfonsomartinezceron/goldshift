import tensorflow as tf
import pickle
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import L1L2
import numpy as np

import time
from datetime import datetime, timedelta
from colorama import Fore, Style
from typing import Tuple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Normalization
# from goldshift.params import *
from goldshift.ml_logic.data import *
from goldshift.ml_logic.data_sequences import *

def load_model():
    model = False
    try:
        model = pickle.load(open("goldshift/model/gold_model.pkl", 'rb'))
    except Exception:
        print("Model could not be loaded...")
    if not model:
        df = load_data()
        df_train, df_test = train_test_split(df,
                     train_test_ratio =0.9,
                     input_length = 10)
        N_TRAIN = 550 # number_of_sequences_train
        N_TEST = 55  # number_of_sequences_test

        X_train, y_train, y_train_price = get_X_y(df_train, N_TRAIN, 10, 1)
        #print(f"Shape {X_train.shape} X_train: {X_train}")
        X_test, y_test, y_test_price = get_X_y(df_test, N_TEST, 10, 1)
        #model = initialize_model((10, 1), X_train)
        model = initialize_model(X_train)
        model = compile_model(model)
        model, history = train_model(model,
        X_train,
        y_train,
        batch_size=32,
        patience=30)
        # (tbd): Save the model.
        try:
            with open("goldshift/model/gold_model.pkl", "wb") as file:
                pickle.dump(model, file)
            print("Saved new model...")
        except Exception:
            pass
    return model

#def initialize_model(input_shape, X_train) -> Model:
def initialize_model(X_train) -> Model:
    # input_shape --> sequence length (10,1)
    # X --> X_train which coming from 'get_X_y' function
    normalizer = Normalization(axis=-1)
    normalizer.adapt(X_train)
    model = Sequential()
    model.add(normalizer)
    model.add(LSTM(units=256, activation ='tanh',return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(units=64,activation ='relu'))
    model.add(Dense(units=32,activation ='relu'))
    model.add(Dense(units=1, activation = 'linear'))

    print("✅ Model initialized")

    return model

# Compile the model

def compile_model(model, learning_rate=0.0005):
    """
    Compile the Neural Network
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model

# First split the data.
def train_model(
        model,
        X_train,
        y_train,
        batch_size=32,
        patience=30
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split = 0.2,
        shuffle = False,
        epochs=1000,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X_train)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

# Print the model summary
def model_summary(model):
    model.summary()

# Test the model
def model_test(model):
    result_dict = {}
    test_input = np.random.random((7, 10))  # Batch of 7 samples, each with 10 features
    predictions = model.predict(test_input)
    # print(predictions)  # Result is 7 times 5 in a matrix.
    return predictions

if __name__ == "__main__":
    model = initialize_model()
    model = compile_model(model)
    model_summary(model)
    model_test(model)
