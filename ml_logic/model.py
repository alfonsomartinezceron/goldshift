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
from goldsghift.params import *

def initialize_model(input_shape, X_train):
    # input_shape --> sequence length (10,1)
    # X --> X_train which coming from 'get_X_y' function
    normalizer = Normalization(input_shape)
    normalizer.adapt(X_train)
    model = Sequential()
    model.add(normalizer)
    model.add(LSTM(units=64, activation ='tanh',return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25,activation ='relu'))
    model.add(Dense(units=1, activation = 'linear'))

    print("✅ Model initialized")

    return model

def compile_model(model, learning_rate=0.0005):
    """
    Compile the Neural Network
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        batch_size=32,
        patience=10,

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

    print(f"✅ Model trained on {len(x_train)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history

def evaluate_model(
        model: Model,
        X_test: np.ndarray, # X --> X_test
        y_test: np.ndarray, # y --> y_test
        batch_size=32
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size=batch_size,
        return_dict=True
    )
    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
