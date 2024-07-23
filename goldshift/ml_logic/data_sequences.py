import pandas as pd
import numpy as np
from typing import Tuple
from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

def get_Xi_yi(
    # Generating one a single sequence randomly
    # Inputs:
    # - Train or Test Dataframes from train_test_split
    # - Number of sequences for the train or test dataframes
    df,
    input_length = 10,
    output_length = 1,
    TARGET = 'gold_price') -> Tuple[pd.DataFrame]:
    """given a dataframe (train or test), it returns one sequence (X_i, y_i) as based on the desired
    input_length and output_length with the starting point of the sequence being chosen at random based

    Args:
        df (pd.DataFrame): A single df
        input_length (int): How long each X_i should be
        output_length (int): How long each y_i should be

    Returns:
        Tuple[pd.DataFrame]: A tuple of two dataframes (X_i, y_i)
    """

    first_possible_start = 0
    last_possible_start = len(df) - (input_length + output_length) + 1
    random_start = np.random.randint(first_possible_start, last_possible_start)
    X_i = df.iloc[random_start:random_start+input_length]
    y_i = df.iloc[random_start+input_length:
                  random_start+input_length+output_length][[TARGET]]

    print("✅ generation of single random sequences for the dataframe")

    return (X_i, y_i)

def get_X_y(
    # Generating multiple random sequences
    # Inputs:
    # - Train or Test Dataframes from train_test_split
    # - Number of sequences for the train or test dataframes
    df,
    number_of_sequences,
    input_length = 10,
    output_length = 1):
    X, y = [], []

    for i in range(number_of_sequences):
        (Xi, yi) = get_Xi_yi(df, input_length, output_length)
        X.append(Xi)
        y.append(yi)

    print("✅ generation of multiple random sequences for the dataframe")

    return np.array(X), np.array(y)

# TO USE IT IN THE MAIN

# N_TRAIN = 550 # number_of_sequences_train
# N_TEST = 55  # number_of_sequences_test
# X_train, y_train = get_X_y(train, N_TRAIN, INPUT_LENGTH, OUTPUT_LENGTH)
# X_test, y_test = get_X_y(test, N_TEST, INPUT_LENGTH, OUTPUT_LENGTH)
