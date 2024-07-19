import pandas as pd
import numpy as np

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

def create_df_bd():
    # The model will only work with that range of data (daily basis)
    # Business days from Monday to Friday
    business_days = pd.bdate_range(start='2004-01-01', end='2024-07-01') # TO CHANGE THE LAST DATE
    df = pd.DataFrame({'timestamp': business_days})
    return df

def df_gold_price():
    df = create_df_bd()
    df1 = pd.read_excel('../raw_data/gold_price.xlsx')
    df1 = df1.rename(columns = {'DATE': 'timestamp','USD': 'gold_price'})
    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    df = pd.merge(df, df1, on='timestamp', how='left')
    return df

def load_data():
    # Dataframe with the info from API Alpha (commodities and stock market prices of gold companies)
    #df1 = df_alpha()

    # Dataframe with the info from Kaggle exchange rates
    #df2 = df_kaggle()

    # Dataframe with gold prices
    #df3 = df_gold_price()

    #df= pd.merge(df3, df2, on='timestamp', how='left')
    #df= pd.merge(df,df1, on='timestamp', how='left')

    df = df_gold_price()
    df.set_index('timestamp', inplace=True)

    #We drop duplicate in case there are
    df = df.drop_duplicates()

    # NaN will be filled with the previous value (i.e. because of holiday days)
    df = df.ffill()

    # If there are still NaN, it is forced with the following value
    df = df.bfill()

    print("✅ data loaded and cleaned")

    return df

def train_test_split(df:pd.DataFrame,
                     train_test_ratio =0.9,
                     input_length = 10):
    # Function to split the whole dataframe into train and test
    # The train and test outputs are used for the sequences
    # If the ratio is not given 0.9 by default

    # TRAIN SET
    # ======================
    last_train_idx = round(train_test_ratio * len(df))
    df_train = df.iloc[0:last_train_idx, :]

    # TEST SET
    # ======================
    first_test_idx = last_train_idx - input_length
    df_test = df.iloc[first_test_idx:, :]

    print("✅ data splited into train and test")

    return (df_train, df_test)

#  To use it in the main
# train, test = train_test_split(df, TRAIN_TEST_RATIO, INPUT_LENGTH)
