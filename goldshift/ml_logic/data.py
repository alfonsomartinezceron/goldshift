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
    df1 = pd.read_excel('goldshift/raw_data/gold_price.xlsx')
    df1 = df1.rename(columns = {'DATE': 'timestamp','USD': 'gold_price'})
    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    df = pd.merge(df, df1, on='timestamp', how='left')
    return df

def load_data():
    # Dataframe with the info from API Alpha commodities
    df_alpha = pd.read_csv('goldshift/raw_data/alpha.csv')
    df_alpha = df_alpha.drop('Unnamed: 0',axis=1)
    df_alpha = df_alpha.drop(columns= ['BRENT','TREASURY_YIELD'])
    df_alpha = df_alpha.replace('.', np.nan)
    listcols=['WTI','FEDERAL_FUNDS_RATE','NATURAL_GAS']
    for col in listcols:
        df_alpha[col] = pd.to_numeric(df_alpha[col],downcast = 'float')

    # Dataframe with the info from API Alpha gold companies
    df_stock = pd.read_csv('goldshift/raw_data/stock.csv')
    df_stock = df_stock.drop('Unnamed: 0',axis=1)
    df_stock = df_stock[['timestamp','NEM','KGC','HMY','CDE']]

    # Dataframe with the info from Kaggle exchange rates
    df_exchange = pd.read_csv('goldshift/raw_data/kaggle.csv')
    df_exchange = df_exchange.drop('Unnamed: 0',axis=1)
    df_exchange = df_exchange[['timestamp','chinese_yuan_to_usd','brazilian_real_to_usd','russian_ruble_to_usd',
                              'south_african_rand_to_usd','indian_rupee_to_usd']]

    # Dataframe with gold prices
    df_gold = pd.read_csv('goldshift/raw_data/gold_usd.csv')
    df_gold = df_gold.drop('Unnamed: 0',axis=1)

    #Adding a % column to gold price variation between T and T-1
    df_gold['gold_old'] = df_gold['gold_price'].shift(1)
    df_gold['gold_change'] = ((df_gold['gold_price'] - df_gold['gold_old']) / df_gold['gold_old']) * 100
    df_gold.drop('gold_old', axis=1, inplace=True)

    df = pd.merge(df_gold,df_alpha, on='timestamp', how='left')
    df = pd.merge(df,df_exchange, on='timestamp', how='left')
    df = pd.merge(df,df_stock, on='timestamp', how='left')
    df.set_index('timestamp', inplace=True)

    #We drop duplicate in case there are
    df = df.drop_duplicates()

    # NaN will be filled with the previous value (i.e. because of holiday days)
    df = df.ffill()

    # If there are still NaN, it is forced with the following value
    df = df.bfill()

    print("✅ data loaded and cleaned")

    return df

def last_ten_gold(df):
    last_ten = df['gold_price'].iloc[-10:]
    y_last_ten = np.array(df.drop(columns= ['gold_price']).iloc[-10:])
    return(last_ten, y_last_ten)

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
