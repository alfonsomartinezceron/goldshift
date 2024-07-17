import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from goldshift.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Data with only business days
    """

    # The model will only work with that range of data (daily basis)
    # Business days from Monday to Friday

    business_days = pd.bdate_range(start='2004-01-01', end='2024-07-01')
    df_bd = pd.DataFrame({'DATE': business_days})
    df = pd.merge(df_bd, df, on='DATE', how='left')

    df = df.drop_duplicates()

    # NaN will be filled with the previous value (i.e. because of holiday days)
    df = df.ffill()

    # If there are still NaN, it is forced with the following value
    df = df.bfill()

    df.set_index('DATE', inplace=True)

    print("✅ data cleaned")

    return df
