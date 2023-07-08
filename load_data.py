import pandas as pd
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
def dataF(DATA_URL):
    column_types = {
        'Open': float,
        'High': float,
        'Low' : float,
        'Close': float,
        'Volume': float,
        'percent_change': float
    }
    df = pd.read_csv(DATA_URL, index_col=0, parse_dates=True, dtype=column_types)
    df=df[::-1]
    return df
