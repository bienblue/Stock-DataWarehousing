import pandas as pd
import numpy  as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import  mean_squared_error, mean_absolute_error
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

def calculate(x,y):
    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    
    mape_ann = MAPE(x,y)
    RegScoreFun = r2_score(x,y)
    meanAbsoluteError_ann = mean_absolute_error(x,y)
    RMSE_ann = mean_squared_error(x, y)
    return mape_ann,RegScoreFun,meanAbsoluteError_ann,RMSE_ann