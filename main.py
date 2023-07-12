#import funtion from file
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import load_data
import plot
#Library
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import streamlit as st
import os

from keras.models import Sequential , load_model
from keras.layers import LSTM, Dense, Dropout

# title
st.title('Stock Prediction App')
# Data URL
dataset = ( 'CTG_Trade',
           'EIB_Trade',
           'MBB_Trade', 
           'STB_Trade',
           'ACB_Trade')
option = st.selectbox('Select dataset for prediction', dataset)
DATA_URL = ('data/'+option+'.csv')
####

# up load file
uploaded_file = st.file_uploader("Choose a file .csv", type=['csv'])
# load data
####
if uploaded_file is not None:
    df = load_data.dataF(uploaded_file)
    #df = pd.read_csv(uploaded_file)
    st.write("filename:", uploaded_file.name)
else:
    data_load_state = st.text('Loading data...')
    df = load_data.dataF(DATA_URL)
    data_load_state = st.text('Loading data... done!')
####
# columns of dataframe
col_count = len(df.columns)
st.write('Number of columns of data sets: ', col_count)
# rows of dataframe
row_count = len(df)
st.write('Number of rows of data sets: ', row_count)
st.dataframe(df)

# # chart

plot.plot_x_y(df['Open'], 'Open', df['Close'], 'Close')
#plot.plot_x_y(df['High'], 'High', df['Low'], 'Low')


#@st.cache()
def MODEL_LSTM(rebuild = False):


    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)


    # Define the number of time steps to use for prediction
    n_steps = 30

    # Split the data into input sequences and corresponding output values
    X = []
    y = []
    for i in range(n_steps, len(data_scaled)):
        X.append(data_scaled[i-n_steps:i])
        y.append(data_scaled[i])

    X = np.array(X)
    y = np.array(y)

    # Split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    model = None
    _path_model = None
    if uploaded_file is not None:
        _path_model = f'models/{uploaded_file.name.split(".")[0]}.h5'
    else:
        _path_model = f'models/{option}.h5'
    if not os.path.exists(_path_model) or rebuild:
        if _path_model != None:
            
            model = Sequential()
            model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(LSTM(units=50, activation='relu'))
            model.add(Dense(units=y_train.shape[1]))

            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
            model.save(_path_model)
    else:
        model = load_model(_path_model)

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')

    st.write('Test Loss: ', loss)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Inverse transform the predictions and the actual values
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    # Calculate the root mean squared error (RMSE)
    rmse = np.sqrt(np.mean((predictions[:,3] - y_test[:,3])**2))
    st.write('RMSE: ', rmse)

    # Đánh giá MAPE trên tập kiểm tra
    mape = np.mean(np.abs((y_test[:,3] - predictions[:,3]) / y_test[:,3])) * 100
    st.write("Mean Absolute Percentage Error (MAPE):", mape)


    mae = np.mean(np.abs(predictions[:,3] - y_test[:,3]))
    st.write("Mean Absolute Error (MAE):", mae)

    ss_res = np.sum(np.square(predictions[:,3] - y_test[:,3]))
    ss_total = np.sum(np.square(y_test[:,3] - np.mean(y_test[:,3])))
    r2 = 1 - (ss_res / ss_total)
    st.write("R-squared Score (R²):", r2)



    df_final=pd.DataFrame(df)
    df_final = df_final[-len(y_test):]
    df_final['Actual Price'] = y_test[:,3]
    df_final['Predicted Price'] = predictions[:,3]

    plot.plot_x_y(df_final['Actual Price'],'Actual Price',df_final['Predicted Price'],'Predicted Price')


    y_testtest = scaler.inverse_transform(y)
    df_train=pd.DataFrame(df)
    df_train = df_train[30:-len(y_test)]
    df_train['Train'] = y_testtest[:-len(y_test),3]
    plot.plot_x_y_z(df_train['Train'],'Train',df_final['Actual Price'],'Actual Price',df_final['Predicted Price'],'Predicted Price')


    
if __name__ == '__main__':
    _option = st.selectbox("Retrain model:", ['No', 'Yes'])
    if _option == 'Yes':
        MODEL_LSTM(True)
    else:
        MODEL_LSTM()
    #os.remove("modeltrained.h5")
