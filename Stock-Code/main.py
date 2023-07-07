#import funtion from file
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import load_data
import plot
import ANN_model
#Library
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import streamlit as st

from keras.models import Sequential , load_model
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd

# title
st.title('Stock Prediction App')
# Data URL
dataset = ( 'CTG_Trade',
           'ACB_Trade',
           'EIB_Trade',
           'MBB_Trade', 
           'NVB_Trade',
           'STB_Trade')
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
plot.plot_x_y(df['High'], 'High', df['Low'], 'Low')


#@st.cache()
def MODEL_ANN():


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


    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(units=y_train.shape[1]))

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    model.save('modeltrained.h5')
    #model = load_model('modeltrained.h5')

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
    rmse = np.sqrt(np.mean((predictions - y_test)**2))
    print(f'RMSE: {rmse}')

    st.write('RMSE: ', rmse)

    df_final=pd.DataFrame(df)
    df_final = df_final[-len(y_test):]
    df_final['Actual Price'] = y_test[:,3]
    df_final['Predicted Price'] = predictions[:,3]

    plot.plot_x_y(df_final['Actual Price'],'Actual Price',df_final['Predicted Price'],'Predicted Price')

    calculate_ann = load_data.calculate(df_final['Actual Price'],df_final['Predicted Price'])
    mape_ann,RegScoreFun,meanAbsoluteError_ann,RMSE_ann = calculate_ann[0],calculate_ann[1],calculate_ann[2],calculate_ann[3]
    ### MAE:lỗi tuyệt đối đề cập đến mức độ khác biệt giữa dự đoán của một quan sát và giá trị thực sự của quan sát đó
    st.write('RegScoreFun r2_score- độ phù hợp:',RegScoreFun)
    st.write('MAPE-sai số tương đối trung bình:',mape_ann)
    st.write('meanAbsoluteError-MAE_sai số tuyệt đối trung bình:',meanAbsoluteError_ann)
    st.write('RMSE mean_squared_error-căn bậc 2 của sai số bình phương trung bình:',RMSE_ann)

    y_testtest = scaler.inverse_transform(y)
    df_train=pd.DataFrame(df)
    df_train = df_train[30:-len(y_test)]
    df_train['Train'] = y_testtest[:-len(y_test),3]
    plot.plot_x_y_z(df_train['Train'],'Train',df_final['Actual Price'],'Actual Price',df_final['Predicted Price'],'Predicted Price')


    
if __name__ == '__main__':
    MODEL_ANN()
