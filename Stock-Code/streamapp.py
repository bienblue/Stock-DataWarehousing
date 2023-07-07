import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the stock data into a pandas DataFrame
data = pd.read_csv('data/EIB_Trade.csv')  # Replace 'stock_data.csv' with the path to your data file

# Preprocess the data
data['TradingDate'] = pd.to_datetime(data['TradingDate'])
data = data.sort_values('TradingDate')
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'percent_change']
data = data[features]
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

n_steps = 30
X = []
y = []
dates = []
for i in range(n_steps, len(data_scaled)):
    X.append(data_scaled[i-n_steps:i])
    y.append(data_scaled[i])

dates = np.array(dates)
X_train, X_test, y_train, y_test, train_dates, test_dates = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val, train_dates, val_dates = train_test_split(
    X_train, y_train, train_dates, test_size=0.2, shuffle=False)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=y_train.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)
rmse = np.sqrt(np.mean((predictions - y_test)**2))

# Streamlit app
st.title('Stock Price Prediction')

# Display the evaluation results
st.subheader('Evaluation Results')
st.write('Test Loss:', loss)
st.write('RMSE:', rmse)

# Plot the actual price and the predicted price
plt.figure(figsize=(12, 6))
plt.plot(test_dates, predictions[:, 3], label='Predicted Price')
plt.plot(test_dates, y_test[:, 3], label='Actual Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.xticks(rotation=45)
plt.legend()
st.pyplot()

