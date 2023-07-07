import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

data = pd.read_csv('data/STB_Trade.csv')  # Replace 'stock_data.csv' with the path to your data file

# Convert the 'TradingDate' column to datetime
data['TradingDate'] = pd.to_datetime(data['TradingDate'])

# Sort the data by date in ascending order
data = data.sort_values('TradingDate')

# Select the features for modeling
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'percent_change']
data = data[features]

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

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

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and the actual values
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Calculate the root mean squared error (RMSE)
rmse = np.sqrt(np.mean((predictions - y_test)**2))
print(f'RMSE: {rmse}')


# predictions = scaler.inverse_transform(predictions)
# y_test = scaler.inverse_transform(y_test)

# Plot the actual price and the predicted price
plt.figure(figsize=(12, 6))
plt.plot(predictions[:, -1], label='Predicted Price')
plt.plot(y_test[:, -1], label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()
