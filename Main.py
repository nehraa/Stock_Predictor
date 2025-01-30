import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Data.csv')

# Convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Use the Close price for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM model
def create_sequences(data, seq_length=12):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

# Prepare the data with a sequence length of 12 months
X, y = create_sequences(data_scaled, seq_length=12)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape data for LSTM (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Output the predicted price

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict the stock price for the next month
predictions = model.predict(X_test)
predictions_rescaled = scaler.inverse_transform(predictions)

# Plot the results
plt.figure(figsize=(12,6))

# Plot the true prices (actual values from y_test)
plt.plot(df.index[len(df) - len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='True Price')

# Plot the predicted prices (model's prediction)
plt.plot(df.index[len(df) - len(y_test):], predictions_rescaled, color='red', label='Predicted Price')

# Customize the plot
plt.title('Tesla Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Show the plot
plt.show()

# Optional: Save the predictions to a CSV file
predicted_df = pd.DataFrame({
    'Date': df.index[len(df) - len(y_test):],
    'True Price': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
    'Predicted Price': predictions_rescaled.flatten()
})
predicted_df.to_csv('tesla_stock_predictions.csv', index=False)
