import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

# Reshape data for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Output the predicted price

# Compile and fit the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)

# Predict the next 5 days of stock prices
last_sequence = data_scaled[-12:].reshape(1, 12, 1)  # Last 12 months for prediction

# Store predictions for the next 100 days
predicted_prices = []

for _ in range(100):
    pred_price = model.predict(last_sequence)
    predicted_prices.append(pred_price[0][0])
    last_sequence = np.append(last_sequence[:, 1:, :], pred_price.reshape(1, 1, 1), axis=1)

# Inverse scaling to get the predicted stock prices back to original scale
predicted_prices_rescaled = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Generate future dates for plotting
last_date = df.index[-1]
future_dates = pd.date_range(last_date, periods=101, freq='B')[1:]  # Next 5 trading days

# Plotting
plt.figure(figsize=(12,6))

# Plot the actual stock prices (Close)
plt.plot(df.index, scaler.inverse_transform(data_scaled), color='blue', label='Actual Stock Price')

# Plot the predicted stock prices (next 5 days)
plt.plot(future_dates, predicted_prices_rescaled, color='red', label='Predicted Stock Price')

# Customize the plot
plt.title('Tesla Stock Price Prediction for the Next 100 days')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()

# Show the plot
plt.show()

# Optional: Save the predictions to a CSV file
predicted_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': predicted_prices_rescaled.flatten()
})
predicted_df.to_csv('tesla_stock_predictions_next_week.csv', index=False)
