import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Page Title
st.title("📈 Stock Price Prediction App using LSTM")

# Sidebar - File uploader
st.sidebar.title("Upload Your Stock CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Display raw data
    st.subheader("Raw Data")
    st.write(df.head())

    # Ensure only 'Date' and 'Close' columns are used
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Plot closing prices
    st.subheader("📉 Closing Price Chart")
    st.line_chart(df['Close'])

    # Normalize close prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])

    # Prepare data for LSTM
    sequence_length = 60
    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split into train and test sets (optional, here we use all for training)
    # X_train, X_test = X, X
    # y_train, y_test = y, y

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    # Save model
    model.save("lstm_stock_model.h5")

    # Predict for the entire dataset
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y.reshape(-1, 1))

    # Plot actual vs predicted
    st.subheader("📊 Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(actual_prices, label='Actual')
    ax.plot(predictions, label='Predicted')
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.set_title("Actual vs Predicted Stock Prices")
    ax.legend()
    st.pyplot(fig)

    # Predict the next day price
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, 60, 1))
    next_day_scaled = model.predict(last_60_days)
    next_day_price = scaler.inverse_transform(next_day_scaled)

    st.subheader("📅 Next Day Predicted Price")
    st.success(f"Predicted Close Price for Next Day: ₹{next_day_price[0][0]:.2f}")
