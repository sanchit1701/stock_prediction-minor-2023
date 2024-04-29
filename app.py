import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import altair as alt

# Load the saved LSTM model
model = load_model('stock_lstm.h5')

# Function to predict future stock prices
@st.cache
def predict_future(data, num_years, scaler):
    # Create future dates
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_years * 365 + 1)]

    # Get the last 60 days of existing data
    previous_data = data[-60:]

    # Predict future stock prices
    predicted_prices = []
    for i in range(num_years * 365):
        input_data = np.concatenate([previous_data[-60:], np.array(predicted_prices[-60:])])

        # Ensure input_data has the correct shape for MinMaxScaler
        input_data = input_data.reshape(-1, 1)

        # Scale the input data using the same scaler used during training
        input_data = scaler.transform(input_data)

        # Reshape input data for prediction
        input_data = np.reshape(input_data, (1, input_data.shape[0], 1))

        # Make the prediction
        predicted_price = model.predict(input_data)

        # Inverse transform the predicted price
        predicted_price = scaler.inverse_transform(np.array([[predicted_price[0, 0]]]))

        # Add the predicted price to the list
        predicted_prices.append(predicted_price[0, 0])

    return future_dates, predicted_prices

# Set up Streamlit app
st.title("Stock Data Visualization and Prediction")

# Get user input for stock symbol
stock_symbol = st.selectbox("Select a stock symbol:", ['AAPL', 'GOOG', 'MSFT', 'AMZN'])

# Get user input for date range
start_date = st.date_input("Select the start date:", pd.to_datetime('2012-01-01'))
end_date = st.date_input("Select the end date:", pd.to_datetime('2023-11-30'))

# Download stock data using yfinance
yf.pdr_override()
stock_data = yf.download(stock_symbol, start_date, end_date)

# Display stock data
st.write(f"Displaying stock data for {stock_symbol} from {start_date} to {end_date}")
st.write(stock_data)

# Create MinMaxScaler for data normalization
scaler = MinMaxScaler()

# Fit the scaler on training data and transform the data
scaler.fit(stock_data['Close'].values.reshape(-1, 1))

# Get user input for future predictions
num_years = st.slider("Select the number of years for future predictions:", 1, 5)

# Predict future stock prices
future_dates, predicted_prices = predict_future(stock_data['Close'], num_years, scaler)

# Ensure both arrays have the same length
min_length = min(len(future_dates), len(predicted_prices))
future_dates = future_dates[:min_length]
predicted_prices = predicted_prices[:min_length]

# Combine historical and predicted prices into a single DataFrame
combined_df = pd.DataFrame({'Date': future_dates + stock_data.index.tolist(), 
                            'Close': predicted_prices + stock_data['Close'].tolist(),
                            'Type': ['Predicted'] * len(future_dates) + ['Historical'] * len(stock_data)})

# Plot the combined stock prices
st.subheader("Stock Price Predictions")
chart = alt.Chart(combined_df).mark_line().encode(
    x='Date:T',
    y='Close:Q',
    color='Type:N'
)
st.altair_chart(chart, use_container_width=True)

# Display historical closing price chart
st.subheader("Historical Closing Price")
st.line_chart(stock_data['Close'], use_container_width=True)

# Display historical volume chart
st.subheader("Historical Volume")
st.area_chart(stock_data['Volume'], use_container_width=True)
