import streamlit as st
import pandas as pd
import yfinance as yf
import altair as alt

# Function to fetch stock data
def load_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data

# Function to create interactive line chart
def plot_line_chart(data, y_column, title, color):
    chart = alt.Chart(data).mark_line().encode(
        x='Date:T',
        y=f'{y_column}:Q',
        color=alt.value(color),
        tooltip=['Date', y_column]
    ).properties(
        title=title
    ).interactive()
    return chart

# Streamlit app
st.title("Stock Data Dashboard")

# Get user input for stock symbol, start date, and end date
stock_symbol = st.sidebar.selectbox("Select a stock symbol:", ['AAPL', 'GOOG', 'MSFT', 'AMZN'])
start_date = st.sidebar.date_input("Select the start date:", pd.to_datetime('2012-01-01'))
end_date = st.sidebar.date_input("Select the end date:", pd.to_datetime('2023-11-30'))

# Load stock data
stock_data = load_data(stock_symbol, start_date, end_date)

# Display stock data
st.subheader(f"Displaying stock data for {stock_symbol} from {start_date} to {end_date}")
st.write(stock_data)

# Plot close price
st.subheader("Close Price Over Time")
close_price_chart = plot_line_chart(stock_data.reset_index(), 'Close', 'Close Price Over Time', 'blue')
st.altair_chart(close_price_chart, use_container_width=True)

# Plot volume
st.subheader("Volume Over Time")
volume_chart = plot_line_chart(stock_data.reset_index(), 'Volume', 'Volume Over Time', 'green')
st.altair_chart(volume_chart, use_container_width=True)

# Plot high and low prices
st.subheader("High and Low Prices Over Time")
high_low_chart = alt.Chart(stock_data.reset_index()).mark_line(opacity=0.5).encode(
    x='Date:T',
    y='High:Q',
    color=alt.value('red'),
    tooltip=['Date', 'High']
).interactive() + alt.Chart(stock_data.reset_index()).mark_line(opacity=0.5).encode(
    x='Date:T',
    y='Low:Q',
    color=alt.value('orange'),
    tooltip=['Date', 'Low']
).interactive()
st.altair_chart(high_low_chart, use_container_width=True)

# Add a sidebar for user controls
st.sidebar.header("Settings")
moving_average_days = st.sidebar.slider("Select Moving Average Days", 1, 100, 20)

# Plotting closing price with moving averages
st.subheader(f"Closing Price with {moving_average_days}-day Moving Average")
closing_chart = alt.Chart(stock_data.reset_index()).mark_line().encode(
    x='Date:T',
    y='Close:Q',
    color=alt.value('blue'),
    tooltip=['Date', 'Close']
).interactive()

# Calculate moving average
stock_data['Moving_Avg'] = stock_data['Close'].rolling(window=moving_average_days).mean()

# Plot moving average
moving_avg_chart = alt.Chart(stock_data.reset_index()).mark_line().encode(
    x='Date:T',
    y=alt.Y('Moving_Avg:Q', title='Moving Average'),
    color=alt.value('red'),
    tooltip=['Date', 'Moving_Avg']
).interactive()

# Combine charts
st.altair_chart(closing_chart + moving_avg_chart, use_container_width=True)



