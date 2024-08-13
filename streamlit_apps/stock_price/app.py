import yfinance as yf
import streamlit as st
import pandas as pd

st.write(
    """
# Simple stock price app

Shown are the stock closing price and volume of Google!
"""
)

# ticker symbol to use
tickerSymbol = "GOOGL"
# Get data for the ticker
tickerData = yf.Ticker(tickerSymbol)
# Get the historical prices for the ticker
tickerDf = tickerData.history(period="1d", start="2010-5-31", end="2020-5-31")
# Open  High  Low  Close  Volume  Dividends  Stock  solits
st.line_chart(tickerDf.Close)
st.line_chart(tickerDf.Volume)
