#!/usr/bin/env python3
"""
The column Weighted_Price should be removed
Rename the column Timestamp to Date
Convert the timestamp values to date values
Index the data frame on Date
Missing values in Close should be set to the previous row value
Missing values in High, Low, Open should be set to the same row’s Close value
Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
Plot the data from 2017 and beyond at daily intervals and
    group the values of the same day such that:

    High: max
    Low: min
    Open: mean
    Close: mean
    Volume(BTC): sum
    Volume(Currency): sum
Return the transformed pd.DataFrame before plotting.
"""
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date')
df['Close'] = df['Close'].fillna(method='ffill')
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
df = df.loc['2017':]
print(df)
df = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Close', color='red')
plt.plot(df.index, df['Open'], label='Open', color='green')
plt.plot(df.index, df['High'], label='High', color='blue')
plt.plot(df.index, df['Low'], label='Low', color='orange')
plt.plot(df.index, df['Volume_(BTC)'], label='Volume_(BTC)', color='purple')
plt.plot(
    df.index, df['Volume_(Currency)'],
    label='Volume_(Currency)',
    color='brown')
plt.xlabel('Date')
plt.legend()
plt.show()
