#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
print(df1.head())
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')
print(df2.tail())
$ ./2-main.py
    Timestamp   Open   High    Low  Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
0  1417411980  300.0  300.0  300.0  300.0          0.01                3.0           300.0
1  1417412040    NaN    NaN    NaN    NaN           NaN                NaN             NaN
2  1417412100    NaN    NaN    NaN    NaN           NaN                NaN             NaN
3  1417412160    NaN    NaN    NaN    NaN           NaN                NaN             NaN
4  1417412220    NaN    NaN    NaN    NaN           NaN                NaN             NaN
          Timestamp     Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
4363452  1587513360  6847.97  6856.35  6847.97  6856.35      0.125174         858.128697     6855.498790
4363453  1587513420  6850.23  6856.13  6850.23  6850.89      1.224777        8396.781459     6855.763449
4363454  1587513480  6846.50  6857.45  6846.02  6857.45      7.089168       48533.089069     6846.090966
4363455  1587513540  6854.18  6854.98  6854.18  6854.98      0.012231          83.831604     6854.195090
4363456  1587513600  6850.60  6850.60  6850.60  6850.60      0.014436          98.896906     6850.600000
$

