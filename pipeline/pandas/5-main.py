#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
slice = __import__('5-slice').slice

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = slice(df)

print(df.tail())
$ ./5-main.py
            High      Low    Close  Volume_(BTC)
2099460  4020.08  4020.07  4020.08      4.704989
2099520  4020.94  4020.93  4020.94      2.111411
2099580  4020.00  4019.01  4020.00      4.637035
2099640  4017.00  4016.99  4017.00      2.362372
2099700  4014.78  4013.50  4014.72      1.291557
$

