#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
rename = __import__('3-rename').rename

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = rename(df)

print(df.tail())
$ ./3-main.py
                   Datetime    Close
2099755 2019-01-07 22:02:00  4006.01
2099756 2019-01-07 22:03:00  4006.01
2099757 2019-01-07 22:04:00  4006.01
2099758 2019-01-07 22:05:00  4005.50
2099759 2019-01-07 22:06:00  4005.99
$

