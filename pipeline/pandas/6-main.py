#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
flip_switch = __import__('6-flip_switch').flip_switch

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = flip_switch(df)

print(df.tail(8))
$ ./6-main.py
                        2099759       2099758       2099757  ...       2             1             0
Timestamp          1.546899e+09  1.546899e+09  1.546899e+09  ...  1.417412e+09  1.417412e+09  1.417412e+09
Open               4.005510e+03  4.006010e+03  4.006010e+03  ...           NaN           NaN  3.000000e+02
High               4.006010e+03  4.006010e+03  4.006010e+03  ...           NaN           NaN  3.000000e+02
Low                4.005510e+03  4.005500e+03  4.006000e+03  ...           NaN           NaN  3.000000e+02
Close              4.005990e+03  4.005500e+03  4.006010e+03  ...           NaN           NaN  3.000000e+02
Volume_(BTC)       1.752778e+00  2.699700e+00  1.192123e+00  ...           NaN           NaN  1.000000e-02
Volume_(Currency)  7.021184e+03  1.081424e+04  4.775647e+03  ...           NaN           NaN  3.000000e+00
Weighted_Price     4.005746e+03  4.005720e+03  4.006004e+03  ...           NaN           NaN  3.000000e+02

[8 rows x 2099760 columns]
$

