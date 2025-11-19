#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
concat = __import__('11-concat').concat

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df = concat(df1, df2)

print(df)
$ ./11-main.py
                        Open     High      Low    Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
         Timestamp
bitstamp 1325317920     4.39     4.39     4.39     4.39      0.455581           2.000000        4.390000
         1325317980      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318040      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318100      NaN      NaN      NaN      NaN           NaN                NaN             NaN
         1325318160      NaN      NaN      NaN      NaN           NaN                NaN             NaN
...                      ...      ...      ...      ...           ...                ...             ...
coinbase 1546898520  4006.01  4006.57  4006.00  4006.01      3.382954       13553.433078     4006.390309
         1546898580  4006.01  4006.57  4006.00  4006.01      0.902164        3614.083168     4006.017232
         1546898640  4006.01  4006.01  4006.00  4006.01      1.192123        4775.647308     4006.003635
         1546898700  4006.01  4006.01  4005.50  4005.50      2.699700       10814.241898     4005.719991
         1546898760  4005.51  4006.01  4005.51  4005.99      1.752778        7021.183546     4005.745614

[3634661 rows x 7 columns]
$

