#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
hierarchy = __import__('12-hierarchy').hierarchy

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df = hierarchy(df1, df2)

print(df)
$ ./12-main.py
                       Open   High     Low   Close  Volume_(BTC)  Volume_(Currency)  Weighted_Price
Timestamp
1417411980 bitstamp  379.99  380.0  379.99  380.00      3.901265        1482.461708      379.995162
           coinbase  300.00  300.0  300.00  300.00      0.010000           3.000000      300.000000
1417412040 bitstamp  380.00  380.0  380.00  380.00     35.249895       13394.959997      380.000000
           coinbase     NaN    NaN     NaN     NaN           NaN                NaN             NaN
1417412100 bitstamp  380.00  380.0  380.00  380.00      3.712000        1410.560000      380.000000
...                     ...    ...     ...     ...           ...                ...             ...
1417417860 coinbase     NaN    NaN     NaN     NaN           NaN                NaN             NaN
1417417920 bitstamp  380.09  380.1  380.09  380.10      1.503000         571.285290      380.096667
           coinbase     NaN    NaN     NaN     NaN           NaN                NaN             NaN
1417417980 bitstamp  380.10  380.1  378.85  378.85     26.599796       10079.364182      378.926376
           coinbase     NaN    NaN     NaN     NaN           NaN                NaN             NaN

[202 rows x 7 columns]
$

