#!/usr/bin/env python3

from_file = __import__('2-from_file').from_file
array = __import__('4-array').array

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

A = array(df)

print(A)
$ ./4-main.py
[[4009.54 4007.01]
 [4007.01 4003.49]
 [4007.29 4006.57]
 [4006.57 4006.56]
 [4006.57 4006.01]
 [4006.57 4006.01]
 [4006.57 4006.01]
 [4006.01 4006.01]
 [4006.01 4005.5 ]
 [4006.01 4005.99]]
$

