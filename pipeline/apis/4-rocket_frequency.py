#!/usr/bin/env python3
"""4-rocket_frequency.py"""
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()
    id_freq = {}
    for launch in launches:
        rocket_id = launch.get('rocket')
        if rocket_id in id_freq:
            id_freq[rocket_id] += 1
        else:
            id_freq[rocket_id] = 1
    id_freq = dict(sorted(id_freq.items(),
                          key=lambda item: item[1],
                          reverse=True))
    rocket_freq = {}
    for rocket_id, freq in id_freq.items():
        rocket = requests.get(
            f'https://api.spacexdata.com/v4/rockets/{rocket_id}').json()
        rocket_name = rocket.get('name')
        rocket_freq[rocket_name] = freq
        print(rocket_name + ': ' + str(freq))
