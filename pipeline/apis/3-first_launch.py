#!/usr/bin/env python3
"""3-first_launch.py"""
import requests


if __name__ == '__main__':
    response = requests.get(
        'https://api.spacexdata.com/v4/launches/upcoming').json()
    launches = sorted(response, key=lambda x: x['date_utc'])
    response = launches[0]
    rocket_url = (
        f"https://api.spacexdata.com/v4/rockets/{response['rocket']}"
    )
    rocket = requests.get(rocket_url).json()
    launchpad_url = (
        f"https://api.spacexdata.com/v4/launchpads/{response['launchpad']}"
    )
    launchpad = requests.get(launchpad_url).json()
    launch_time = response['date_local']
    print(
        f"{response['name']} ({launch_time}) {rocket['name']} - "
        f"{launchpad['name']} ({launchpad['locality']})"
    )
