#!/usr/bin/env python3
"""3-first_launch.py"""
import requests


if __name__ == '__main__':
    response = requests.get('https://api.spacexdata.com/v4/launches/upcoming')
    info = response.json()[0]
    rocket_url = (
        f"https://api.spacexdata.com/v4/rockets/{info['rocket']}"
    )
    rocket = requests.get(rocket_url).json()
    launchpad_url = (
        f"https://api.spacexdata.com/v4/launchpads/{info['launchpad']}"
    )
    launchpad = requests.get(launchpad_url).json()
    print(
        f"{info['name']} ({info['date_utc']}) {rocket['name']} - "
        f"{launchpad['name']} ({launchpad['locality']})"
    )
