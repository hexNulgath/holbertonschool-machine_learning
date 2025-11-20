#!/usr/bin/env python3
"""0-passengers.py"""
import requests


def availableShips(passengerCount):
    """
    returns the list of ships that can hold a given number of passengers:
    """
    url = 'https://swapi-api.hbtn.io/api/starships/'
    suitable_ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        ships_data = data.get('results', [])

        for ship in ships_data:
            passengers = ship.get('passengers', 'n/a')

            if passengers.lower() == 'n/a':
                continue

            try:
                passengers_int = int(passengers.replace(',', ''))
            except ValueError:
                continue

            if passengers_int >= passengerCount:
                suitable_ships.append(ship['name'])

        url = data.get('next')

    return suitable_ships
