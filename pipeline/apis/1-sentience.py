#!/usr/bin/env python3
"""1. Sentient Species"""
import requests


def sentientPlanets():
    """
    returns the list of names of the home planets of all sentient species
    """
    url = "https://swapi-api.hbtn.io/api/species/"
    results = []

    while url:
        response = requests.get(url)
        data = response.json()

        for specie in data.get("results", []):
            classification = specie.get("classification", "").lower()
            designation = specie.get("designation", "").lower()

            if (("sentient" in classification or "sentient" in designation)
                    and specie.get("homeworld")):
                world = requests.get(specie["homeworld"]).json()
                results.append(world["name"])

        url = data.get("next")

    return results
