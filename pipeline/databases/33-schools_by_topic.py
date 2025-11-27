#!/usr/bin/env python3
"""returns the list of school having a specific topic"""
import pymongo


def schools_by_topic(mongo_collection, topic):
    """returns the list of school having a specific topic
    Args:
        mongo_collection: pymongo collection object
        topic: topic searched
    Returns:
        list of schools having that topic
    """
    return list(mongo_collection.find({"topics": topic}))
