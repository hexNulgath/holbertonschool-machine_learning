#!/usr/bin/env python3
"""
returns all students sorted by average score
"""


def top_students(mongo_collection):
    """
    mongo_collection will be the pymongo collection object
    The top must be ordered
    The average score must be part of each item
        returns with key = averageScore
    """
    students = list(mongo_collection.find())
    for student in students:
        scores = [topic.get('score', 0) for topic in student.get('topics', [])]
        if scores:
            average_score = sum(scores) / len(scores)
        else:
            average_score = 0
        student['averageScore'] = average_score
    sorted_students = sorted(students, key=lambda x: x['averageScore'], reverse=True)
    return sorted_students
