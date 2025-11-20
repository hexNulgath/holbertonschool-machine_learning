#!/usr/bin/env python3
"""
2-user_location module
"""
import requests
import sys
from datetime import datetime

if __name__ == '__main__':
    url = sys.argv[1] if len(sys.argv) > 1 else None
    if url:
        response = requests.get(url)
        if response.status_code == 403:
            reset_timestamp = int(response.headers.get('X-RateLimit-Reset', 0))
            now_timestamp = int(datetime.now().timestamp())

            seconds_left = reset_timestamp - now_timestamp
            minutes_left = int(seconds_left / 60)
            print(f"Reset in {minutes_left} min")
        elif response.status_code == 404:
            print("Not found")
        else:
            print(response.json().get("location"))
