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
            ti = int(response.headers.get('X-RateLimit-Reset', 0))
            ti = datetime.fromtimestamp(ti).minute
            print(f"Reset in {ti} min")
        elif response.status_code == 404:
            print("Not found")
        else:
            print(response.json().get("location"))
