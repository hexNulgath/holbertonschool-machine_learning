#!/usr/bin/env python3
"""
provides some stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:27017/")
    db = client.logs
    collection = db.nginx
    print("{} logs".format(collection.count_documents({})))
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))
    status = collection.count_documents({"path": "/status"})
    print("{} status check".format(status))
    ips = collection.distinct("ip")
    counts = [collection.count_documents({"ip": ip}) for ip in ips]
    print("IPs:")
    counted_ips = sorted(zip(ips, counts), key=lambda x: x[1], reverse=True)
    for ip, count in counted_ips[:10]:
        print("\t{}: {}".format(ip, count))