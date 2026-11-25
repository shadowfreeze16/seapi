"""Test basic dashboard without Pi: python test_basic.py"""
import requests
url = "http://127.0.0.1:5000"
for level, msg in [("GREEN", "Driver active and safe"), ("YELLOW", "Eyes closed - stay alert"), ("RED", "Eyes closed over 2 seconds - possible microsleep")]:
    r = requests.post(url + "/api/alert", data={"vehicle_id": "VEH-001", "driver": "Driver 01", "level": level, "message": msg}, timeout=5)
    print(level, r.status_code)
print("Open http://127.0.0.1:5000")
