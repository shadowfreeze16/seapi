"""BASIC Laptop Dashboard - Driver Monitoring System v1.
One vehicle for now. Fleet (multiple vehicles) is a later upgrade.
Run: pip install Flask requests
     python app.py
Open: http://127.0.0.1:5000
"""
import json
import os
import time
from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
ALERTS_DIR = os.path.join(BASE, "alerts")
ALERTS_JSON = os.path.join(BASE, "alerts.json")
os.makedirs(ALERTS_DIR, exist_ok=True)
app = Flask(__name__)
vehicles = {}  # vehicle_id -> last status


def load_alerts():
    if not os.path.exists(ALERTS_JSON):
        return []
    try:
        return json.load(open(ALERTS_JSON))
    except Exception:
        return []


def save_alerts(a):
    json.dump(a[-200:], open(ALERTS_JSON, "w"), indent=2)


@app.route("/")
def home():
    return render_template("dashboard.html", vehicles=vehicles, alerts=load_alerts()[::-1][:20])


@app.route("/api/status", methods=["POST"])
def status():
    d = request.get_json() or {}
    d["received_at"] = datetime.now().strftime("%H:%M:%S")
    vehicles[d.get("vehicle_id", "VEH-001")] = d
    return jsonify({"ok": True})


@app.route("/api/alert", methods=["POST"])
def alert():
    data = request.form.to_dict() if not request.is_json else request.get_json()
    img = request.files.get("image")
    name = None
    if img and img.filename:
        name = f"{data.get('level','ALERT').lower()}_{int(time.time())}.jpg"
        img.save(os.path.join(ALERTS_DIR, name))
    a = {"id": int(time.time()*1000), "vehicle_id": data.get("vehicle_id", "VEH-001"),
         "driver": data.get("driver", ""), "level": data.get("level", "YELLOW"),
         "message": data.get("message", ""), "timestamp": data.get("timestamp", ""),
         "image": name, "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    alerts = load_alerts()
    alerts.append(a)
    save_alerts(alerts)
    vehicles[a["vehicle_id"]] = {**a, "level": a["level"]}
    print(f"[{a['received_at']}] {a['vehicle_id']} {a['level']}: {a['message']}")
    return jsonify({"ok": True}), 201


@app.route("/api/test", methods=["POST"])
def test():
    for level, msg in [("GREEN", "Driver active and safe"), ("YELLOW", "Eyes closed - stay alert"), ("RED", "Eyes closed over 2 seconds - possible microsleep")]:
        a = {"id": int(time.time()*1000), "vehicle_id": "VEH-001", "driver": "Driver 01",
             "level": level, "message": msg, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             "image": None, "received_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        alerts = load_alerts(); alerts.append(a); save_alerts(alerts)
        vehicles["VEH-001"] = a
    return jsonify({"ok": True})


@app.route("/alerts/<path:fn>")
def serve(fn):
    return send_from_directory(ALERTS_DIR, fn)


if __name__ == "__main__":
    print("DMS BASIC dashboard at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
