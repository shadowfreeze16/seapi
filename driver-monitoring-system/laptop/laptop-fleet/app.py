"""Fleet Monitoring Dashboard - Laptop Server.

Accepts status and events from multiple Raspberry Pi Driver Monitoring Systems
over the local network via HTTP POST matching dms.dashboard_client:
  POST /api/session/start
  POST /api/session/end
  POST /api/status
  POST /api/event
  GET  /
  GET  /api/fleet
  GET  /snapshots/<filename>
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from datetime import datetime
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE = os.path.dirname(os.path.abspath(__file__))
SNAPSHOTS_DIR = os.path.join(BASE, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

app = Flask(__name__)

# session_id -> dict
vehicles: dict[str, dict] = {}
# list of event dicts, newest first
events: list[dict] = []
# session_id -> metadata
sessions: dict[str, dict] = {}


def _get_vehicle_view(v: dict, now_epoch: float) -> dict:
    item = dict(v)
    last_epoch = item.get("last_seen_epoch", 0.0)
    elapsed = now_epoch - last_epoch
    item["offline"] = elapsed > 10.0
    item["elapsed_s"] = int(elapsed)
    return item


@app.route("/")
def index():
    now = time.time()
    vehicle_views = [_get_vehicle_view(v, now) for v in vehicles.values()]
    # Sort: active first, then by safety level (RED, YELLOW, GREEN)
    level_prio = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    vehicle_views.sort(
        key=lambda x: (
            1 if x["offline"] else 0,
            level_prio.get(x.get("safety_level", "GREEN"), 3),
            x.get("session_id", ""),
        )
    )
    return render_template(
        "dashboard.html",
        vehicles=vehicle_views,
        events=events[:50],
        total_vehicles=len(vehicle_views),
        active_vehicles=sum(1 for x in vehicle_views if not x["offline"]),
        critical_alerts=sum(
            1 for x in vehicle_views if not x["offline"] and x.get("safety_level") == "RED"
        ),
        server_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


@app.route("/api/fleet", methods=["GET"])
def api_fleet():
    now = time.time()
    vehicle_views = [_get_vehicle_view(v, now) for v in vehicles.values()]
    return jsonify(
        {
            "vehicles": vehicle_views,
            "events": events[:50],
            "total_vehicles": len(vehicle_views),
            "active_vehicles": sum(1 for x in vehicle_views if not x["offline"]),
            "critical_alerts": sum(
                1 for x in vehicle_views if not x["offline"] and x.get("safety_level") == "RED"
            ),
            "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


@app.route("/api/session/start", methods=["POST"])
def session_start():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id") or "session_" + uuid.uuid4().hex[:8]
    sessions[session_id] = {
        "session_id": session_id,
        "started_at": data.get("started_at", datetime.now().isoformat()),
        "active": True,
    }
    now_epoch = time.time()
    if session_id not in vehicles:
        vehicles[session_id] = {
            "session_id": session_id,
            "safety_level": "GREEN",
            "safety_label": "ATTENTIVE",
            "safety_reason": "NORMAL",
            "state": "NORMAL",
            "message": "Session started",
            "driver_name": "Unassigned",
            "identity_state": "OFF",
            "identity_score": None,
            "perclos": 0.0,
            "closure_s": 0.0,
            "blink_rate": 0.0,
            "fps": 0.0,
            "zone": "FORWARD",
            "last_seen_epoch": now_epoch,
            "last_updated": datetime.now().strftime("%H:%M:%S"),
            "snapshot_url": None,
        }
    else:
        vehicles[session_id]["last_seen_epoch"] = now_epoch
        vehicles[session_id]["last_updated"] = datetime.now().strftime("%H:%M:%S")
    return jsonify({"ok": True, "session_id": session_id}), 200


@app.route("/api/session/end", methods=["POST"])
def session_end():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id", "")
    if session_id in sessions:
        sessions[session_id]["active"] = False
        sessions[session_id]["ended_at"] = data.get("ended_at", datetime.now().isoformat())
    if session_id in vehicles:
        vehicles[session_id]["message"] = "Session ended"
    return jsonify({"ok": True, "session_id": session_id}), 200


@app.route("/api/status", methods=["POST"])
def status():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id") or "default"
    now_epoch = time.time()

    prev_snap = vehicles.get(session_id, {}).get("snapshot_url")
    vehicles[session_id] = {
        "session_id": session_id,
        "safety_level": data.get("safety_level", "GREEN"),
        "safety_label": data.get("safety_label", "ATTENTIVE"),
        "safety_reason": data.get("safety_reason", "NORMAL"),
        "state": data.get("state", "NORMAL"),
        "message": data.get("message", "Driver attentive"),
        "zone": data.get("zone", "FORWARD"),
        "driver_name": data.get("driver_name") or "Driver",
        "identity_state": data.get("identity_state", "OFF"),
        "identity_score": data.get("identity_score"),
        "perclos": float(data.get("perclos") or 0.0),
        "closure_s": float(data.get("closure_s") or 0.0),
        "blink_rate": float(data.get("blink_rate") or 0.0),
        "fps": float(data.get("fps") or 0.0),
        "yaw": data.get("yaw"),
        "pitch": data.get("pitch"),
        "roll": data.get("roll"),
        "gps": data.get("gps", {}),
        "last_seen_epoch": now_epoch,
        "last_updated": datetime.now().strftime("%H:%M:%S"),
        "snapshot_url": prev_snap,
    }
    return jsonify({"ok": True}), 200


@app.route("/api/event", methods=["POST"])
def event():
    data = request.get_json(force=True, silent=True) or {}
    session_id = data.get("session_id") or "default"
    event_id = data.get("event_id") or uuid.uuid4().hex

    snap_b64 = data.get("snapshot_jpeg_base64")
    snapshot_url = None
    if snap_b64:
        try:
            img_bytes = base64.b64decode(snap_b64)
            fn = f"{event_id}.jpg"
            fp = os.path.join(SNAPSHOTS_DIR, fn)
            with open(fp, "wb") as f:
                f.write(img_bytes)
            snapshot_url = f"/snapshots/{fn}"
        except Exception as exc:
            print(f"[EVENT] snapshot decode failed: {exc}")

    event_item = {
        "event_id": event_id,
        "session_id": session_id,
        "timestamp": data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": data.get("event", "ALERT"),
        "kind": data.get("kind", data.get("event", "ALERT")),
        "status": data.get("status", "WARNING"),
        "severity": data.get("severity", "warning").lower(),
        "direction": data.get("direction", "FORWARD"),
        "message": data.get("message", ""),
        "safety_level": data.get("safety_level", "YELLOW"),
        "safety_reason": data.get("safety_reason", ""),
        "driver_name": data.get("driver_name") or "Driver",
        "identity_state": data.get("identity_state", "OFF"),
        "perclos": float(data.get("perclos") or 0.0),
        "closure_s": float(data.get("closure_s") or 0.0),
        "fps": float(data.get("fps") or 0.0),
        "snapshot_url": snapshot_url,
    }
    events.insert(0, event_item)
    if len(events) > 500:
        events.pop()

    if session_id in vehicles:
        if snapshot_url:
            vehicles[session_id]["snapshot_url"] = snapshot_url
        if data.get("safety_level"):
            vehicles[session_id]["safety_level"] = data.get("safety_level")

    print(
        f"[FLEET EVENT] {session_id} | {event_item['severity'].upper()} | "
        f"{event_item['event']}: {event_item['message']}"
    )
    return jsonify({"ok": True, "event_id": event_id}), 200


@app.route("/snapshots/<path:filename>")
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOTS_DIR, filename)


if __name__ == "__main__":
    print("=" * 50)
    print("  FLEET MONITORING DASHBOARD (Laptop Server)")
    print("  Listening on http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
