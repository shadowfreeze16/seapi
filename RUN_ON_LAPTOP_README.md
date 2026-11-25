# 💻 Running the Driver Monitoring System (DMS) Directly on a Laptop

This guide explains how to run both the **Driver Monitoring System (AI Camera)** and the **Fleet Management Web Dashboard** completely on a laptop (Windows, Mac, or Linux) using your **built-in webcam**.

**No Raspberry Pi, no SD card, no external monitor, and no hardware wires required!**

---

## 📁 1. Files & Folders Needed

Make sure the following three folders are in your workspace directory:
```
RP/
├── driver_monitor/          # Core AI models & computer vision logic
├── laptop-camera-test/      # Laptop webcam test runner (test_laptop_cam.py)
├── laptop-fleet/            # Real-time web fleet dashboard (app.py)
└── RUN_ON_LAPTOP_README.md  # This instruction guide
```

---

## ⚙️ 2. Prerequisites & Library Installation

Ensure you have **Python 3.10** or **3.11** installed.

Open PowerShell, Command Prompt, or Terminal and run this single command to install all required dependencies:

```bash
pip install opencv-python onnxruntime requests flask numpy
```

*(Note: No Pi-specific packages like `picamera2` or `gpiozero` are needed).*

---

## 🚀 3. Step-by-Step Running Instructions

You will need **two terminal windows**:
1. **Terminal 1:** Runs the Fleet Dashboard web server.
2. **Terminal 2:** Runs the Webcam Driver Monitoring AI.

---

### Step A: Start the Fleet Dashboard (Terminal 1)

1. Open your first Terminal window.
2. Navigate to `laptop-fleet`:
   ```bash
   cd laptop-fleet
   ```
3. Start the dashboard server:
   ```bash
   python app.py
   ```
4. You will see:
   ```
   * Serving Flask app 'app'
   * Running on all addresses (0.0.0.0)
   * Running on http://127.0.0.1:5000
   ```
5. **Open your web browser** (Chrome / Edge / Firefox) and visit:  
   👉 **`http://127.0.0.1:5000`**

You will see the **Fleet Monitoring Dashboard** with 0 active vehicles.

---

### Step B: Start the Webcam Driver Monitor (Terminal 2)

1. Open a **second** Terminal window.
2. Navigate to `laptop-camera-test`:
   ```bash
   cd laptop-camera-test
   ```
3. Start the laptop camera test harness:
   ```bash
   python test_laptop_cam.py --dashboard-url http://127.0.0.1:5000
   ```
4. A window titled **`DMS Laptop Camera Test`** will pop up showing your live camera feed!

---

## 🎮 4. Camera Window HUD & Controls

### Live Overlay (HUD)
The webcam window displays:
- **Top Banner**: Safety Status (`GREEN: ACTIVE`, `YELLOW: DROWSY`, or `RED: CRITICAL`).
- **EAR**: Eye Aspect Ratio (openness score).
- **PERCLOS**: Percentage of eye closure over the sliding window.
- **Closure**: Duration in seconds of continuous eye closure.
- **Gaze / Head Pose**: Yaw, Pitch, Roll angles.

### Keyboard Shortcuts (while camera window is focused):
- **`Q`** or **`ESC`**: Quit the webcam monitor.
- **`C`**: Recalibrate neutral forward-facing head pose (sit straight, look forward, and press `C`).

---

## 🧪 5. How to Test Safety Detections

| Test Case | How to Test | Expected Behavior |
| :--- | :--- | :--- |
| **Normal Driving** | Look forward with eyes open. | **GREEN (ACTIVE)** on HUD and Dashboard. |
| **Drowsiness Warning** | Close eyes or look down for **1.5 seconds**. | **YELLOW (DROWSY)** alert triggered. Alert logged on Dashboard. |
| **Microsleep / Critical** | Keep eyes closed for **3.0 seconds**. | **RED (CRITICAL)** alarm triggered. Event snapshot photo sent to Dashboard. |
| **Distraction** | Turn head left or right away from the screen for **2+ seconds**. | **DISTRACTED** alert triggered. |

---

## 📊 6. Viewing Results on the Dashboard

In your browser at **`http://127.0.0.1:5000`**:
1. You will see vehicle **`VEH-LAPTOP-TEST`** connected with live status updates (EAR, PERCLOS, Attention Level).
2. Any triggered alert (Drowsy, Microsleep) automatically appears in the **Alert History** table.
3. Click the **View Photo** link next to an alert to see the exact snapshot captured by the AI during the event!

---

## ❓ 7. Troubleshooting

- **Webcam doesn't turn on:**  
  If you have an external webcam or multiple cameras, specify camera index 1:  
  ```bash
  python test_laptop_cam.py --cam 1 --dashboard-url http://127.0.0.1:5000
  ```
- **Port 5000 already in use:**  
  You can run the dashboard on another port (e.g., 5050):  
  Edit `laptop-fleet/app.py` bottom line `port=5050`, or pass `--dashboard-url http://127.0.0.1:5050` to the camera script.
- **Face not detected:**  
  Make sure your room has decent lighting and sit ~50–80 cm away from the screen.
