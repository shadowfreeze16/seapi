# SEAPI - Driver Monitoring System (DMS) & Fleet Management

A real-time edge Driver Monitoring System (DMS) built for edge deployment (Raspberry Pi 5) and cross-platform testing (Laptops / Workstations). It features deep-learning face detection, head pose estimation, eye state analysis (EAR / PERCLOS), distraction tracking, and a live web-based fleet monitoring dashboard.

---

## 🚀 Quick Start Guides

- 💻 **[Running on a Laptop (Built-in Webcam)](RUN_ON_LAPTOP_README.md)**  
  Test the entire AI vision pipeline and fleet dashboard directly on Windows, macOS, or Linux without any Raspberry Pi hardware.
- 🍓 **[Running on Raspberry Pi with Monitor & USB](COPY_TO_PI_WITH_MONITOR_README.md)**  
  How to copy and run the application on Raspberry Pi OS using a standard USB flash drive.
- 💾 **[Raspberry Pi Direct SD Card Guide](COPY_TO_PI_SD_CARD_README.md)**  
  Reference guide for flashing and headless configuration.

---

## 📦 Project Structure

```
RP/
├── driver_monitor/          # Core AI models, ONNX inference & state machine
│   ├── dms/                 # Eye analysis, safety scoring, head pose, config
│   ├── models/              # Pretrained ONNX models (YuNet, MobileNetV3, OCEC)
│   ├── tests/               # Unit and regression test suites
│   └── requirements.txt     # Python dependencies
├── laptop-camera-test/      # Laptop webcam runner (cv2.VideoCapture(0))
│   └── test_laptop_cam.py   # Test harness with live HUD and fleet sync
├── laptop-fleet/            # Real-time Web Dashboard (Flask + HTML)
│   ├── app.py               # Telemetry API server (ports 5000)
│   ├── templates/           # Real-time web UI
│   └── snapshots/           # Captured incident photo storage
└── README.md
```

---

## ⚡ Core Capabilities

- **Face Detection:** YuNet ONNX model for high-speed edge face localization.
- **Head Pose Estimation:** 6-DoF yaw, pitch, roll tracking for distraction detection.
- **Eye & Drowsiness Analysis:** Closed/open estimation with calibrated PERCLOS and continuous closure timers.
- **Fleet Sync:** HTTP telemetry and incident snapshot uploads to the central dashboard.
