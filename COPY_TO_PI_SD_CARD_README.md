# Raspberry Pi SD Card Copy & Deployment Guide

Step-by-step checklist for copying the fixed **`driver_monitor`** folder from your Windows laptop to the Raspberry Pi 4 SD card, and launching the Driver Monitoring System (DMS).

---

## 1. What to Copy and What NOT to Copy

### Source Location on Laptop:
`C:\Users\Wajida\Downloads\RP\driver_monitor`

### Folders to Exclude Completely:
- [ ] **DO NOT** copy `laptop-fleet/`
- [ ] **DO NOT** copy `laptop-camera-test/`
- [ ] **DO NOT** copy `driver-monitoring-system/` (backup/old reference)

### Inside `driver_monitor/`, Exclude:
- [ ] **DO NOT** copy `venv/` (The Pi already has its own Linux virtual environment)
- [ ] **DO NOT** copy `__pycache__/` or any `*.pyc` files
- [ ] **DO NOT** copy `.git/` (if present)

### ONLY Copy These Folders & Files:
```text
driver_monitor/
├── dms/                                (All 20 Python modules)
├── models/                             (All 4 required ONNX files):
│   ├── face_detection_yunet_2026may.onnx
│   ├── headpose_mobilenetv3_small.onnx
│   ├── eyes_ocec_s.onnx
│   └── face_recognition_sface_2021dec.onnx
├── tools/                              (benchmark_models.py, download_models.py, self_test.py, check_stream.py)
├── tests/                              (Unit tests)
├── driver_monitor.py                   (Main application)
├── start_dms.sh                        (Launcher script)
├── requirements.txt                    (Package list)
└── phone_camera.json                   (Camera config)
```

---

## 2. Safe Copy to the SD Card (Windows)

The Raspberry Pi SD card contains two separate partitions:
1. **`bootfs`** (FAT32): Visible in Windows Explorer as a drive (e.g., `D:` or `E:`).
2. **`rootfs`** (ext4 Linux): **Invisible/unwritable by Windows Explorer natively.**

> [!WARNING]
> - **DO NOT** format the SD card when Windows prompts: *"You need to format the disk before you can use it"*. Click **Cancel**.
> - **DO NOT** copy files into the `boot` partition (e.g. `bootfs`). The code must go into the Linux `rootfs` partition.
> - **DO NOT** pull out the SD card without safely ejecting.

### Step 2.1: Make a Backup First
Before touching the SD card files:
1. Open Windows Explorer.
2. In `C:\Users\Wajida\Downloads\RP`, make a safety copy of your working folder (e.g. copy `driver_monitor` to `driver_monitor_backup`).

### Step 2.2: Use DiskInternals Linux Reader / Linux Writer
Because Windows cannot natively write to Linux ext4 partitions, use **DiskInternals Linux Writer**:
1. Download and install **DiskInternals Linux Writer** (or Linux Reader Pro).
2. Right-click the application icon and select **Run as Administrator**.
3. Insert your Pi micro-SD card into the laptop card reader.
4. In Linux Writer, locate and open the **Linux rootfs partition** (ext4).
5. Navigate into the user home folder:
   ```text
   home > nightmare
   ```
6. Inside `/home/nightmare/`:
   - Rename the existing `driver_monitor` folder to `driver_monitor_old` (for safety).
   - Copy the clean `driver_monitor` folder from your laptop:
     `C:\Users\Wajida\Downloads\RP\driver_monitor`
   - Paste it into `/home/nightmare/`.
   - Ensure the new folder path on the SD card is exactly:
     `/home/nightmare/driver_monitor/`
7. Close Linux Writer.
8. Right-click the SD card drive in Windows System Tray / Explorer and click **Safely Remove / Eject**. Remove the card.

---

## 3. Setup on the Raspberry Pi

### Step 3.1: Start the Laptop Fleet Dashboard FIRST
On your Windows laptop, make sure the Fleet Dashboard server is running so the Pi can connect immediately:
1. Open PowerShell on the laptop:
   ```powershell
   cd C:\Users\Wajida\Downloads\RP\laptop-fleet
   python app.py
   ```
2. Confirm it says:
   ```text
   Listening on http://0.0.0.0:5000
   ```
3. Open `http://127.0.0.1:5000` in your laptop browser.

---

### Step 3.2: Insert Card into Pi & Run

1. Insert the micro-SD card firmly into the Raspberry Pi 4.
2. Connect power, camera, and display (or connect via SSH).
3. Open a terminal on the Pi (user `nightmare`):

```bash
# 1. Navigate to the project directory
cd ~/driver_monitor

# 2. Make scripts executable
chmod +x start_dms.sh driver_monitor.py tools/*.py

# 3. Activate the Pi virtual environment
source venv/bin/activate

# 4. Verify the 4 ONNX models are present and valid
python3 tools/download_models.py --check
```
*(You should see `[ok]` next to all 4 models).*

```bash
# 5. Run initial camera framing and neutral pose setup
python3 driver_monitor.py --setup-camera
```

```bash
# 6. Start the Driver Monitoring System with laptop dashboard streaming
python3 driver_monitor.py --fast --dashboard-url http://192.168.0.109:5000
```

*(Note: If your laptop Wi-Fi IP address changes, replace `192.168.0.109` with your current laptop IP from `ipconfig`).*

---

## 4. Verification Checklist

### On the Raspberry Pi Terminal:
- [ ] Confirms camera connected (Picamera2 or V4L2).
- [ ] Prints status line every 5 seconds:
  ```text
  [STATUS 5s] FPS=12.5 | Left=0.85 Right=0.82 | PERCLOS=0.00 Closure= 0.0s | State=NORMAL     Safety=GREEN
  ```
- [ ] FPS stays above 10 FPS with `--fast` mode enabled.

### On the Laptop Fleet Dashboard (`http://127.0.0.1:5000`):
- [ ] A new vehicle card appears with the Pi session ID.
- [ ] **Active Driver**: Card shows **GREEN** (`ATTENTIVE`).
- [ ] **Eyes Closed 1.5s**: Emits `DROWSY`, card turns **YELLOW** (`WARNING`), event logged in Alert History.
- [ ] **Eyes Closed 3.0s+**: Card turns **RED** (`SERIOUS`), critical alert logged with live snapshot photo.
- [ ] Alert History table shows exact non-zero `PERCLOS` and `Closure` values with photo thumbnails.

---

## 5. Troubleshooting

- **Pi does not boot (solid red light, no green flicker)**:
  - Check that the micro-SD card is inserted fully into the slot.
  - Confirm you did not accidentally overwrite or format the `bootfs` (FAT32) partition.
- **`Permission denied` when running scripts on the Pi**:
  - Run: `chmod +x ~/driver_monitor/start_dms.sh ~/driver_monitor/driver_monitor.py`
- **`ModuleNotFoundError: No module named 'cv2'` on Pi**:
  - Ensure you ran `source venv/bin/activate` before launching Python.
- **Dashboard shows OFFLINE on laptop**:
  - Verify laptop and Pi are on the same Wi-Fi network.
  - Verify laptop IP with `ipconfig` in Windows Command Prompt and update `--dashboard-url http://<YOUR_LAPTOP_IP>:5000`.
