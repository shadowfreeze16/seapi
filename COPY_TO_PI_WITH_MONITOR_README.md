# Raspberry Pi Setup & Deployment Guide (Normal Monitor Method)

This guide is for setting up the Driver Monitoring System on your Raspberry Pi 4 using a standard **HDMI monitor (or TV), keyboard, and mouse**, and transferring the project via a regular **USB Pen Drive (Flash Drive)**.

No complex ext4 card readers or SSH setup required!

---

## What You Need
1. Raspberry Pi 4 with its micro-SD card already inserted.
2. Micro-HDMI to HDMI cable (connected to a monitor or TV).
3. USB Keyboard and Mouse connected to the Pi.
4. Official Raspberry Pi Power Supply (USB-C).
5. A regular **USB Flash Drive (Pen Drive)**.

---

## Step 1: Copy `driver_monitor` to a USB Pen Drive (On Laptop)

1. Plug a standard USB pen drive into your Windows laptop.
2. Open Windows File Explorer and navigate to:
   `C:\Users\Wajida\Downloads\RP`
3. Copy the clean **`driver_monitor`** folder.
4. Paste it onto your USB pen drive.
5. Safely eject the USB pen drive from your laptop.

*(Note: `driver_monitor` has already been cleaned — no heavy `venv` or cache files are inside, total size is only ~68 MB).*

---

## Step 2: Boot the Raspberry Pi with Monitor & Keyboard

1. Ensure the micro-SD card is inserted into the Pi.
2. Plug in the HDMI monitor, USB keyboard, and USB mouse.
3. Plug in the power cable to boot the Pi.
4. Wait for the **Raspberry Pi OS Desktop** to appear on screen.
5. Connect the Pi to your Wi-Fi (click the Wi-Fi icon at the top right of the screen and select your network).

---

## Step 3: Plug USB Drive into Pi and Copy Files

1. Plug your USB pen drive into one of the Pi's blue (USB 3.0) or black (USB 2.0) ports.
2. A window will pop up on the Pi desktop showing the USB drive contents.
3. Open the **Terminal** application (click the black terminal icon at the top-left menu).
4. Run these commands to copy the folder into your home directory:

```bash
# 1. Check your USB drive location (usually /media/nightmare/DRIVENAME)
ls /media/nightmare/

# 2. Backup old folder if it exists
mv ~/driver_monitor ~/driver_monitor_backup 2>/dev/null || true

# 3. Copy the driver_monitor folder from your USB drive to your home directory:
# (Replace 'USB_NAME' with the actual name of your pen drive shown by ls above)
cp -r /media/nightmare/*/driver_monitor ~/driver_monitor

# 4. Verify the folder copied successfully
ls -la ~/driver_monitor
```

---

## Step 4: Setup & Run on the Raspberry Pi

In the same Pi terminal, run:

```bash
# 1. Enter the project directory
cd ~/driver_monitor

# 2. Grant execution permissions
chmod +x start_dms.sh driver_monitor.py tools/*.py

# 3. Activate the Pi virtual environment
source venv/bin/activate

# 4. Verify all 4 ONNX models are present and healthy
python3 tools/download_models.py --check
```
*(All 4 models should display `[ok]`)*.

```bash
# 5. Run the guided camera framing setup
python3 driver_monitor.py --setup-camera
```
*(Look straight ahead in normal driving posture until calibration succeeds).*

```bash
# 6. Start the Driver Monitoring System:
# (Replace 192.168.0.109 with your laptop's current IP address)
python3 driver_monitor.py --fast --dashboard-url http://192.168.0.109:5000
```

---

## Step 5: How to Verify It Worked

1. **On the Pi Monitor**:
   - The camera window opens displaying the driver's face, eye probability scores, and current state.
   - Terminal prints status updates every 5 seconds:
     ```text
     [STATUS 5s] FPS=12.8 | Left=0.85 Right=0.82 | PERCLOS=0.00 Closure= 0.0s | State=NORMAL     Safety=GREEN
     ```
2. **On the Laptop Fleet Dashboard (`http://127.0.0.1:5000`)**:
   - The vehicle card updates live.
   - Normal attention shows **GREEN** (`ATTENTIVE`).
   - Closing eyes for 1.5 seconds triggers **YELLOW** (`WARNING`), emitting a `DROWSY` event.
   - Closing eyes for 3.0+ seconds triggers **RED** (`CRITICAL`) with photo capture.
   - Alert history logs real `PERCLOS` and `Closure` times with photo thumbnails.
