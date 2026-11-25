# Driver Monitoring System v2

Raspberry Pi 4 driver monitoring with **YuNet face detection**, **6DRepNet head
pose**, **OCEC eye state**, a temporal attention/drowsiness state machine,
event logging, and a live **3D head** that rotates with the driver.

The original v1 project and its full development history are preserved in
[`docs/README_v1_original.md`](docs/README_v1_original.md) and
[`docs/driver_monitor_v1.py`](docs/driver_monitor_v1.py).

---

## 1. What changed from v1

| Area | v1 | v2 |
| --- | --- | --- |
| Head direction | Face-box position in the image | True yaw / pitch / roll from 6DRepNet |
| Eye state | None | OCEC classifier, blink and long-blink detection |
| Drowsiness | None | Closure timing, PERCLOS, blink rate, microsleep |
| Attention logic | Left / centre / right + timer | Debounced zones, timers, priorities and calibration |
| States | Ad-hoc `if` logic | `NORMAL`, `SHORT_GLANCE`, `DISTRACTED`, `DROWSY`, `NO_FACE`, `CRITICAL` |
| 3D display | None | Live 3D head with yaw/pitch/roll, eyelids and top view |
| Log format | 4 CSV columns | 18 CSV columns (first 4 stay v1-compatible) + session JSON |
| Event photos | Yes | Yes, annotated with the HUD (and 3D head when enabled), with storage limits |
| Tooling | Startup script | Model downloader, benchmark, self-test and 100+ unit/integration tests |
| Code | One script | `dms/` package with focused modules |

The upgrade keeps the v1 hardware, camera workflow, event directory and CSV
prefix. Existing four-column logs are rotated automatically instead of being
corrupted.

---

## 2. Features

- **Face detection and tracking** — OpenCV YuNet with five landmarks, largest
  face selection, box/landmark smoothing and short dropout hold.
- **Half-resolution detection** — YuNet runs on a 0.5× frame by default, then
  boxes and landmarks are scaled back. This is roughly 3–4× faster on a Pi 4.
- **Full head pose** — 6DRepNet MobileNetV3-small gives yaw, pitch and roll at
  224×224; a One Euro filter removes jitter without adding noticeable lag.
- **Eye-state analysis** — OCEC-S classifies both eyes in one batch; an EMA,
  hysteresis and temporal logic produce blinks, long blinks, closure duration,
  PERCLOS and blink rate.
- **Attention state machine** — Debounced gaze zones, off-road timing,
  face-loss timing, drowsiness and priority handling.
- **Three-level safety status** — `GREEN` attentive, `YELLOW` caution and
  `RED` serious or persistent warning, with short hysteresis on recovery.
- **Driver identity** — OpenCV SFace embeddings verify the enrolled driver;
  three consecutive mismatches produce a `RED` unauthorized-driver event.
- **Camera framing setup** — Guided overlay checks centering, distance, head
  tilt, lighting and sharpness, then saves the neutral pose automatically.
- **Voice warnings** — Optional English speech via `espeak-ng`.
- **GPS location** — Reads a local `gpsd` so dashboard events carry coordinates.
- **Laptop dashboard sender** — Pushes status/events over LAN and queues them
  to disk while the dashboard is offline.
- **Calibration** — Press `C` while looking forward to save a neutral pose.
- **Live 3D head** — Pure OpenCV + NumPy renderer, no extra 3D dependency.
  Includes a head window, HUD inset, top view, angle bars and animated eyelids.
- **Events** — Annotated JPEGs, a v3 CSV log (safety, identity and GPS),
  session summaries, cooldowns and automatic pruning.
- **Buzzer** — Non-blocking GPIO beeper with a short pulse pattern.
- **Recording** — Annotated video output, at the source frame rate.
- **Headless / file / network operation** — Useful over SSH and for repeatable
  tests; files use the media clock and live streams use the monotonic clock.
  `--video` accepts HTTP(S) MJPEG, RTSP and RTMP URLs, including DroidCam.
- **Phone setup mode** — `--phone` shows the DroidCam installation steps,
  prompts for the phone IP/port with saved defaults, and builds the stream URL.
- **Diagnostics** — `tools/self_test.py`, `tools/benchmark_models.py` and a
  `unittest` suite.

---

## 3. Tested hardware and software

The reference system:

| Component | Value |
| --- | --- |
| Computer | Raspberry Pi 4 Model B, 4 GB RAM |
| Architecture | aarch64 / ARM64 |
| OS | Debian GNU/Linux 13 (Trixie) |
| Python | 3.13.5 |
| OpenCV | 5.0.0 (`opencv-contrib-python`) |
| NumPy | 2.2.4 |
| ONNX Runtime | 1.30.0 |
| Camera | OV5647 CSI camera (Picamera2) |
| Phone camera (optional) | DroidCam IP Cam over Wi-Fi (`http://PHONE_IP:4747/video`) |
| Buzzer | GPIO 18 through `gpiozero` |

Other webcams should work through V4L2. A desktop PC can run the application
for development if the models are present.

### Buzzer safety

Do **not** drive a high-current buzzer directly from a GPIO pin. Use a
transistor/MOSFET, a suitable resistor, and an external supply if needed, with
a common ground. The GPIO should switch the driver, not power the load.

---

## 4. Quick start

```bash
cd ~/driver_monitor
source venv/bin/activate
python tools/download_models.py
python driver_monitor.py
```

or use the launcher, which checks the environment, model hashes and camera
before starting:

```bash
./start_dms.sh
```

Stop with `Q` or `Esc` in the camera window.

Press `?` for the keyboard help overlay.

To use an Android phone as the camera, run:

```bash
./start_dms.sh --phone
```

Optional features (all documented in section 10A):

```bash
./start_dms.sh --phone --setup-camera          # frame the camera first
./start_dms.sh --phone --enroll-driver Alice   # create an identity profile
./start_dms.sh --phone --driver Alice --voice  # enforce identity + speech
./start_dms.sh --phone --gps gpsd \
    --dashboard-url http://192.168.1.20:8080
```

It prints the DroidCam installation steps, asks for the phone IP and port
(press Enter to keep the saved defaults), and then starts monitoring. See
[section 10.3](#103-phone-camera-with-droidcam-android). Running
`./start_dms.sh` with no arguments also offers a source menu.

---

## 5. Installation from scratch

### 5.1 System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 \
                    libgl1 libglib2.0-0 \
                    espeak-ng gpsd gpsd-clients
```

`python3-picamera2` is provided by Raspberry Pi OS and is not a pip package.
On a non-Pi system this step is not needed when using a USB camera or a video
file. `espeak-ng` provides spoken warnings and `gpsd`/`gpsd-clients` provide GPS
coordinates. The DMS runs without them, but those features stay off.

After installing gpsd, connect a receiver and set its device in
`/etc/default/gpsd` (`DEVICES="/dev/ttyUSB0"`, for example), then restart it:

```bash
sudo systemctl restart gpsd.socket gpsd
cgps -s
```

Until a real 2D/3D fix is available, the dashboard shows `GPS: waiting`.

### 5.2 Python environment

```bash
cd ~/driver_monitor
python3 -m venv --system-site-packages venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`--system-site-packages` lets the venv use the apt-installed Picamera2 while
still installing the Python packages into the project environment.

If a wheel is unavailable on your Pi, install OpenCV from apt
(`sudo apt install python3-opencv`) and remove `opencv-contrib-python` from
`requirements.txt`. For ONNX Runtime, the `piwheels` index can help:

```bash
pip install --extra-index-url https://www.piwheels.org/simple -r requirements.txt
```

### 5.3 Models

Model files are not bundled with the source tree.

```bash
python tools/download_models.py          # required models
python tools/download_models.py --check  # verify SHA-256 only
python tools/download_models.py --extra  # optional reference models
python tools/download_models.py --force  # re-download
```

Every download is checked against a pinned SHA-256 before it is moved into
`models/`.

### 5.4 Verify the installation

```bash
python tools/self_test.py
```

The self-test verifies imports, model hashes, one real inference, the 3D
renderer, the eye analyser, the state machine, the event logger and the CLI. It
does not open a camera.

---

## 6. Models

### 6.1 Required models

| Stage | Model | File | Size | Input | Output |
| --- | --- | --- | --- | --- | --- |
| Face + landmarks | YuNet 2026may | `face_detection_yunet_2026may.onnx` | 0.23 MB | 640×480 (0.5× by default) | boxes, 5 landmarks |
| Head pose | 6DRepNet MobileNetV3-small | `headpose_mobilenetv3_small.onnx` | 6.11 MB | 224×224 RGB | 3×3 rotation matrix |
| Eye state | OCEC-S | `eyes_ocec_s.onnx` | 0.49 MB | 24×40 RGB, batch of 2 | open probability per eye |
| Driver identity | SFace 2021dec | `face_recognition_sface_2021dec.onnx` | 38.7 MB | 112×112 BGR aligned face | 128-value embedding |

### 6.2 Optional models

These are downloaded only with `--extra`. They are useful for comparison but
are not used by default.

| Model | File | Size | Notes |
| --- | --- | --- | --- |
| 6DRepNet MobileNetV2 | `headpose_mobilenetv2.onnx` | 8.9 MB | Slower reference pose model |
| OCEC-M | `eyes_ocec_m.onnx` | 1.7 MB | Larger eye classifier |

### 6.3 SHA-256

```text
face_detection_yunet_2026may.onnx
  ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0

headpose_mobilenetv3_small.onnx
  e8ae4d932b3d13221638fc72e171603e020c6da28b770753f76146867f40e190

eyes_ocec_s.onnx
  9a346a08b256ad70725044cd2aa582858e108c6f45d42a9c3415afc604ba9b64

face_recognition_sface_2021dec.onnx
  0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79

# optional
headpose_mobilenetv2.onnx
  1e902872868e483bd0e4f8f4a8ff2a4d61c2ccbca9dadf748e5479b5cc86a9e9

eyes_ocec_m.onnx
  810cf02f9c5a933fde688285aaf4548be0355092bcacbf3ee256ccfd01df6f6e
```

### 6.4 Model sources

- YuNet: [OpenCV Zoo face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- 6DRepNet ONNX weights:
  [yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation)
- OCEC:
  [PINTO0309/OCEC](https://github.com/PINTO0309/OCEC)
- SFace:
  [OpenCV Zoo face recognition](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)

---

## 7. Project structure

```text
driver_monitor/
├── driver_monitor.py          # CLI entry point
├── start_dms.sh               # environment/model check + launcher
├── requirements.txt
├── README.md                  # this document
├── calibration.json           # neutral pose (C key, F key or setup guide)
├── phone_camera.json          # saved DroidCam IP/port (optional)
├── driver_identity.json       # enrolled SFace embeddings, mode 0600 (optional)
│
├── dms/
│   ├── __init__.py
│   ├── config.py              # all thresholds, paths and runtime options
│   ├── camera.py              # Picamera2, V4L2, files and network streams
│   ├── models.py              # YuNet, 6DRepNet and OCEC wrappers
│   ├── geometry.py            # rotations, projection and angle helpers
│   ├── filters.py             # One Euro, EMA and timing statistics
│   ├── eyes.py                # blink / closure / PERCLOS / blink rate
│   ├── state.py               # zones, timers and the attention state machine
│   ├── safety.py              # GREEN / YELLOW / RED traffic-light logic
│   ├── identity.py            # SFace embeddings, store and verification
│   ├── setup.py               # framing checks and automatic calibration
│   ├── voice.py               # espeak-ng warning announcer
│   ├── gps.py                 # threaded gpsd JSON client
│   ├── dashboard_client.py    # LAN sender, payloads and offline queue
│   ├── enroll.py              # guided driver-enrolment flow
│   ├── events.py              # CSV, JPEG, pruning and session JSON
│   ├── head3d.py              # pure OpenCV/NumPy 3D head renderer
│   ├── render.py              # HUD, safety badge, identity chip, head inset
│   └── app.py                 # application loop, keyboard, buzzer, recording
│
├── models/                    # ONNX files (downloaded, not committed)
├── events/                    # event photos, dms_log.csv, session JSON,
                               # dashboard_outbox.jsonl (offline queue)
├── docs/
│   ├── README_v1_original.md  # preserved v1 documentation
│   ├── driver_monitor_v1.py   # preserved v1 application
│   └── head_pose_signs.png    # renderer reference
│
├── tools/
│   ├── download_models.py     # pinned-hash model downloader
│   ├── benchmark_models.py    # per-stage benchmark
│   ├── check_stream.py        # network camera probe + preview
│   └── self_test.py           # end-to-end installation check
│
└── tests/                     # unittest suite
```

---

## 8. Architecture

```text
 Camera / Picamera2 / V4L2 / video file / network stream
                  │
                  ▼
        resize + optional mirror
                  │
                  ▼
   YuNet at 0.5×  ──►  FaceTracker (largest face, EMA box/landmarks,
                  │                  short dropout hold)
                  │
      ┌───────────┴────────────┐
      ▼                        ▼
 6DRepNet (every 2 frames)   OCEC (both eyes, one batch)
      │                        │
 One Euro smoother         EMA + hysteresis
      │                        │
 yaw / pitch / roll       blink / closure / PERCLOS / rate
      └───────────┬────────────┘
                  ▼
       StateMachine + calibration
       zones → debounce → timers → priority
                  │
      ┌───────────┼─────────────┬──────────────┐
      ▼           ▼             ▼              ▼
   3D head      HUD          buzzer       EventLogger
  (OpenCV)   (OpenCV)      (gpiozero)   CSV + JPEG + JSON
```

The state machine and eye analyzer are independent of OpenCV and can be tested
with synthetic data. The models are loaded once and reused.

---

## 9. How the pipeline works

### 9.1 Face detection and tracking

YuNet is created with the configured camera resolution and an internal
detection scale (`face_input_scale`, default `0.50`). Each frame is downscaled
with `INTER_AREA`, the detector runs, and the returned boxes and five
landmarks are multiplied back to full-frame coordinates.

The largest detected face is treated as the driver. `FaceTracker` smooths the
box and landmarks and holds the previous detection for up to three missed
frames. The state machine still sees the real detection age, so a long loss
becomes `NO_FACE` rather than being hidden by the hold.

YuNet landmark order:

```text
0  image-left eye
1  image-right eye
2  nose tip
3  image-left mouth corner
4  image-right mouth corner
```

### 9.2 Head pose

The face is expanded by 30%, cropped to a square with edge replication near
the frame border, converted BGR→RGB, resized to 224×224, and normalized with
ImageNet mean/std. 6DRepNet outputs a 3×3 rotation matrix, decomposed into
pitch, yaw and roll with the same convention as the reference implementation.

Sign conventions used throughout the project:

```text
yaw   > 0  face turns to image-left
yaw   < 0  face turns to image-right
pitch > 0  face looks up
pitch < 0  face looks down
roll  > 0  head tilts clockwise on screen
```

Directions in the log and HUD are **image-relative**. When a camera faces the
driver, image-left corresponds to the driver's right. This is stated
explicitly so the labels are not misread as the driver's own left/right.

Head pose is scheduled every `headpose_every` frames (default `2`) and read
from a cache for the intervening frames. A One Euro filter smooths the angles;
the raw rotation matrix is used for the projected camera axes.

### 9.3 Eye state

OCEC receives a batch of two eye crops. The crop width is
`0.34 × face_width`, the height is `0.62 × crop_width`, and the crop is centred
on each YuNet eye landmark. Crops near the frame edge are padded with edge
pixels. If a crop is too small, too dark, or the head is beyond
`max_eye_yaw` (60°), the observation is marked invalid instead of guessing.

The classifier returns one open probability per eye. The eye analyzer applies:

- an EMA (`alpha = 0.55`) to reduce flicker,
- hysteresis: closed below `0.38`, open above `0.60`,
- blink completion when the eyes reopen,
- blink (`≤ 0.5 s`) vs long-blink (`> 0.5 s`) classification,
- continuous closure timing,
- PERCLOS over a rolling 60 s window,
- blink rate over a rolling 60 s window,
- microsleep counting from long closures.

PERCLOS is only considered ready after 20 s of valid eye samples, so the first
seconds after startup do not produce a false drowsiness warning.

### 9.4 State machine

The machine evaluates, in priority order:

1. face loss,
2. microsleep,
3. extreme distraction,
4. drowsiness,
5. sustained distraction,
6. short glance,
7. normal.

| State | Entry condition | Buzzer | Event request |
| --- | --- | --- | --- |
| `NORMAL` | Face present, pose forward, eyes open | off | — |
| `SHORT_GLANCE` | Zone leaves forward but off-road time < 2.5 s | off | — |
| `DISTRACTED` | Off-road ≥ 2.5 s | on | `SIDE_ATTENTION` |
| `DROWSY` | Both eyes closed ≥ 1.2 s, or PERCLOS ≥ 35% after warm-up | on | `DROWSY` |
| `NO_FACE` | Face missing; warning after 2.5 s | on | `NO_FACE` |
| `CRITICAL` | Microsleep ≥ 3.0 s, off-road ≥ 5.0 s, extreme angle for 2.0 s, or face missing ≥ 6.0 s | on | `MICROSLEEP`, `EXTREME_ATTENTION` or `NO_FACE_CRITICAL` |

Extreme angle means `|yaw| ≥ 55°` or `|pitch| ≥ 35°`. Critical/drowsy states
repeat their event request every 4 s while they persist; the event logger also
applies a per-event cooldown.

Gaze zones are built from the calibrated yaw/pitch:

```text
FORWARD
LEFT / RIGHT / UP / DOWN
LEFT_UP / LEFT_DOWN / RIGHT_UP / RIGHT_DOWN
NO_FACE / UNKNOWN
```

A zone must be stable for `offroad_debounce` (0.20 s) before it changes.
Returning forward must be stable for `forward_debounce` (0.40 s) before the
off-road timer is cleared. This prevents single-frame flicker from resetting a
warning.

### 9.5 Calibration

With a face visible, press `C`. The current yaw/pitch/roll is saved to
`calibration.json` and subtracted from future attention decisions. This
compensates for camera mounting angle. The 3D head still shows the measured
orientation relative to the camera.

- `--no-calibration` ignores the saved file.
- `R` resets the session state and clears the in-memory timers, but does not
  delete `calibration.json`.
- Delete `calibration.json` or press `C` again to re-zero.

### 9.6 3D head

The renderer builds a low-poly head mesh once at startup:

- about 1,250 triangles in seven parts,
- ellipsoid skull, custom hair cap, neck, ears, shoulders and collar,
- 2D brows, nose and lips projected onto the face,
- animated eyelids that follow the eye-open probabilities,
- top-view orientation dial,
- yaw, pitch and roll bars,
- status pill and eye labels.

Everything is drawn with OpenCV primitives and NumPy projection. There is no
OpenGL, pygame, VTK or other rendering dependency. Identical consecutive poses
are cached, so a still head costs almost nothing; a changing pose is the normal
case and is measured by the benchmark.

![3D head pose examples](docs/head_pose_signs.png)

The renderer follows the same sign convention as the model: yaw −45° turns the
face to image-right, yaw +45° to image-left, pitch +30° looks up, and roll +30°
tilts clockwise.

### 9.7 Timing and timestamps

Camera frames and live network streams use `time.monotonic()`. Video files use
`frame_index / fps`, so a slow CPU does not change PERCLOS or state durations
when replaying a clip. The recorder uses the source frame rate, not the
configured camera rate. A dropped network stream is retried for roughly 30
seconds before the application reports an error.

---

## 10. Running the application

### 10.1 Camera

```bash
source venv/bin/activate
python driver_monitor.py
```

Backend selection is automatic: Picamera2 first, then V4L2. To select one:

```bash
python driver_monitor.py --camera picamera2
python driver_monitor.py --camera v4l2 --camera-index 1
```

List devices and backends:

```bash
python driver_monitor.py --list-cameras
```

### 10.2 Video file or network stream

```bash
python driver_monitor.py --video clip.mp4
python driver_monitor.py --video clip.mp4 --loop-video
```

Video files are useful for repeatable demos and tests. The media clock is used
for all timing decisions.

`--video` also accepts live `http://`, `https://`, `rtsp://`, `rtmp://`,
`tcp://` and `udp://` URLs. Live streams use `time.monotonic()` and reconnect
after a short interruption.

```bash
python driver_monitor.py --video http://192.168.1.50:4747/video
python driver_monitor.py --video rtsp://user:pass@192.168.1.60:554/stream
```

Credentials and query strings are hidden in the banner and session summary.

### 10.3 Phone camera with DroidCam (Android)

The Pi does **not** need the DroidCam Linux client: the official Linux package
contains x86_64 binaries only, while this Pi is ARM64. The Android app's
built-in **IP Cam** mode serves an MJPEG stream that `--video` can read
directly.

The easiest setup is the built-in phone mode:

```bash
./start_dms.sh --phone
```

It prints the steps below, then asks for the address. Press Enter to accept
the saved/default value:

```text
Phone IP [10.107.190.123]:
Port [4747]:
Connecting to http://10.107.190.123:4747/video
```

The values are saved in `phone_camera.json`, so the next run offers the same
defaults. For scripts or non-interactive use:

```bash
python driver_monitor.py --phone --phone-ip 10.107.190.123 --phone-port 4747
```

Manual setup:

1. Install **DroidCam** from Google Play on the phone. The classic app and the
   newer DroidCam Webcam app both expose the camera over Wi-Fi.
2. Put the phone and the Pi on the same Wi-Fi network. Avoid guest networks
   with client isolation.
3. Open DroidCam and note the **Wi-Fi IP** and **port** shown on screen
   (normally port `4747`). If the IP is `0.0.0.0`, the phone is not on Wi-Fi.
4. Test the stream in a browser or VLC on the Pi:

   ```text
   http://PHONE_IP:4747/video
   ```

   Or use the built-in probe, which prints the resolution and frame rate and
   saves a preview:

   ```bash
   python tools/check_stream.py http://PHONE_IP:4747/video
   ```

5. Start the monitor:

   ```bash
   python driver_monitor.py --video http://PHONE_IP:4747/video
   ```

   or with the launcher:

   ```bash
   ./start_dms.sh --video http://PHONE_IP:4747/video
   ```

Notes:

- Only one client can use the phone stream at a time. Close the browser/VLC
  preview before starting the monitor.
- To request 720p or override an existing connection, append
  `/force/1280x720` to the URL, for example
  `http://PHONE_IP:4747/video/force/1280x720`.
- The free app usually delivers standard definition. HD quality may require
  the paid DroidCamX app or the newer DroidCam app's HD option.
- Keep DroidCam in the foreground and disable battery optimisation for it.
- USB tethering and phone hotspots work too, as long as the Pi can reach the
  phone's IP address.
- The browser IP Cam mode is documented for Android. On iPhone, use an app
  that exposes an MJPEG or RTSP URL; the same `--video` option works.

### 10.4 Headless recording

```bash
python driver_monitor.py \
    --video clip.mp4 \
    --headless \
    --record /tmp/session.mp4 \
    --event-dir /tmp/dms_events
```

Headless mode opens no windows. If `--record` is used with the default head
inset, the 3D head is still rendered into the recording. Use `--no-inset` to
skip it and save CPU.

### 10.5 Common options

| Option | Meaning |
| --- | --- |
| `--video FILE\|URL` | Read a video file or network stream instead of a camera |
| `--phone` | Android phone via DroidCam: show setup and prompt for IP/port |
| `--phone-ip IP` | Phone IP/hostname; skips the IP prompt (with `--phone`) |
| `--phone-port PORT` | Phone port; default `4747` (with `--phone`) |
| `--camera {auto,picamera2,v4l2}` | Camera backend |
| `--camera-index N` | V4L2 device index, default 0 |
| `--width`, `--height`, `--fps` | Requested camera geometry and rate |
| `--loop-video` | Restart the video file at the end |
| `--mirror` | Mirror the frame (also toggled with `M`) |
| `--model-dir DIR` | Directory containing the ONNX models |
| `--headpose-model FILE` | Alternate head-pose model filename/path |
| `--eye-model FILE` | Alternate eye model filename/path |
| `--headpose-every N` | Run head pose every N frames (default 2) |
| `--headless` | Do not open OpenCV windows |
| `--record FILE` | Write the annotated stream (`.mp4` or `.avi`) |
| `--event-dir DIR` | Event photos and CSV log directory |
| `--max-frames N` | Stop after N frames; 0 = unlimited |
| `--no-head-window` | Hide the separate 3D head window |
| `--no-inset` | Hide the 3D head in the HUD |
| `--no-buzzer` | Disable GPIO buzzer output |
| `--no-calibration` | Ignore `calibration.json` |
| `--debug` | Show the debug overlay (also toggled with `D`) |
| `--list-cameras` | Print V4L2 devices and Picamera2/display status |
| `--driver NAME` | Enforce an enrolled driver (repeatable) |
| `--enroll-driver NAME` | Capture a face profile and exit |
| `--enroll-force` | Replace an existing profile |
| `--enroll-samples N` | Good frames to capture (default 30) |
| `--identity-store FILE` | Profile JSON path (default `driver_identity.json`) |
| `--identity-threshold X` | SFace cosine threshold (default 0.363) |
| `--identity-every N` | Identity check interval in frames (default 5) |
| `--no-identity` | Disable identity verification |
| `--list-drivers` | List enrolled drivers and exit |
| `--setup-camera` | Run the guided framing setup, then monitor |
| `--setup-only` | Run framing setup and exit |
| `--auto-calibrate` | Run framing setup automatically at startup |
| `--setup-timeout S` | Framing setup timeout (default 30 s) |
| `--voice` | Enable spoken English warnings |
| `--voice-test` | Speak sample warnings and exit |
| `--voice-command CMD` | Speech command (default `espeak-ng`) |
| `--gps {off,gpsd}` | Read coordinates from gpsd (default `off`) |
| `--gps-host HOST` | gpsd host (default `127.0.0.1`) |
| `--gps-port PORT` | gpsd port (default `2947`) |
| `--dashboard-url URL` | Laptop dashboard, e.g. `http://192.168.1.20:8080` |
| `--dashboard-interval S` | Status push interval (default 1 s) |
| `--dashboard-timeout S` | HTTP timeout (default 5 s) |
| `--version` | Print the DMS version |

### 10.6 Keyboard shortcuts

| Key | Action |
| --- | --- |
| `Q` / `Esc` | Quit |
| `C` | Calibrate the neutral head pose |
| `F` | Open/close the framing and auto-calibration overlay |
| `R` | Reset session state and timers |
| `V` | Toggle spoken warnings (if installed) |
| `H` | Toggle the compact HUD |
| `M` | Mirror the camera |
| `B` | Toggle the buzzer |
| `D` | Toggle the debug overlay |
| `S` | Save an annotated snapshot |
| `?` or `/` | Show/hide help |

### 10.7 Startup script

```bash
./start_dms.sh
./start_dms.sh --phone
./start_dms.sh --video clip.mp4 --headless --record session.mp4
./start_dms.sh --video http://192.168.1.50:4747/video
./start_dms.sh --no-buzzer --debug
```

Started with no arguments, the script offers a menu: Pi/USB camera, Android
phone via DroidCam, camera framing setup, or driver enrolment. For monitoring
it also offers spoken warnings, a dashboard URL and gpsd. The script activates
the venv, checks the Python packages, verifies all four model hashes, prints
camera/display information, and then passes every argument to
`driver_monitor.py`.

To create an alias:

```bash
echo 'alias dms="$HOME/driver_monitor/start_dms.sh"' >> ~/.bashrc
source ~/.bashrc
dms
```

---

## 10A. Safety, identity, camera setup, voice, GPS and dashboard

These features are additive: the v2 detection pipeline, DroidCam mode, event
logging, buzzer, recording and 3D head all keep working.

### 10A.1 Three-level safety status

The HUD shows a traffic-light badge, and every event/status message carries the
same level.

| Level | Meaning | Produced by |
| --- | --- | --- |
| `GREEN` | Attentive | `NORMAL` and `SHORT_GLANCE` |
| `YELLOW` | Caution | `DISTRACTED`, `DROWSY`, `NO_FACE` |
| `RED` | Serious | `CRITICAL`, 8 s of continuous `YELLOW`, or a confirmed identity mismatch |

Hysteresis stops the badge from flickering: `RED` holds for 1.5 s before it can
drop, and `YELLOW` holds for 2 s before `GREEN`. The underlying v2 state machine
is unchanged, so the original event kinds and buzzer behaviour remain. The
buzzer also sounds on `RED`. The HUD border and alert banner use the safety
colour.

New event kinds: `SAFETY_RED` (persistent caution) and `SAFETY_RECOVERED`
(info, when returning from `YELLOW`/`RED`).

### 10A.2 Driver identity verification

SFace aligns the five YuNet landmarks to a 112×112 face and produces a 128-value
embedding. Only embeddings are stored, never enrolment photographs.

```bash
# 1. Enrol the driver once (about 30 good frontal frames)
python driver_monitor.py --phone --enroll-driver Alice

# 2. List / replace profiles
python driver_monitor.py --list-drivers
python driver_monitor.py --phone --enroll-driver Alice --enroll-force

# 3. Run with enforcement
python driver_monitor.py --phone --driver Alice

# Identify-only mode (no unauthorized-driver alarm):
python driver_monitor.py --phone
```

The profile lives in `driver_identity.json` (override with
`--identity-store`) and is written with mode `0600`. A check runs every 5
frames on faces at least 64 px wide and reasonably frontal. Three consecutive
mismatches confirm an unauthorized driver, which turns the status `RED` and
raises a `IDENTITY_MISMATCH` event. The default cosine threshold is `0.363`;
raise it if a similar person is accepted, lower it if the real driver is
rejected. Use `--no-identity` to disable the model entirely.

The store can hold several drivers; `--driver NAME` can be repeated to
authorize more than one. With no `--driver`, the system reports the best match
but never treats a face as unauthorized. OpenCV's `cv2.face.FaceRecognizerSF` is
not available in the pinned OpenCV build, so inference runs directly through
ONNX Runtime.

### 10A.3 Camera framing and auto-calibration

Head pose is measured relative to the camera, so moving the phone or Pi camera
invalidates `calibration.json`. Three ways to fix it:

```bash
# Guided setup screen, then continue into monitoring
python driver_monitor.py --phone --setup-camera

# Guided setup and exit
python driver_monitor.py --phone --setup-only

# Run the guide automatically whenever monitoring starts
python driver_monitor.py --phone --auto-calibrate
```

In the window, the guide draws a target frame, the detected face box and a
single instruction: face centering, distance, yaw/pitch/roll, lighting,
sharpness, then "HOLD STILL". After about 2 s of a good frame the neutral pose
is saved automatically. Press `Q`/`Esc` to cancel. Over SSH the guide prints
the same instructions and works headlessly.

During monitoring press `F` to reopen the framing overlay; pressing `F` again
closes it. `C` still calibrates from the current pose instantly. The new
`calibration.json` is version 3 and also records face ratio, brightness,
sharpness, resolution and source.

### 10A.4 Voice warnings

Install the English speech engine:

```bash
sudo apt install espeak-ng
python driver_monitor.py --voice-test      # hear the test phrases
python driver_monitor.py --phone --voice --driver Alice
```

Speech is off by default. The warner is non-blocking and rate-limits repeated
phrases (10 s). It speaks caution warnings (distraction, drowsiness, face not
visible), critical warnings (microsleep, face lost, extreme distraction,
identity mismatch, persistent warning), recovery and setup completion. Press
`V` in the camera window to mute/unmute at runtime. If `espeak-ng` is missing
the feature disables itself with an install hint. Use `--voice-command` only if
you have a compatible speech command.

### 10A.5 GPS coordinates

GPS comes from a local `gpsd`, not directly from the serial port:

```bash
sudo apt install gpsd gpsd-clients
# Set DEVICES="/dev/ttyUSB0" (or your GPS device) in /etc/default/gpsd
sudo systemctl restart gpsd
cgps -s                       # confirm a 2D/3D fix first

python driver_monitor.py --phone --gps gpsd --dashboard-url http://192.168.1.20:8080
```

The client reads the gpsd JSON protocol on `127.0.0.1:2947` (override with
`--gps-host` / `--gps-port`). Until a fresh 2D/3D fix arrives, and for 5 s after
a fix goes stale, the dashboard shows `GPS: waiting`. With `--gps off` it shows
`GPS: off`. No GPS hardware is required for the rest of the system.

### 10A.6 Laptop dashboard (Pi sender)

The dashboard runs on the Windows laptop; the Pi only pushes data. It is a
LAN-only service with **no login**, so never expose port 8080 to the internet
or forward it on your router.

```bash
python driver_monitor.py --phone --driver Alice \
    --dashboard-url http://192.168.1.20:8080
```

`start_dms.sh` prompts for the URL in interactive mode. The sender pushes a
status snapshot about once per second and one message per logged event. Events
carry a JPEG snapshot (max 960 px wide, quality 80), so the dashboard shows an
image per event without needing the video stream.

Endpoints the Pi uses (JSON, UTF-8):

| Method | Endpoint | Purpose |
| --- | --- | --- |
| POST | `/api/session/start` | session id, start time, source and driver |
| POST | `/api/status` | latest one-second snapshot |
| POST | `/api/event` | one event, deduplicate by `event_id` |
| POST | `/api/session/end` | session id, end time and summary |

Endpoints the dashboard should expose (read-only):

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/status` | last status snapshot |
| GET | `/api/events` | newest events first, with `event_id` |
| GET | `/api/sessions` | session list |
| GET | `/api/events.csv` | CSV export |
| GET | `/api/stream` | optional Server-Sent Events feed |
| GET | `/snapshots/<photo_name>` | event JPEG by filename |

Status object fields:

```text
session_id, timestamp, safety_level, safety_label, safety_reason,
state, reason, message, zone, face_present,
yaw, pitch, roll, eye_left, eye_right, perclos, blink_rate, closure_s,
offroad_seconds, no_face_seconds, status_seconds, fps,
identity_state, driver_name, identity_score, identity_message,
gps, voice_enabled, camera_setup, session_seconds
```

Event object fields:

```text
event_id, session_id, timestamp, event, kind, status, severity, direction,
message, safety_level, safety_reason, face_count,
yaw, pitch, roll, eye_left, eye_right, perclos, blink_rate, closure_s,
fps, latency_ms, identity_state, driver_name, identity_score,
identity_message, gps, snapshot_jpeg_base64, photo_name
```

The `gps` value is a nested object with `enabled`, `fix`, `lat`, `lon`, `alt`,
`speed_mps`, `track`, `mode`, `mode_label`, `satellites`, `hdop`, `age_s`,
`stale`, `text`, `received` and `error`. `text` is the ready-to-display string,
e.g. `GPS: waiting` or `GPS: -34.60370, -58.38160`.

If the laptop is offline the events are appended to
`events/dashboard_outbox.jsonl` and replayed automatically on the next start.
Duplicate `event_id`s should be ignored by the dashboard. No video is streamed
to the dashboard; the optional `/api/stream` is the dashboard's own SSE feed.

---

## 11. Logs, photos and sessions

### 11.1 Event photos

Event JPEGs are written to `events/` (or `--event-dir`) with a timestamp,
event kind and direction:

```text
events/20261113_200441_651692_SIDE_ATTENTION_RIGHT.jpg
```

Photos are annotated frames, including the HUD and, when the 3D renderer is
active in windowed or recording mode, the 3D head, at JPEG quality 90.
A per-event cooldown prevents a 30 fps camera from writing hundreds of files:
5 s for warnings, 4 s for critical events.

### 11.2 CSV schema

`events/dms_log.csv` keeps the four legacy columns first so older analysis
scripts still work, then the v2 columns, then the v3 safety/identity/GPS
columns:

```text
timestamp,event,direction,face_count,status,severity,yaw,pitch,roll,
eye_left,eye_right,perclos,blink_rate,closure_s,fps,latency_ms,message,photo,
safety_level,safety_reason,identity_state,identity_name,identity_score,
gps_lat,gps_lon,gps_text
```

| Column | Meaning |
| --- | --- |
| `timestamp` | Local wall-clock time |
| `event` | `SIDE_ATTENTION`, `DROWSY`, `MICROSLEEP`, `EXTREME_ATTENTION`, `NO_FACE`, `NO_FACE_CRITICAL`, `IDENTITY_MISMATCH`, `SAFETY_RED`, `SAFETY_RECOVERED` |
| `direction` | Gaze zone at the event (`LEFT`, `RIGHT`, `UP`, `LEFT_UP`, `NO_FACE`, …) |
| `face_count` | Faces detected in the frame |
| `status` | State-machine state |
| `severity` | `info`, `warning` or `critical` |
| `yaw`, `pitch`, `roll` | Head pose in degrees (raw model values) |
| `eye_left`, `eye_right` | OCEC open probabilities |
| `perclos` | Rolling PERCLOS at the event |
| `blink_rate` | Blinks per minute |
| `closure_s` | Current or last closure duration |
| `fps` | Processing frame rate |
| `latency_ms` | Sum of the current stage latencies |
| `message` | Human-readable warning text |
| `photo` | Path to the saved JPEG, or empty |
| `safety_level` | `GREEN`, `YELLOW` or `RED` |
| `safety_reason` | `DISTRACTION`, `DROWSINESS`, `NO_FACE`, `CRITICAL`, `IDENTITY_MISMATCH`, `PERSISTENT_WARNING`, … |
| `identity_state` | `OFF`, `NO_FACE`, `CHECKING`, `MATCH`, `MISMATCH`, `UNKNOWN` |
| `identity_name`, `identity_score` | Best/matched driver and cosine score |
| `gps_lat`, `gps_lon`, `gps_text` | Position at the event, or empty |

Example row (columns wrapped for readability):

```text
2026-11-23 11:48:41,SIDE_ATTENTION,RIGHT,1,DISTRACTED,warning,
-34.2,-2.7,4.0,1.00,1.00,0.0,0.0,0.0,8.0,110.3,
DISTRACTION WARNING,events/20261123_114841_651692_SIDE_ATTENTION_RIGHT.jpg,
YELLOW,DISTRACTION,MATCH,Alice,0.742,-34.60374,-58.38162,
"GPS: -34.60374, -58.38162"
```

### 11.3 Log rotation

The v1 log has only four columns and the v2 log has 18. On first start after an
upgrade, the current header is detected and the old file is moved aside:

```text
events/dms_log.csv                  # new 26-column v3 log
events/dms_log_v1_20261113_200438.csv
events/dms_log_v2_20261123_114800.csv
```

Old files are never modified. The new log starts with the compatible prefix.

### 11.4 Session summary

Every run writes a JSON summary:

```text
events/session_20261113_200438.json
```

It contains session start/end, duration, event counts, photo count, the log
path, the rotated legacy log path, and a nested summary with:

- frames processed and recorded,
- processing FPS and source info,
- state-frame and safety-level frame counts,
- final identity state and GPS snapshot,
- dashboard send/failure counters,
- longest off-road, closure and no-face durations,
- PERCLOS range and blink-rate maximum,
- blink, long-blink and microsleep counts,
- per-stage latency mean/p95/last,
- calibration offsets,
- the effective configuration.

### 11.5 Storage limits

By default the logger keeps at most 1,000 event photos and 250 MB of photos.
The oldest matching files are deleted when a limit is exceeded. Limits live in
`dms/config.py` (`max_events`, `max_event_mb`). Session JSON files are kept for
history.

---

## 12. Performance

Measured on the reference Pi 4 with the desktop and development tools running.
Use `tools/benchmark_models.py` on your own system; absolute numbers vary with
load, cooling, camera resolution and SD-card speed.

| Stage | Typical mean | Notes |
| --- | --- | --- |
| YuNet, 0.5× (320×240) | 27–45 ms | Full-resolution detection was 82–128 ms on the same Pi |
| 6DRepNet MobileNetV3-small | 25–45 ms | 3 ONNX threads; runs every 2 frames by default |
| OCEC-S, both eyes | 8–12 ms | Batch of two crops, 3 ONNX threads |
| 3D head at 420×420 | 12–18 ms | Varies with pose; cached poses are ~0.01 ms |
| Effective camera pipeline | ~90 ms/frame | ≈ 11 fps from the stage benchmark with pose every second frame |

The application summary reports the actual processing FPS for the run. During
the reference testing the desktop session and the development agent were also
using CPU, so observed application throughput was often 6–8 fps. On an idle Pi
the same code is expected to be faster; the benchmark reports both mean and
p95 so load spikes are visible.

### Performance controls

- `--headpose-every 3` reduces pose cost at the price of more interpolation.
- `--no-inset --no-head-window` removes the 3D renderer entirely.
- `--width 480 --height 360` reduces camera and detector work.
- `H` switches to the compact HUD.
- `--headless` avoids window compositing on SSH.
- The 3D renderer caches identical poses, which helps when the driver is still.

### Why these models

- **YuNet** is tiny, fast and gives five landmarks in one pass. A separate
  landmark model (PFLD106) was benchmarked and rejected at ~124 ms/frame.
- **6DRepNet MobileNetV3-small** measured ~25–45 ms on this Pi, versus
  ~66 ms for MobileNetV2, while providing full yaw/pitch/roll.
- **OCEC-S** is ~0.5 MB and fast enough to run on every frame. The larger
  OCEC-M is available with `--extra` for comparison.
- **MediaPipe** was not usable on this Python 3.13 / aarch64 installation
  (no compatible wheel, and the source build failed). The v1 README documents
  that investigation in detail.

---

## 13. Testing and diagnostics

### 13.1 Self-test

```bash
python tools/self_test.py
python tools/self_test.py --skip-inference
python tools/self_test.py --image events/your_photo.jpg
```

The self-test performs 28 checks:

- Python, OpenCV, NumPy and ONNX Runtime imports,
- all `dms` module imports,
- SHA-256 of the four required models (and optional models if present),
- one YuNet detection, one 6DRepNet pose and one OCEC eye inference,
- 3D head shape, content and rotation,
- state-machine transitions for normal, distraction, drowsiness, microsleep
  and face loss,
- blink, long-blink and dropout handling,
- CSV, JPEG and session-JSON writing in a temporary directory,
- `driver_monitor.py --version`,
- camera/display information (informational only).

It never opens a camera and never touches the real event log.

### 13.2 Unit and integration tests

```bash
python -m unittest discover -s tests -v
```

The suite contains 171 tests across:

| File | Coverage |
| --- | --- |
| `tests/test_geometry.py` | Angle wrapping, rotations, projection, helpers |
| `tests/test_filters.py` | Low-pass, EMA, One Euro, rate/timing statistics |
| `tests/test_eyes.py` | Blinks, long blinks, microsleep, hysteresis, PERCLOS, blink rate |
| `tests/test_state.py` | Zones, debounce, all states, events, calibration, repeats |
| `tests/test_safety.py` | Traffic-light mapping, persistence, latching, recovery, identity mismatch |
| `tests/test_identity.py` | SFace similarity, profile store/permissions, confirmation, enrolment |
| `tests/test_setup.py` | Framing checks, settle timer, overlay, calibration file |
| `tests/test_voice.py` | Phrase choice, cooldown, missing-command fallback |
| `tests/test_gps.py` | gpsd TPV/SKY parsing, stale fixes, disabled mode |
| `tests/test_dashboard.py` | Payload builders, HTTP posts, session markers, offline outbox replay |
| `tests/test_models.py` | Crops, sigmoid/logits, face tracker, half-res scaling, real models |
| `tests/test_head3d.py` | Output shape, rotation, cache, extreme angles, no-face |
| `tests/test_events.py` | Header, legacy rotation, records, cooldowns, pruning, session |
| `tests/test_camera.py` | Video timestamps, end-of-file, looping, source selection, network streams and reconnect |
| `tests/test_phone.py` | DroidCam defaults, prompts, address/port validation, persistence |
| `tests/test_render.py` | HUD panels, safety badge, identity chip, alerts, debug/help, inset |
| `tests/test_app.py` | Buzzer fallback, config snapshot, toast |
| `tests/test_integration.py` | Full headless pipeline on a generated video |

The integration test converts a real event photo into a short MJPG video, then
runs the complete application with temporary event/recording paths. It checks
processing frames, stage latencies, recording output, CSV header and session
JSON. It skips cleanly when models or a face photo are unavailable.

### 13.3 Benchmark

```bash
python tools/benchmark_models.py
python tools/benchmark_models.py --image events/driver.jpg --runs 30
python tools/benchmark_models.py --json /tmp/bench.json
```

The benchmark measures YuNet, 6DRepNet, OCEC and the 3D head separately. Head
poses are varied on every call so the renderer cache does not distort the
result. The reported "combined estimate" is conservative because it assumes
head pose runs every frame; the application runs it every second frame by
default.

### 13.4 Model verification

```bash
python tools/download_models.py --check
python tools/download_models.py --extra --check
```

A non-zero exit code means a file is missing or has the wrong hash.

---

## 14. Configuration and tuning

All defaults live in `dms/config.py`. The most useful values:

| Setting | Default | Effect |
| --- | --- | --- |
| `face_input_scale` | 0.50 | YuNet input scale; lower is faster, less precise |
| `face_confidence` | 0.60 | YuNet confidence threshold |
| `track_smoothing` | 0.55 | Higher = smoother, more lag |
| `track_max_missing` | 3 | Frames the face box is held after a dropout |
| `headpose_every` | 2 | Pose inference interval |
| `headpose_threads` | 3 | ONNX threads for pose |
| `eye_threads` | 3 | ONNX threads for eye inference |
| `yaw_glance` | 12° | Zone entry threshold |
| `pitch_glance` | 12° | Zone entry threshold |
| `yaw_extreme` | 55° | Extreme-angle trigger |
| `pitch_extreme` | 35° | Extreme-angle trigger |
| `offroad_debounce` | 0.20 s | Zone must be stable this long |
| `forward_debounce` | 0.40 s | Forward must be stable before timer reset |
| `distraction_seconds` | 2.50 s | Off-road warning |
| `distraction_critical_seconds` | 5.00 s | Off-road critical |
| `extreme_seconds` | 2.00 s | Extreme angle must persist this long |
| `no_face_seconds` | 2.50 s | Face-loss warning |
| `no_face_critical_seconds` | 6.00 s | Face-loss critical |
| `eye_closed_seconds` | 1.20 s | Drowsiness closure |
| `eye_closed_critical_seconds` | 3.00 s | Microsleep closure |
| `perclos_window` | 60 s | Rolling PERCLOS window |
| `perclos_threshold` | 0.35 | Drowsiness PERCLOS threshold |
| `perclos_min_time` | 20 s | Warm-up before PERCLOS is trusted |
| `blink_max_seconds` | 0.50 s | Blink vs long blink |
| `event_cooldown` | 5.0 s | Warning photo/log cooldown |
| `critical_repeat` | 4.0 s | Critical repeat/cooldown |
| `max_events` | 1000 | Photo count limit |
| `max_event_mb` | 250 | Photo storage limit |
| `safety_red_after` | 8.0 s | Continuous YELLOW before RED |
| `safety_red_hold` | 1.5 s | RED hold before a downgrade |
| `safety_green_hold` | 2.0 s | YELLOW hold before GREEN |
| `identity_threshold` | 0.363 | SFace cosine threshold |
| `identity_every` | 5 | Frames between identity checks |
| `identity_confirm_checks` | 3 | Checks before MATCH/MISMATCH is stable |
| `setup_settle_seconds` | 2.0 s | Stable framing before auto-calibration |
| `voice_cooldown` | 10.0 s | Minimum gap between identical phrases |
| `gps_stale_seconds` | 5.0 s | Age at which a fix is reported stale |
| `dashboard_interval` | 1.0 s | Status push interval |

### Tuning advice

- **Too many distraction warnings:** raise `distraction_seconds` (for example
  3.0–4.0 s) or raise `yaw_glance`/`pitch_glance`.
- **Warnings too slow:** lower `distraction_seconds`, but test with real
  driving footage first.
- **Too many drowsiness warnings:** raise `eye_closed_seconds` or
  `perclos_threshold`.
- **Face loss too sensitive:** raise `no_face_seconds` or `track_max_missing`.
- **Jittery 3D head:** lower `smooth_min_cutoff` or raise `track_smoothing`.
- **Laggy 3D head:** raise `smooth_min_cutoff` or lower `track_smoothing`.
- **Low FPS:** increase `headpose_every`, reduce the camera resolution, or use
  `--no-inset`. Identity adds a 112×112 SFace pass every 5 frames; use
  `--no-identity` if the Pi is overwhelmed.
- **False identity matches:** raise `identity_threshold`; **false rejects:**
  lower it, and re-enrol in better light with a frontal pose.
- **Bad automatic calibration:** increase `setup_settle_seconds` or run
  `--setup-camera` again with more even lighting.

Thresholds are starting points, not validated safety limits.

---

## 15. Behaviour and test scenarios

A controlled test should cover at least:

1. Driver looking forward → `NORMAL`.
2. Brief look left/right → `SHORT_GLANCE`, no buzzer.
3. Sustained look away → `DISTRACTED`, buzzer, event photo and CSV row.
4. Prolonged extreme angle → `CRITICAL` after 2 s.
5. Eyes closed ~0.2 s → one blink, no warning.
6. Eyes closed > 1.2 s → `DROWSY`.
7. Eyes closed > 3.0 s → `CRITICAL` microsleep.
8. High PERCLOS after 20 s of samples → `DROWSY`.
9. Face out of frame → `NO_FACE` after 2.5 s, `CRITICAL` after 6 s.
10. Short detection dropout → tracker hold, no false `NO_FACE`.
11. Repeated warnings → cooldown limits photos/log rows.
12. Replay a video twice → timing follows the media clock, not CPU speed.
13. Phone camera stream → `--video http://PHONE_IP:4747/video` works over Wi-Fi
    and recovers after a brief network interruption.
14. Enrolled driver → after three checks `IDENTITY MATCH`, `GREEN` stays.
15. Different person → after three checks `IDENTITY MISMATCH`, status `RED`,
    voice warning and a dashboard event.
16. Continuous caution for 8 s → `SAFETY_RED`, then `SAFETY_RECOVERED`.
17. Move the phone camera → `F` framing guide or `--setup-camera` saves a new
    neutral pose.
18. Stop the laptop dashboard → events appear in `dashboard_outbox.jsonl` and
    replay when it returns.

Use `--headless --record` for repeatable runs, then inspect the CSV, session
JSON and recording.

---

## 16. Troubleshooting

### No camera source available

```bash
python driver_monitor.py --list-cameras
ls -l /dev/video*
rpicam-hello
```

- Check the CSI ribbon cable and camera detection.
- Close any other process using the camera.
- For a DroidCam V4L2 setup, check `DROIDCAM_DEV` (default 0).
- Use `--video FILE` or a stream URL to confirm the rest of the pipeline.

### DroidCam stream will not open

1. Open `http://PHONE_IP:4747/video` in a browser on the Pi. If that fails,
   the problem is the phone/network, not the monitor.
2. Re-run `./start_dms.sh --phone` and enter the current address. To forget
   the saved defaults, delete `phone_camera.json` first.
3. Confirm the phone and Pi are on the same subnet and that the router does
   not isolate wireless clients. Try `ping PHONE_IP`.
4. Close browser tabs, VLC or other DroidCam clients. Only one connection is
   allowed at a time. Use `/video/force/1280x720` to override a stale session.
5. Re-check the IP in the app; DHCP may have changed it. A phone hotspot often
   uses `192.168.43.1` or `172.20.10.1`.
6. Keep DroidCam in the foreground and exempt it from battery optimisation.
7. If port `4747` is busy, change the port in the app and use it in the URL.
8. Do not install the DroidCam Linux package on this Pi: it is x86_64-only.
   Use the IP Cam URL instead.

### Picamera2 import or open failure

```bash
python -c "from picamera2 import Picamera2; print('ok')"
```

Install the apt package `python3-picamera2`, and create the venv with
`--system-site-packages`. A missing camera is reported as `No camera number 0
found`; that is a hardware/connection problem, not a model problem.

### ONNX Runtime missing or model load error

```bash
python -c "import onnxruntime as ort; print(ort.__version__)"
python tools/download_models.py --check
```

Install dependencies inside the activated venv. If a model hash fails, run
`python tools/download_models.py --force`.

### No display / `cv2.imshow` error over SSH

Use `--headless`. Windowed mode requires `DISPLAY` or `WAYLAND_DISPLAY`; the
application detects this and falls back to headless automatically.

### Buzzer does not sound

- Check GPIO number (`buzzer_gpio`, default 18).
- Check `gpiozero` and `lgpio`/`RPi.GPIO` installation.
- Verify transistor/MOSFET wiring and the external supply.
- Use `--no-buzzer` for desktop testing.
- On desktop systems `gpiozero` may create a mock device; no physical sound is
  expected.

### Calibration looks wrong

Delete `calibration.json` or start with `--no-calibration`, look straight
ahead, and press `C` again. Check the printed offsets. `R` resets session
timers but not the saved calibration.

### Old log was renamed

That is intentional. The v1 four-column log is moved to
`events/dms_log_v1_*.csv` on first v2 start. The new log uses the v2 schema.

### Low processing FPS

- Use `--headpose-every 3` or `4`.
- Use `--no-inset` and `--no-head-window` to disable the 3D renderer.
- Lower the camera resolution with `--width 480 --height 360`.
- Run headless and close other applications.
- Check the benchmark to see which stage is slow.

### The 3D head is not visible

- The main window shows an inset unless `--no-inset` is set.
- The separate window is enabled unless `--no-head-window` is set.
- In headless mode, the head is only rendered if `--record` is used and the
  inset is enabled.
- The renderer dims the mannequin when no face is present.

### Identity is disabled or always mismatches

- Check `python driver_monitor.py --list-drivers`.
- Enrol with `--enroll-driver NAME --enroll-force` in good light, looking
  straight at the camera.
- Confirm the name passed to `--driver NAME` exactly matches the profile.
- Make sure `models/face_recognition_sface_2021dec.onnx` exists and passes
  `python tools/download_models.py --check`.
- The face must be at least 64 px wide and reasonably frontal; use the `F`
  framing guide if necessary.

### Voice warnings do not play

- Install the engine: `sudo apt install espeak-ng`.
- Run `python driver_monitor.py --voice-test` to hear sample phrases.
- Check the audio device with `aplay -l`. `espeak-ng` does not need a sound
  server, but the Pi still needs an output device.
- Press `V` in the camera window to confirm speech is not muted.

### GPS shows "GPS: waiting"

- Confirm gpsd is running and has a fix: `cgps -s`.
- Check `/etc/default/gpsd` points to the right device and that the receiver
  has a clear sky view; a cold fix can take minutes.
- Verify port 2947 is reachable from the Pi (`nc -z 127.0.0.1 2947`).
- Remember that indoor GPS often never gets a fix; the rest of the system does
  not depend on it.

### Dashboard receives no data

- Confirm the laptop is pingable and its server listens on `0.0.0.0:8080`.
- Check the URL uses `http://LAPTOP_IP:8080` with no trailing path.
- The Pi prints `[DASH] connected` or `[DASH] offline (...)` on each change.
- Offline events are stored in `events/dashboard_outbox.jsonl` and replayed on
  the next start. If an event is missing, check whether its `event_id` was
  deduplicated by the dashboard.
- This is a LAN-only, unauthenticated service. Do not expose it publicly.

---

## 17. Privacy and safety

This is a prototype, not a certified automotive safety system. It must not be
the sole mechanism for detecting driver impairment.

Computer vision can fail with poor lighting, night driving, sunglasses, masks,
occlusion, vibration, camera misalignment, extreme angles, partial faces,
reflections, dirty lenses, multiple faces and false detections. PERCLOS and
blink thresholds here are software parameters, not medical or regulatory
thresholds.

Driver images are sensitive. Processing is local; nothing is uploaded to a
cloud service. Protect the event directory, define a retention period, use the
built-in photo limits, and delete images that are no longer needed. Inform
anyone whose image may be recorded and follow local privacy rules.

`driver_identity.json` stores biometric embeddings. It is written with mode
`0600` and contains no images; back it up securely and delete it if the driver
withdraws consent. The laptop dashboard receives status and event snapshots
over the LAN only, has no authentication, and must never be exposed to the
internet. Do not forward port 8080 on your router or bind the dashboard to a
public interface.

GPIO loads must be driven by a suitable transistor/MOSFET stage. Never power a
high-current buzzer directly from a GPIO pin.

---

## 18. v1 history and migration

The complete v1 README, including the Raspberry Pi camera investigation, the
MediaPipe/ARM64 build attempts, the original face-box heuristic, and the
original lessons learned, is preserved at:

- [`docs/README_v1_original.md`](docs/README_v1_original.md)

The original application is preserved at:

- [`docs/driver_monitor_v1.py`](docs/driver_monitor_v1.py)

Migration notes:

- v1 event photos remain in `events/`.
- The v1 four-column log is rotated, not overwritten.
- The v2 CSV begins with the same `timestamp,event,direction,face_count`
  columns.
- v1 thresholds and camera settings remain documented in the preserved README.
- The v2 code is modular; the old single-file program is still available for
  reference.

---

## 19. Development notes

- The application is tested without a camera using event photos and generated
  videos; camera backends should still be validated on the target Pi.
- Head-pose sign conventions were checked against real event images and the 3D
  renderer. See `docs/head_pose_signs.png`.
- `tools/benchmark_models.py` varies the head pose on every run because the
  renderer caches identical poses.
- Model hashes are duplicated in `tools/download_models.py` and the README;
  update both if a model is intentionally replaced.
- The state machine and eye analyzer do not import OpenCV, which keeps their
  unit tests fast.
- The 3D renderer uses only OpenCV and NumPy by design, so installation stays
  small on the Pi.

---

## 20. Credits

- **YuNet** — OpenCV Zoo face detection model.
- **6DRepNet** — head-pose architecture and ONNX weights via
  [yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation).
- **OCEC** — eye open/closed classifier by
  [PINTO0309](https://github.com/PINTO0309/OCEC).
- **OpenCV** and **ONNX Runtime** — inference and drawing.
- **gpiozero** — buzzer control.