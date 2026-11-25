# Driver Monitoring System (DMS)

## Raspberry Pi 4 (4 GB) + OV5647 Camera + OpenCV + YuNet + ONNX Runtime + GPIO Buzzer

---

# 1. Project Overview

This project implements a Driver Monitoring System (DMS) using a Raspberry Pi 4 with 4 GB RAM.

The purpose of the system is to continuously monitor a driver's face using a camera and detect situations where the driver's attention may be away from the forward driving direction.

The system currently provides:

- Continuous camera monitoring
- Face detection
- Approximate left/right attention detection
- Temporal distraction detection
- No-face detection
- Audible warning using a GPIO buzzer
- Automatic event photographs
- CSV event logging
- FPS monitoring
- A one-command startup script
- A Python virtual environment
- A lightweight ONNX face-detection model

The project is designed so that it can later be expanded into a more advanced DMS with:

- Eye closure detection
- Blink detection
- Drowsiness detection
- Head-pose estimation
- Mirror-check recognition
- Looking-through-window recognition
- Driver identity/tracking
- Multi-level warning systems
- Event severity classification
- Video recording
- Dashboard/GUI
- Automatic startup after Raspberry Pi boot

---

# 2. Hardware

## Main Computer

Raspberry Pi 4 Model B

RAM:

4 GB

Architecture:

aarch64 / ARM64

Operating System:

Debian GNU/Linux 13 (Trixie)

Python:

Python 3.13.x

---

## Camera

Camera sensor:

OV5647

The camera is connected using the Raspberry Pi CSI camera interface.

The camera was verified using the Raspberry Pi camera application.

---

## Buzzer

The project uses a GPIO-controlled buzzer.

Current GPIO:

GPIO 18

The Python application uses the `gpiozero` library to control the buzzer.

IMPORTANT:

Do not connect a high-current buzzer directly to a Raspberry Pi GPIO pin.

If the buzzer requires more current than the GPIO can safely provide, use:

- NPN transistor or MOSFET
- Appropriate resistor
- External power supply if necessary
- Common ground

The GPIO should control the switching device rather than directly powering a high-current load.

---

# 3. Final Project Structure

The project directory is:

    ~/driver_monitor

The structure is:

    driver_monitor/
    │
    ├── venv/
    │   └── Python virtual environment
    │
    ├── models/
    │   └── face_detection_yunet_2026may.onnx
    │
    ├── events/
    │   ├── dms_log.csv
    │   └── event photographs
    │
    ├── driver_monitor.py
    ├── start_dms.sh
    └── README.md

---

# 4. Software Architecture

The current system follows this basic pipeline:

    Camera
       |
       v
    Picamera2
       |
       v
    OpenCV frame
       |
       v
    YuNet face detector
       |
       v
    Face position
       |
       v
    Temporal attention logic
       |
       +--------------------+
       |                    |
       v                    v
    Driver OK          Possible distraction
                            |
                            v
                         Buzzer
                            |
                            v
                       Event photo
                            |
                            v
                         CSV log

The most important concept is that the system does not immediately classify every left/right movement as distraction.

A driver naturally looks around while driving.

Therefore the system uses a time threshold.

For example:

    Short glance
         |
         v
    No alarm

    Long sustained side attention
         |
         v
    Warning + event

This temporal logic is important for reducing false alarms.

---

# 5. Initial Camera Testing

Before building the DMS, the camera was tested independently.

The first command attempted was:

    libcamera-hello

This returned:

    bash: libcamera-hello: command not found

This initially looked like a camera software problem.

However, Debian/Raspberry Pi camera tools on this installation use the newer `rpicam-*` commands.

The following command was checked:

    which rpicam-hello

Result:

    /usr/bin/rpicam-hello

The installed version was checked using:

    rpicam-hello --version

The system reported:

    rpicam-apps build: v1.13.0
    libcamera build: v0.7.2

The camera was then tested using:

    rpicam-hello

The output showed that the OV5647 sensor was successfully detected.

Important conclusion:

The camera was working.

The problem was only that `libcamera-hello` was not the command available on this installation.

---

# 6. Camera Device Verification

The video devices were checked using:

    ls /dev/video*

Several `/dev/video*` devices were present.

The important conclusion was that the Raspberry Pi camera stack was functioning and libcamera successfully registered the OV5647 sensor.

The camera reported:

    ov5647@36

and the Raspberry Pi pipeline was successfully configured.

---

# 7. First Python Problem: OpenCV Haar Cascade

An initial Python program attempted to use:

    cv2.data.haarcascades

The program failed with:

    AttributeError: module 'cv2' has no attribute 'data'

This demonstrated that the installed OpenCV package/version did not expose the expected `cv2.data` interface.

Instead of depending on a hard-coded OpenCV Haar-cascade path, the project was moved toward a model-based face detector.

This eventually led to using YuNet.

---

# 8. MediaPipe Investigation

The project initially attempted to use MediaPipe because MediaPipe provides useful face-landmark functionality.

The first check was:

    python3 -m pip show mediapipe

The result was:

    WARNING: Package(s) not found: mediapipe

An import test produced:

    ModuleNotFoundError: No module named 'mediapipe'

---

# 9. Why MediaPipe Was Difficult

The Raspberry Pi uses:

    aarch64

The system uses:

    Python 3.13

Official MediaPipe Python packages do not provide the required Linux ARM64 wheel for this environment.

Therefore a source build was investigated.

Bazel/Bazelisk was required.

---

# 10. Bazel Installation

The Bazel command was initially unavailable.

Bazelisk was installed for ARM64 and exposed as:

    /usr/local/bin/bazel

The command:

    which bazel

returned:

    /usr/local/bin/bazel

The command:

    bazel --version

initially produced:

    bazel no_version

This indicated that the Bazelisk launcher existed but was not successfully selecting/downloading a Bazel version.

The Bazelisk installation was corrected.

After correction, Bazel became usable.

---

# 11. MediaPipe Source Build Attempt

The MediaPipe source was cloned and a source build was attempted.

The build progressed for several minutes.

However, the build eventually failed with:

    no such package 'mediapipe/tasks/c/decision/decision_maker'

Specifically:

    BUILD file not found

The failing target was related to:

    //mediapipe/tasks/c:libmediapipe.so

This was important because it demonstrated that simply cloning the latest MediaPipe source and building the C library was not a reliable path for this particular Raspberry Pi environment.

The build took approximately ten minutes before failing.

Lesson:

Do not repeatedly rebuild large projects without first confirming that the exact target is appropriate for the required application.

---

# 12. MediaPipe Version Problem

The MediaPipe source also encountered a Python packaging problem.

The source package contained:

    __version__ = 'dev'

The installed Python packaging tools rejected this with:

    InvalidVersion: Invalid version: 'dev'

This was another compatibility issue caused by the interaction between the current source tree and modern Python packaging.

A temporary source modification was considered:

    __version__ = '0.1.0'

However, the subsequent Bazel build still failed because of the missing C package.

This showed that fixing the version string alone was not sufficient.

---

# 13. ARM64 MediaPipe Wheel Investigation

An ARM64 MediaPipe repository was investigated.

The repository contained:

    mediapipe-0.8.1-cp38-cp38-linux_aarch64.whl

and:

    mediapipe-0.8.4-cp38-cp38-linux_aarch64.whl

However, these wheels were compiled for:

    CPython 3.8

while this project uses:

    Python 3.13

Therefore these wheels cannot be directly installed into the current Python environment.

An attempted newer wheel URL also returned:

    HTTP 404

Lesson:

Never assume that a wheel URL exists merely because a project/repository exists.

Always inspect the available files/releases and verify:

- Python version
- CPU architecture
- operating system
- ABI
- package version

before installing a wheel.

---

# 14. Decision to Use OpenCV + ONNX Runtime

Because MediaPipe was creating unnecessary build complexity on this system, the project switched to:

    OpenCV
        +
    ONNX Runtime
        +
    YuNet

This is much lighter and more practical for a Raspberry Pi 4.

The environment already had:

    OpenCV 5.0.0

and:

    NumPy 2.2.4

---

# 15. ONNX Runtime Installation

ONNX Runtime was installed inside the virtual environment.

The installed version was:

    ONNX Runtime 1.30.0

The available execution providers were:

    AzureExecutionProvider
    CPUExecutionProvider

For the Raspberry Pi, the important provider is:

    CPUExecutionProvider

The test command was:

    python3 -c "import onnxruntime as ort; print('ONNX OK:', ort.__version__); print('Providers:', ort.get_available_providers())"

The environment successfully returned:

    ONNX OK: 1.30.0

Therefore ONNX Runtime was working.

---

# 16. Final Python Environment Test

The following components were tested together:

    Picamera2
    OpenCV
    NumPy
    ONNX Runtime

The command:

    python3 -c "from picamera2 import Picamera2; import cv2; import numpy; import onnxruntime; print('DMS ENVIRONMENT OK')"

returned:

    DMS ENVIRONMENT OK

This confirmed that the Python environment was ready.

---

# 17. YuNet Face Detector

The project uses the OpenCV Zoo YuNet face detector.

Model:

    face_detection_yunet_2026may.onnx

The model is stored at:

    models/face_detection_yunet_2026may.onnx

The model was downloaded from the OpenCV Zoo repository.

The downloaded file was approximately:

    225 KB

---

# 18. Why YuNet

YuNet is a lightweight face detector.

It is suitable as the first stage of a Raspberry Pi DMS because it provides:

- Face detection
- Bounding box
- Confidence score
- Lightweight computation
- OpenCV integration
- ONNX model format

However:

IMPORTANT:

YuNet is primarily a face detector.

It is NOT by itself a complete driver-monitoring system.

It does not automatically provide accurate:

- Eye state
- Head yaw
- Head pitch
- Head roll
- Drowsiness
- Blink duration
- Mirror-check classification
- Looking-through-window classification

These require additional analysis.

---

# 19. YuNet Test

The model was tested with:

    python3 -c "import cv2; m=cv2.FaceDetectorYN.create('models/face_detection_yunet_2026may.onnx','',(640,480),0.6,0.3,5000); print('YUNET OK')"

The result was:

    YUNET OK

OpenCV displayed a warning similar to:

    Targets are not supported by the new graph engine for now

This warning did not prevent the model from loading.

The important result was:

    YUNET OK

---

# 20. Current DMS Algorithm

The current system operates at:

    640 x 480

The camera provides frames through Picamera2.

Each frame is passed to YuNet.

The largest detected face is selected as the driver.

The approximate horizontal face location is calculated.

The image is divided conceptually into:

    LEFT | CENTER | RIGHT

The system uses a normalized horizontal offset.

If the face is sufficiently left:

    LEFT

If the face is sufficiently right:

    RIGHT

Otherwise:

    CENTER

---

# 21. Why Time-Based Detection Is Used

A simple system might do:

    LEFT = DISTRACTED
    RIGHT = DISTRACTED

This would be a poor DMS design.

Drivers naturally look:

- At the left mirror
- At the right mirror
- At traffic
- At intersections
- At pedestrians
- At cyclists
- Over their shoulder
- Through side windows
- At instruments
- At passengers
- Around obstacles

Therefore a brief glance should not automatically cause an alarm.

The current implementation uses:

    SIDE_WARNING_SECONDS = 3.0

Meaning:

    side glance < 3 seconds
        =
    short glance

and:

    side attention >= 3 seconds
        =
    possible distraction

This is only a starting threshold.

Real-world testing should be used to tune the threshold.

---

# 22. No-Face Detection

Another important condition is:

    NO DRIVER FACE

If the face detector does not find a face, a timer starts.

Current threshold:

    NO_FACE_WARNING_SECONDS = 2.5

If the face is missing for less than this time:

    No immediate alarm

If the face remains missing beyond this time:

    warning
    +
    event photograph
    +
    buzzer

This prevents a single missed detection from immediately triggering an alarm.

---

# 23. Event Photographs

When a significant event occurs, the system saves a JPEG.

Files are stored in:

    events/

Example:

    events/
        20261002_130900_SIDE_ATTENTION_LEFT.jpg

The timestamp makes it possible to identify when the event occurred.

---

# 24. CSV Logging

The system maintains:

    events/dms_log.csv

The CSV contains fields such as:

    timestamp
    event
    direction
    face_count

This allows the system history to be analyzed later.

For example:

    timestamp,event,direction,face_count
    2026-10-02 13:09:00,SIDE_ATTENTION,LEFT,1
    2026-10-02 13:10:14,NO_FACE,NO_FACE,1

The logging system can later be expanded with:

- Event duration
- Confidence
- Eye state
- Head yaw
- Head pitch
- Head roll
- Drowsiness score
- Warning level
- Vehicle speed
- GPS information
- Accelerator/brake information

---

# 25. Buzzer Logic

The buzzer is connected through:

    GPIO 18

The Python program uses:

    gpiozero.Buzzer

The alarm function briefly activates the buzzer.

Current behavior:

    warning
       |
       v
    short beep

This can later become a multi-stage warning system.

Example future design:

    Level 0:
    Normal

    Level 1:
    Visual warning

    Level 2:
    Short beep

    Level 3:
    Repeated beep

    Level 4:
    Continuous warning

This should be implemented carefully to avoid unnecessary alarm fatigue.

---

# 26. Event Cooldown

The system contains:

    EVENT_COOLDOWN = 5.0

This prevents the system from writing an event photograph every frame.

Without a cooldown:

    30 FPS
       |
       v
    hundreds of photographs
    per minute

With the cooldown:

    event
      |
      v
    save photograph
      |
      v
    wait for cooldown
      |
      v
    next event

This greatly reduces storage usage.

---

# 27. Current Project Limitations

The current implementation is a foundation.

It should NOT be described as a production-certified driver monitoring system.

The biggest limitation is head direction.

The current algorithm uses:

    face location in image

as an approximation.

This is not the same thing as:

    actual head yaw

For example, a driver could:

- Turn their head while remaining near the image center
- Move their entire body
- Lean toward the passenger side
- Lean toward the window
- Look into a mirror while the face remains near the same position

Therefore a proper next-generation DMS needs facial landmarks and head-pose estimation.

---

# 28. Mirror Checks

Mirror checks are a particularly important DMS scenario.

A driver may look:

    LEFT MIRROR
         |
         v
    brief glance
         |
         v
    return CENTER

This should normally be treated differently from:

    LEFT
      |
      v
    remains LEFT
      |
      v
    prolonged off-road attention

The correct solution is not simply to remove left/right detection.

Instead, the system should track:

- Direction
- Duration
- Frequency
- Return-to-center behavior
- Eye state
- Head pose
- Context

---

# 29. Looking Out the Window

A driver may legitimately look through a side window.

Examples:

- Checking traffic
- Merging
- Turning
- Parking
- Watching pedestrians
- Giving directions
- Driving through a narrow road

Therefore:

    looking LEFT

does not automatically mean:

    distracted

A more advanced system should combine:

    head pose
    +
    eye gaze
    +
    duration
    +
    vehicle context

before deciding whether an alert is appropriate.

---

# 30. Traffic Guidance Scenario

A driver may temporarily look away from the road while:

- Guiding another vehicle
- Checking traffic
- Talking to a traffic officer
- Navigating a difficult intersection
- Watching a pedestrian
- Checking a cyclist

The system should therefore avoid overly aggressive thresholds.

This is one reason the project uses temporal filtering.

---

# 31. Drowsiness Detection

Drowsiness is NOT currently fully implemented.

A proper drowsiness system should analyze:

- Eye aspect ratio
- Eye closure duration
- Blink frequency
- Long blinks
- PERCLOS-like measures
- Head nodding
- Face orientation
- Temporal behavior

For example:

    eyes open
       |
       v
    normal

versus:

    eyes closed
       |
       v
    0.2 seconds
       |
       v
    normal blink

versus:

    eyes closed
       |
       v
    1.5 seconds
       |
       v
    possible drowsiness

This is a future enhancement.

---

# 32. Why Eye Detection Is Important

A driver may have their face facing forward but still be looking down.

Examples:

- Looking at a phone
- Looking at a dashboard
- Looking at controls
- Looking down due to fatigue

Face detection alone cannot reliably detect these cases.

Therefore the next major upgrade should include eye/iris or facial-landmark analysis.

---

# 33. Recommended Future Architecture

A more advanced DMS should use:

    CAMERA
       |
       v
    Face Detector
       |
       v
    Facial Landmarks
       |
       +--------------------+
       |                    |
       v                    v
    Eye Analysis        Head Pose
       |                    |
       +---------+----------+
                 |
                 v
          Temporal Tracker
                 |
                 v
           State Machine
                 |
       +---------+---------+
       |         |         |
       v         v         v
    NORMAL   DISTRACTED  DROWSY
       |         |         |
       +---------+---------+
                 |
                 v
          Warning Manager
                 |
       +---------+---------+
       |                   |
       v                   v
    Buzzer              Event Log
                           |
                           v
                      Event Photo

---

# 34. State Machine Concept

A state machine is better than a collection of independent `if` statements.

Possible states:

    NORMAL

    SHORT_GLANCE

    SUSTAINED_LEFT

    SUSTAINED_RIGHT

    NO_FACE

    EYES_CLOSED

    DROWSY

    WARNING

The system can transition between states.

Example:

    NORMAL
       |
       | driver looks left
       v
    SHORT_GLANCE
       |
       | returns center
       v
    NORMAL

Another example:

    NORMAL
       |
       | looks left
       v
    SHORT_GLANCE
       |
       | remains left
       v
    SUSTAINED_LEFT
       |
       v
    WARNING

This is much more robust than:

    if left:
        buzzer

---

# 35. Why Temporal Filtering Matters

Computer vision produces noisy measurements.

A single frame may incorrectly detect:

    CENTER

then:

    LEFT

then:

    CENTER

then:

    LEFT

even if the driver barely moved.

Therefore decisions should be based on a sequence of frames.

Conceptually:

    Frame 1 -> LEFT
    Frame 2 -> LEFT
    Frame 3 -> LEFT
    Frame 4 -> LEFT
    ...
    Frame N -> LEFT

Only after enough consistent evidence should the system trigger a warning.

---

# 36. Performance Considerations

The Raspberry Pi 4 has limited CPU resources compared with a desktop computer.

Therefore:

- Use modest camera resolution
- Use lightweight models
- Avoid unnecessarily large neural networks
- Avoid running several heavy models every frame
- Use temporal sampling where appropriate
- Monitor FPS
- Use hardware acceleration only when properly supported

Current resolution:

    640 x 480

This is a reasonable starting point.

Higher resolution improves visual detail but increases computation.

---

# 37. FPS Monitoring

The application displays FPS.

Example:

    FPS: 24.5

FPS is important because DMS decisions depend on temporal measurements.

If FPS drops significantly:

    30 FPS

may become:

    8 FPS

Then a time calculation based purely on frame count becomes inaccurate.

The current system uses real elapsed time rather than assuming a fixed FPS.

This is important.

---

# 38. Python Virtual Environment

The project uses:

    venv/

This isolates project dependencies from the system Python installation.

Activate it using:

    cd ~/driver_monitor
    source venv/bin/activate

The prompt should show something similar to:

    (venv) nightmare@nightmare:~/driver_monitor $

This is useful because:

- System packages remain untouched
- Python dependencies remain isolated
- Different projects can use different versions
- The DMS can be reproduced more easily

---

# 39. Starting the Application

The project includes:

    start_dms.sh

The script automatically:

1. Changes into the project directory
2. Activates the virtual environment
3. Checks Python dependencies
4. Checks Picamera2
5. Checks the YuNet model
6. Checks `driver_monitor.py`
7. Starts the DMS

Run:

    ~/driver_monitor/start_dms.sh

or:

    cd ~/driver_monitor
    ./start_dms.sh

---

# 40. Optional Short Command

An alias can be created:

    echo 'alias dms="$HOME/driver_monitor/start_dms.sh"' >> ~/.bashrc

Then reload the shell:

    source ~/.bashrc

After that:

    dms

starts the application.

---

# 41. Stopping the Application

The application can be stopped from the camera window by pressing:

    Q

The program then:

- Turns off the buzzer
- Stops the camera
- Closes the OpenCV window
- Exits cleanly

---

# 42. Checking Event Files

To view saved event photographs:

    ls -lh ~/driver_monitor/events

To inspect the CSV:

    cat ~/driver_monitor/events/dms_log.csv

For a larger log:

    less ~/driver_monitor/events/dms_log.csv

---

# 43. Troubleshooting

## Camera command not found

If:

    libcamera-hello

does not exist, check:

    which rpicam-hello

Modern Raspberry Pi camera installations may use:

    rpicam-hello

instead.

---

## Camera not detected

Check:

    ls /dev/video*

Then test:

    rpicam-hello

Check that the camera sensor is detected.

---

## YuNet model not found

Expected location:

    ~/driver_monitor/models/face_detection_yunet_2026may.onnx

Check:

    ls -lh ~/driver_monitor/models/

---

## OpenCV import failure

Test:

    python3 -c "import cv2; print(cv2.__version__)"

Expected in this project:

    OpenCV 5.0.0

---

## ONNX Runtime failure

Test:

    python3 -c "import onnxruntime as ort; print(ort.__version__)"

---

## Picamera2 failure

Test:

    python3 -c "from picamera2 import Picamera2; print('CAMERA OK')"

---

## Buzzer failure

Check the physical wiring.

Check GPIO number.

Current software configuration:

    BUZZER_GPIO = 18

Also check that the buzzer is compatible with the electrical setup.

---

# 44. Important GPIO Safety

Raspberry Pi GPIO pins are not general-purpose power outputs.

Never assume a GPIO can directly drive any buzzer, motor, relay, lamp, or other load.

For loads requiring significant current, use:

    GPIO
      |
      v
    transistor/MOSFET
      |
      v
    load

The driver circuit must be designed according to the load requirements.

---

# 45. Lessons Learned During Development

## Lesson 1: Verify hardware first

Before installing machine-learning libraries, verify the camera independently.

This prevents software debugging from being confused with hardware problems.

---

## Lesson 2: Check the actual operating system

This project uses:

    Debian 13 / Trixie

Some Raspberry Pi tutorials assume Raspberry Pi OS versions with different packages.

Commands and available Python versions can differ.

---

## Lesson 3: Check architecture

The system reports:

    aarch64

Architecture matters when installing:

- Python wheels
- ONNX Runtime
- MediaPipe
- TensorFlow
- PyTorch
- OpenCV

A wheel compiled for x86_64 cannot be used on ARM64.

---

## Lesson 4: Check Python ABI

A wheel containing:

    cp38-cp38

is for:

    CPython 3.8

It is not automatically compatible with:

    CPython 3.13

This was one of the major issues encountered while investigating MediaPipe.

---

## Lesson 5: Do not confuse face detection with head pose

Face detection answers:

    "Where is the face?"

Head-pose estimation answers:

    "Which direction is the head facing?"

These are different problems.

---

## Lesson 6: Do not classify every glance as distraction

Drivers naturally look around.

A good DMS needs:

    temporal logic
    +
    head pose
    +
    eye state
    +
    context

rather than a single-frame rule.

---

## Lesson 7: Lightweight models are important

The Raspberry Pi 4 is capable of computer vision, but computational resources are limited.

Lightweight models are preferable for continuous real-time monitoring.

---

# 46. Current Configuration

Current camera resolution:

    640 x 480

Face confidence:

    0.60

NMS threshold:

    0.30

Maximum detections:

    5000

Side-attention warning:

    3.0 seconds

No-face warning:

    2.5 seconds

Event cooldown:

    5.0 seconds

Buzzer GPIO:

    GPIO 18

---

# 47. How to Tune the System

If the system gives warnings too easily:

Increase:

    SIDE_WARNING_SECONDS

For example:

    3.0 -> 4.0

If the system reacts too slowly:

Decrease it:

    3.0 -> 2.0

Similarly:

    NO_FACE_WARNING_SECONDS

controls how long the face can be missing before a warning.

These values should be experimentally evaluated.

They should not be treated as universally correct safety thresholds.

---

# 48. Testing Method

A controlled testing procedure is recommended.

Test 1:

Driver looking forward.

Expected:

    NORMAL

Test 2:

Brief left glance.

Expected:

    SHORT_GLANCE

No immediate alarm.

Test 3:

Brief right glance.

Expected:

    SHORT_GLANCE

No immediate alarm.

Test 4:

Long left attention.

Expected:

    WARNING

Test 5:

Long right attention.

Expected:

    WARNING

Test 6:

Driver moves outside camera view.

Expected:

    NO_DRIVER_FACE

followed by warning after the configured timeout.

Test 7:

Temporary detection failure.

Expected:

    No immediate alarm.

Test 8:

Repeated left/right movements.

Expected:

    System should not continuously generate event photos because of the event cooldown.

---

# 49. Future Improvements

The next development stage should add a proper facial landmark system.

Recommended components:

## Facial landmarks

Needed for:

- Eyes
- Nose
- Mouth
- Face geometry

## Head pose

Estimate:

- Yaw
- Pitch
- Roll

Example:

    YAW = -35 degrees
    PITCH = +5 degrees
    ROLL = -2 degrees

This is more meaningful than only measuring face position.

---

# 50. Eye Analysis

The next version should calculate eye state.

Possible outputs:

    EYES_OPEN

    BLINK

    EYES_CLOSED

Then use duration:

    0.1 sec closed
        =
    normal blink

    1.5 sec closed
        =
    possible drowsiness

Again, thresholds need validation.

---

# 51. Drowsiness Score

A future scoring system could combine:

    Eye closure
    +
    Blink duration
    +
    Blink frequency
    +
    Head movement
    +
    Face orientation

Example conceptual score:

    0-30    NORMAL

    30-60   ATTENTION

    60-80   DROWSINESS WARNING

    80-100  HIGH ALERT

These values are examples for software design only and must not be interpreted as medically validated thresholds.

---

# 52. Mirror-Check Intelligence

A future system can model mirror checks as patterns.

Example:

    CENTER
       |
       v
    LEFT
       |
       v
    CENTER

If this lasts only a short period, it can be classified as a normal glance candidate.

Another pattern:

    CENTER
       |
       v
    LEFT
       |
       v
    LEFT
       |
       v
    LEFT
       |
       v
    LEFT

This can be treated as sustained side attention.

The key idea is:

    Pattern + duration

rather than:

    Single direction

---

# 53. Driver Monitoring State Machine

A future implementation could use:

    NORMAL
       |
       +--> SHORT_GLANCE
       |
       +--> DROWSINESS
       |
       +--> NO_FACE
       |
       +--> DISTRACTION
       |
       +--> WARNING

Each state should have:

- Entry condition
- Exit condition
- Timer
- Confidence
- Warning level

This prevents unstable behavior.

---

# 54. Event Severity

Future events could have levels:

    INFO
    ATTENTION
    WARNING
    CRITICAL

For example:

    INFO:
    short mirror glance

    ATTENTION:
    repeated prolonged side attention

    WARNING:
    sustained distraction

    CRITICAL:
    prolonged no-face + possible drowsiness

The actual thresholds need to be validated experimentally.

---

# 55. Storage Management

Continuous event photography can eventually fill the SD card.

Future versions should implement:

- Maximum storage size
- Automatic deletion of old events
- Daily folders
- Event compression
- Configurable retention period

For example:

    events/
       2026-10-02/
       2026-10-03/
       2026-10-04/

---

# 56. Automatic Startup

The next stage can use a Linux `systemd` service.

For example:

    driver-monitor.service

The service could automatically start the DMS after boot.

Conceptually:

    Raspberry Pi boot
          |
          v
    systemd
          |
          v
    DMS service
          |
          v
    Camera
          |
          v
    Monitoring

This is preferable to manually starting the application every time.

---

# 57. Possible Web Dashboard

A future version could provide a local dashboard.

Possible information:

    Driver Status: NORMAL

    Face: DETECTED

    Head: CENTER

    Eyes: OPEN

    Drowsiness: LOW

    FPS: 24

    Events Today: 4

The dashboard could run locally on the Raspberry Pi.

---

# 58. Possible Data Storage

Instead of only CSV, future versions could use:

    SQLite

Database tables could include:

    events
    sessions
    driver_state
    warnings

This makes querying historical data easier.

---

# 59. Privacy

Driver-monitoring systems process potentially sensitive visual information.

The system should preferably process data locally.

Avoid uploading driver images to external servers unless there is a clear requirement and appropriate authorization.

If event photographs are stored:

- Protect the storage
- Define a retention period
- Delete unnecessary images
- Restrict access
- Inform users appropriately

---

# 60. Safety Considerations

This project is a prototype.

It should not be relied upon as the sole safety mechanism for operating a vehicle.

Computer vision can fail because of:

- Poor lighting
- Night driving
- Sunglasses
- Masks
- Occlusion
- Camera vibration
- Camera misalignment
- Extreme head angles
- Partial face visibility
- Rain
- Reflections
- Dirty camera lens
- Multiple faces
- False detections
- Missed detections

A real automotive DMS requires extensive validation and safety engineering.

---

# 61. Current Status

The following components are working:

    [x] Raspberry Pi camera
    [x] OV5647 detection
    [x] Picamera2
    [x] Python virtual environment
    [x] OpenCV
    [x] NumPy
    [x] ONNX Runtime
    [x] YuNet
    [x] Face detection
    [x] Approximate left/right detection
    [x] Temporal distraction timer
    [x] No-face detection
    [x] GPIO buzzer
    [x] Event photographs
    [x] CSV logging
    [x] One-command startup

---

# 62. Not Yet Fully Implemented

    [ ] True head-pose estimation
    [ ] Facial landmarks
    [ ] Eye detection
    [ ] Blink detection
    [ ] Drowsiness detection
    [ ] PERCLOS-like measurement
    [ ] Mirror-check recognition
    [ ] Side-window recognition
    [ ] Phone detection
    [ ] Driver gaze estimation
    [ ] Advanced temporal state machine
    [ ] Warning levels
    [ ] Automatic boot service
    [ ] Web dashboard
    [ ] SQLite database
    [ ] Storage cleanup
    [ ] Night-time optimization
    [ ] Extensive real-world validation

---

# 63. Quick Start

To start the system:

    cd ~/driver_monitor
    source venv/bin/activate
    ./start_dms.sh

Or simply:

    ~/driver_monitor/start_dms.sh

If the `dms` alias was configured:

    dms

---

# 64. Useful Commands

Check architecture:

    uname -m

Check operating system:

    cat /etc/os-release

Check Python:

    python3 --version

Check pip:

    python3 -m pip --version

Check OpenCV:

    python3 -c "import cv2; print(cv2.__version__)"

Check NumPy:

    python3 -c "import numpy; print(numpy.__version__)"

Check ONNX Runtime:

    python3 -c "import onnxruntime as ort; print(ort.__version__)"

Check Picamera2:

    python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"

Check YuNet:

    python3 -c "import cv2; cv2.FaceDetectorYN.create('models/face_detection_yunet_2026may.onnx','',(640,480),0.6,0.3,5000); print('YuNet OK')"

Check camera:

    rpicam-hello

Check video devices:

    ls /dev/video*

Check event files:

    ls -lh events/

Check logs:

    cat events/dms_log.csv

---

# 65. Development Philosophy

The project should be developed in layers.

Recommended order:

    1. Hardware
       |
       v
    2. Camera
       |
       v
    3. Face detection
       |
       v
    4. Facial landmarks
       |
       v
    5. Head pose
       |
       v
    6. Eye analysis
       |
       v
    7. Temporal state machine
       |
       v
    8. Warning manager
       |
       v
    9. Logging
       |
       v
    10. Dashboard
       |
       v
    11. Automatic startup
       |
       v
    12. Testing and optimization

This approach makes debugging much easier.

If something breaks, the developer can determine which layer is responsible.

---

# 66. Final Summary

This project began with a Raspberry Pi 4, an OV5647 camera, and the goal of creating a driver-monitoring system.

The development encountered several compatibility problems:

- `libcamera-hello` was unavailable
- OpenCV did not provide the expected `cv2.data`
- MediaPipe was not installed
- ARM64 MediaPipe wheels did not match Python 3.13
- A MediaPipe source build failed because of a missing C package
- MediaPipe packaging encountered an invalid `dev` version
- An attempted third-party wheel URL returned 404
- Python 3.11 was not available from the configured Debian 13 repositories

Instead of continuing to fight those dependencies, the project moved to a lightweight architecture based on:

    Picamera2
       +
    OpenCV 5
       +
    YuNet
       +
    ONNX Runtime
       +
    GPIO buzzer

This architecture is currently working.

The system can detect a face, monitor approximate horizontal attention, use time-based logic, warn the driver, save event images, and maintain a CSV history.

The most important next technical improvement is to replace approximate face-position logic with:

    facial landmarks
       +
    true head-pose estimation
       +
    eye-state analysis
       +
    temporal state machine

That will make the system substantially more capable of distinguishing normal driving behavior such as mirror checks and traffic observation from prolonged attention loss.

---

# 67. Project Completion Checklist

Before considering the prototype complete:

    [x] Camera tested
    [x] Python environment tested
    [x] OpenCV tested
    [x] ONNX Runtime tested
    [x] YuNet tested
    [x] Face detection tested
    [x] Buzzer tested
    [x] Event image saving implemented
    [x] CSV logging implemented
    [x] Startup script implemented
    [x] Documentation created

Next major milestone:

    [ ] Advanced facial landmarks
    [ ] Head pose
    [ ] Eye state
    [ ] Drowsiness
    [ ] Intelligent distraction classification
    [ ] Better mirror-check handling
    [ ] Production-style testing

---

END OF DOCUMENTATION
