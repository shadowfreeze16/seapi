import cv2
import time
import os
import csv
from datetime import datetime
from gpiozero import Buzzer


# ============================================================
# DRIVER MONITORING SYSTEM
# Raspberry Pi 4 + OV5647 + OpenCV YuNet
# ============================================================

WIDTH = 640
HEIGHT = 480

MODEL = "models/face_detection_yunet_2026may.onnx"

EVENT_DIR = "events"
LOG_FILE = "events/dms_log.csv"

BUZZER_GPIO = 18


# ------------------------------------------------------------
# TIMING
# ------------------------------------------------------------

# Short glances are allowed.
# Alarm only after sustained off-center attention.
SIDE_WARNING_SECONDS = 3.0

# If no face is detected for this long, warn.
NO_FACE_WARNING_SECONDS = 2.5

# Prevent continuous event-photo generation.
EVENT_COOLDOWN = 5.0


# ------------------------------------------------------------
# FACE DETECTION
# ------------------------------------------------------------

CONFIDENCE = 0.60
NMS_THRESHOLD = 0.30
TOP_K = 5000


# ------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------

os.makedirs(EVENT_DIR, exist_ok=True)


# ------------------------------------------------------------
# BUZZER
# ------------------------------------------------------------

buzzer = Buzzer(BUZZER_GPIO)


def alarm():

    buzzer.on()
    time.sleep(0.25)
    buzzer.off()


# ------------------------------------------------------------
# CAMERA (DroidCam / USB webcam via V4L2)
# ------------------------------------------------------------

CAMERA_SOURCE = int(os.environ.get("DROIDCAM_DEV", "0"))

cap = cv2.VideoCapture(CAMERA_SOURCE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open camera source {CAMERA_SOURCE}. "
        "Start DroidCam and check /dev/video* devices."
    )

time.sleep(1)


# ------------------------------------------------------------
# YUNET
# ------------------------------------------------------------

if not os.path.exists(MODEL):

    raise FileNotFoundError(
        f"Model missing: {MODEL}"
    )


detector = cv2.FaceDetectorYN.create(
    MODEL,
    "",
    (WIDTH, HEIGHT),
    CONFIDENCE,
    NMS_THRESHOLD,
    TOP_K
)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

if not os.path.exists(LOG_FILE):

    with open(LOG_FILE, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "timestamp",
            "event",
            "direction",
            "face_count"
        ])


def log_event(event, direction, face_count):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(LOG_FILE, "a", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            timestamp,
            event,
            direction,
            face_count
        ])


# ------------------------------------------------------------
# EVENT PHOTO
# ------------------------------------------------------------

last_event = 0


def save_event(frame, event, direction):

    global last_event

    now = time.time()

    if now - last_event < EVENT_COOLDOWN:

        return

    last_event = now

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = (
        f"{EVENT_DIR}/"
        f"{timestamp}_{event}_{direction}.jpg"
    )

    cv2.imwrite(filename, frame)

    print(
        f"[EVENT] {event} | "
        f"{direction} | "
        f"{filename}"
    )

    log_event(
        event,
        direction,
        1
    )


# ------------------------------------------------------------
# FACE DIRECTION
# ------------------------------------------------------------

def estimate_direction(face):

    x, y, w, h = face[:4]

    center_x = x + w / 2

    image_center = WIDTH / 2

    normalized = (
        center_x - image_center
    ) / image_center


    if normalized < -0.25:

        return "LEFT"

    elif normalized > 0.25:

        return "RIGHT"

    else:

        return "CENTER"


# ------------------------------------------------------------
# MAIN STATE
# ------------------------------------------------------------

side_start = None
no_face_start = None

last_direction = "CENTER"

fps_start = time.time()
fps_count = 0
fps = 0


print()
print("==========================================")
print("       DRIVER MONITORING SYSTEM")
print("==========================================")
print()
print("Camera       : OV5647")
print("Resolution   :", WIDTH, "x", HEIGHT)
print("Face model   : YuNet")
print("Buzzer GPIO  :", BUZZER_GPIO)
print()
print("Rules:")
print("  Short left/right glance -> allowed")
print("  Sustained side attention -> warning")
print("  No face for > 2.5 sec   -> warning")
print()
print("Press Q to quit.")
print()


# ============================================================
# LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # CAMERA FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (WIDTH, HEIGHT))


        # ----------------------------------------------------
        # DETECT
        # ----------------------------------------------------

        detector.setInputSize(
            (WIDTH, HEIGHT)
        )

        _, faces = detector.detect(frame)


        if faces is None:

            faces = []


        face_count = len(faces)


        # ====================================================
        # NO FACE
        # ====================================================

        if face_count == 0:

            direction = "NO_FACE"

            cv2.putText(
                frame,
                "NO DRIVER FACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            if no_face_start is None:

                no_face_start = time.time()


            no_face_time = (
                time.time() -
                no_face_start
            )

            cv2.putText(
                frame,
                f"No face: {no_face_time:.1f}s",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 100, 255),
                2
            )


            if no_face_time >= NO_FACE_WARNING_SECONDS:

                cv2.putText(
                    frame,
                    "ATTENTION WARNING",
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                save_event(
                    frame,
                    "NO_FACE",
                    direction
                )

                alarm()


            side_start = None


        # ====================================================
        # DRIVER FOUND
        # ====================================================

        else:

            no_face_start = None


            # Use largest face.
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )


            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)


            # Draw face box.

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )


            # ------------------------------------------------
            # DIRECTION
            # ------------------------------------------------

            direction = estimate_direction(face)


            cv2.putText(
                frame,
                f"HEAD REGION: {direction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            if direction == "CENTER":

                side_start = None

                cv2.putText(
                    frame,
                    "ATTENTION: OK",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------------
            # LEFT / RIGHT
            # ------------------------------------------------

            else:

                if (
                    last_direction == "CENTER"
                    or side_start is None
                ):

                    side_start = time.time()


                side_time = (
                    time.time() -
                    side_start
                )


                cv2.putText(
                    frame,
                    f"Side glance: {side_time:.1f}s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 200, 255),
                    2
                )


                # Short glance is allowed.

                if side_time < SIDE_WARNING_SECONDS:

                    cv2.putText(
                        frame,
                        "SHORT GLANCE",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )


                # Sustained side attention.

                else:

                    cv2.putText(
                        frame,
                        "DISTRACTION WARNING",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2
                    )

                    save_event(
                        frame,
                        "SIDE_ATTENTION",
                        direction
                    )

                    alarm()


            last_direction = direction


        # ====================================================
        # FPS
        # ====================================================

        fps_count += 1

        fps_elapsed = (
            time.time() -
            fps_start
        )


        if fps_elapsed >= 1:

            fps = (
                fps_count /
                fps_elapsed
            )

            fps_count = 0

            fps_start = time.time()


        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (WIDTH - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Driver Monitoring System",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


finally:

        print()
        print("Stopping driver monitoring...")

        buzzer.off()

        cap.release()

        cv2.destroyAllWindows()

        print("Stopped.")


