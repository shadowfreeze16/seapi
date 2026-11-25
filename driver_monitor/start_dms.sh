#!/usr/bin/env bash
#
# One-command launcher for the Driver Monitoring System (v2).
#
#   ./start_dms.sh
#   ./start_dms.sh --phone
#   ./start_dms.sh --video clip.mp4 --headless
#   ./start_dms.sh --video http://192.168.1.50:4747/video
#   ./start_dms.sh --record session.mp4 --no-buzzer
#
# Any arguments are passed through to driver_monitor.py.

set -u

# OpenCV's bundled Qt plugin only includes XCB; prefer it over Wayland and
# point Qt at the system DejaVu fonts instead of its missing bundled directory.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"

cd "$(dirname "$0")" || exit 1

echo "=========================================="
echo "       DRIVER MONITORING SYSTEM"
echo "=========================================="

# Activate the project virtual environment.
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Python virtual environment not found."
    echo "Create it with:"
    echo "  python3 -m venv --system-site-packages venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

echo "[1/4] Checking Python packages..."
python3 - <<'PY' || {
import cv2
import numpy
import onnxruntime

print(f"  OpenCV       {cv2.__version__}")
print(f"  NumPy        {numpy.__version__}")
print(f"  ONNX Runtime {onnxruntime.__version__}")
PY
    echo "ERROR: Required Python packages are missing."
    echo "Install them with: pip install -r requirements.txt"
    exit 1
}

echo "[2/4] Verifying ONNX models..."
if ! python3 tools/download_models.py --check; then
    echo
    echo "ERROR: One or more required models are missing or corrupted."
    echo "Download them with:"
    echo "  python3 tools/download_models.py"
    exit 1
fi

echo "[3/4] Camera and display"
python3 driver_monitor.py --list-cameras || true

# Interactive picker when started without arguments.
if [ "$#" -eq 0 ] && [ -t 0 ]; then
    echo
    echo "Choose mode:"
    echo "  1) Monitor with Raspberry Pi / USB camera (default)"
    echo "  2) Monitor with Android phone via DroidCam"
    echo "  3) Camera framing setup (save a neutral pose)"
    echo "  4) Enrol a driver identity"
    mode_choice=1
    read -r -p "Choice [1]: " mode_choice || true
    case "$mode_choice" in
        2) set -- --phone ;;
        3) set -- --setup-camera ;;
        4)
            read -r -p "Driver name: " driver_name || true
            if [ -z "${driver_name:-}" ]; then
                echo "ERROR: a driver name is required."
                exit 1
            fi
            set -- --enroll-driver "$driver_name"
            ;;
    esac

    if [ "${mode_choice:-1}" != "3" ] && [ "${mode_choice:-1}" != "4" ]; then
        read -r -p "Enable voice? [y/N]: " voice_choice || true
        case "${voice_choice:-}" in
            y|Y) set -- "$@" --voice ;;
        esac
        read -r -p "Laptop dashboard URL (blank = off): " dashboard_url || true
        if [ -n "${dashboard_url:-}" ]; then
            set -- "$@" --dashboard-url "$dashboard_url"
        fi
        read -r -p "Read GPS from gpsd? [y/N]: " gps_choice || true
        case "${gps_choice:-}" in
            y|Y) set -- "$@" --gps gpsd ;;
        esac
    fi
fi

echo "[4/4] Starting DMS..."
if [ ! -f "driver_monitor.py" ]; then
    echo "ERROR: driver_monitor.py not found."
    exit 1
fi

echo
echo "Press Q in the camera window to stop."
echo "Use --headless for ssh / no-display operation."
echo

python3 driver_monitor.py "$@"

STATUS=$?

echo
echo "=========================================="
echo "Driver Monitoring System stopped."
echo "Exit code: $STATUS"
echo "=========================================="

exit $STATUS