"""Voice warnings on the Pi - speaks in the vehicle, not just on laptop.
Uses espeak / pyttsx3 if available, otherwise prints and beeps.
Run: python3 voice_alert.py "Warning, you look drowsy"
"""
import subprocess
import sys
import time

_last_spoken = {}
COOLDOWN = 20  # seconds, per message


def speak(text):
    now = time.time()
    if now - _last_spoken.get(text, 0) < COOLDOWN:
        return
    _last_spoken[text] = now
    print(f"[VOICE] {text}")
    # Try espeak first - lightest on Pi 4, no Python dependency
    try:
        subprocess.run(["espeak", "-s", "150", text],
                       timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    except Exception:
        pass
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[VOICE] speaker not available ({e}), message printed only")


MESSAGES = {
    "YELLOW": "Warning, you look drowsy. Please stay alert.",
    "RED_DROWSY": "Danger. You are falling asleep. Pull over safely now.",
    "RED_YAWN": "Danger. You are very drowsy. Pull over safely now.",
    "UNAUTHORIZED": "Warning. Unauthorized driver detected. Admin has been notified.",
    "NO_FACE": "Warning. Driver face not visible. Please face the camera.",
}


if __name__ == "__main__":
    speak(sys.argv[1] if len(sys.argv) > 1 else MESSAGES["YELLOW"])
