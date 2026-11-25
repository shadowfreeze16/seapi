"""Enroll the authorized driver on Pi 4 - run once per driver.
Captures 20 face photos, trains a light OpenCV LBPH recognizer (Pi 4 friendly,
no heavy dlib needed), saves pi/authorized_driver.yml

Run on Pi:
    python3 face_enroll.py --name "Driver Name"
    Look at the camera, keep face straight, move head slightly left/right.
    Press Q when done, or it stops after 20 photos.
"""
import argparse
import cv2
import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, "authorized_driver.yml")
NAMES = os.path.join(BASE, "driver_names.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Driver 01")
    args = ap.parse_args()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    faces, labels = [], []
    print(f"[enroll] Capturing 20 photos for {args.name}. Look straight, move head slowly.")
    count = 0
    while count < 20:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        for (x, y, w, h) in detected:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(0)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/20", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break
        cv2.imshow("Enroll authorized driver - Q to quit", frame)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if len(faces) < 10:
        print("[enroll] FAILED - not enough faces captured. Try better light, face the camera.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, __import__("numpy").array(labels))
    recognizer.write(MODEL)
    open(NAMES, "w").write(args.name)
    print(f"[enroll] DONE - saved {MODEL} for {args.name} ({len(faces)} photos)")


if __name__ == "__main__":
    main()
