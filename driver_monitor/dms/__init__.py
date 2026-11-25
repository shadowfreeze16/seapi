"""Driver Monitoring System (DMS) package.

A modular, Raspberry-Pi-friendly driver monitoring pipeline built on:

* OpenCV YuNet                 -- face detection + 5 facial landmarks
* 6DRepNet (MobileNetV3/V2)    -- full head pose (yaw / pitch / roll)
* OCEC                         -- eye open / closed classification

The package also contains a dependency-free 3D head renderer that rotates in
real time with the estimated head pose.
"""

__version__ = "2.0.0"