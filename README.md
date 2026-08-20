# AI Traffic Light Detection System

A Python project that uses YOLOv8 and OpenCV to detect vehicles in traffic video and adjust signal timing based on how many vehicles are waiting in each lane. It also tries to detect emergency vehicles and logs traffic data to SQLite.

## What it does

- Detects and counts vehicles lane-by-lane from a video feed using YOLOv8
- Adjusts how long a signal stays green based on vehicle count instead of a fixed timer
- Flags emergency vehicles so they could be prioritized
- Logs traffic counts to a SQLite database for later review

## How to run it

1. Install the required libraries: `opencv-python`, `ultralytics` (YOLOv8), `numpy`, `pygame`
2. Place a traffic video file in the project folder
3. Run `python main.py`
4. Watch the detection window — signal timing updates live based on vehicle count

## What I'd improve

- Right now it only works on pre-recorded video, not a live camera feed
- Emergency vehicle detection is basic — it doesn't distinguish ambulance vs. fire truck vs. police reliably yet
- No real UI, just console output and the OpenCV window — a simple dashboard would make it easier to demo
- Signal timing logic is rule-based, not learned — could try a reinforcement learning approach for smarter timing decisions

## Author

Deepak Kushwaha
