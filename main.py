from datetime import datetime
from ultralytics import YOLO
import pygame
import cv2
import numpy as np
import sqlite3
import time

# ================= INIT =================
pygame.mixer.init()
pygame.mixer.music.load("siren.mp3")

conn = sqlite3.connect("traffic.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS traffic_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lane1 INTEGER,
    lane2 INTEGER,
    lane3 INTEGER,
    lane4 INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

MAX_TIME = 60
MIN_TIME = 5

model = YOLO("yolov8n.pt")               # AI model use to vehicle detection
cap = cv2.VideoCapture("Video Project.mp4")

vehicle_classes = [2, 3, 5, 7]

# ================= FPS CONTROL =================
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

frame_skip = 3
delay = int((1000 / fps) * frame_skip)

# ================= FUNCTIONS =================
def calculate_green_times(lanes):
    total = sum(lanes)
    green_times = []
    for lane in lanes:
        if total == 0:
            g = MIN_TIME
        else:
            g = (lane / total) * MAX_TIME
        g = max(g, MIN_TIME)
        green_times.append(int(g))
    return green_times

# ================= VARIABLES =================
current_lane = 0
time_left = 10
frame_count = 0
last_time_update = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # 🔥 Frame skipping
    if frame_count % frame_skip != 0:
        continue

    # ================= YOLO DETECTION =================
    results = model(frame, conf=0.4)

    counts = [0, 0, 0, 0]
    emergency_detected = False
    emergency_lane = -1

    h, w, _ = frame.shape

    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # 🔥 DRAW BOX
        color = (0,255,0)
        if cls == 5:
            color = (0,0,255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        conf = float(box.conf[0])
        label = f"{model.names[cls]} {conf:.2f}"

        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # center
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2

        # 🔹 Lane classification
        if cx < w//2 and cy < h//2:
            lane_index = 0
        elif cx >= w//2 and cy < h//2:
            lane_index = 1
        elif cx < w//2 and cy >= h//2:
            lane_index = 2
        else:
            lane_index = 3

        if cls in vehicle_classes:
            counts[lane_index] += 1

        if cls == 5:
            emergency_detected = True
            emergency_lane = lane_index

    # ================= SHOW DETECTION =================
    cv2.imshow("Detection", frame)

    # ================= SIREN =================
    if emergency_detected:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)
    else:
        pygame.mixer.music.stop()

    # ================= TIMING =================
    green_times = calculate_green_times(counts)

    # ================= DATABASE =================
    cursor.execute(
        "INSERT INTO traffic_data (lane1, lane2, lane3, lane4) VALUES (?, ?, ?, ?)",
        (counts[0], counts[1], counts[2], counts[3])
    )

    if frame_count % 20 == 0:
        conn.commit()

    # ================= UI =================
    ui = np.zeros((700, 700, 3), dtype=np.uint8)

    cv2.rectangle(ui, (300, 0), (400, 700), (50,50,50), -1)
    cv2.rectangle(ui, (0, 300), (700, 400), (50,50,50), -1)
    cv2.rectangle(ui, (300,300), (400,400), (80,80,80), -1)

    positions = [(350,150),(550,350),(350,550),(150,350)]

    for i, (x,y) in enumerate(positions):
        color = (0,255,0) if i == current_lane else (0,0,255)

        cv2.circle(ui, (x,y), 25, color, -1)

        cv2.putText(ui, f"L{i+1}", (x-20, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(ui, f"{counts[i]}", (x-20, y+50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(ui, f"{green_times[i]}s", (x-30, y+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(ui, "AI TRAFFIC CONTROL SYSTEM", (140, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(ui, f"ACTIVE: LANE {current_lane+1}", (180, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.putText(ui, f"TIME: {time_left}s", (250, 690),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow("Traffic Vision PRO", ui)

    # ================= TIMER =================
    if time.time() - last_time_update >= 1:
        time_left -= 1
        last_time_update = time.time()

    if time_left <= 0:
        current_lane = (current_lane + 1) % 4
        time_left = green_times[current_lane]

    # ================= EXIT =================
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# ================= CLEANUP =================
conn.commit()
conn.close()
cap.release()
cv2.destroyAllWindows()




