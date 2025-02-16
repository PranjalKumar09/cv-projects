# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================

from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import Sort
import math

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize video
cap = cv2.VideoCapture('stat9c/5229647-uhd_3840_2160_30fps.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load mask
car_mask = cv2.imread('static/car_count_mask.png', cv2.IMREAD_COLOR)

# Initialize tracker (using SORT)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define class names for YOLO
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Line for counting cars
line_pos = 600  # Adjust position based on frame size
car_count = 0
limits = (100, line_pos, 1180, line_pos)  # Define line limits for counting
totalCount = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # Apply mask if available
    if car_mask is not None and car_mask.shape[:2] == frame.shape[:2]:
        img = cv2.bitwise_and(frame, car_mask)
    else:
        img = frame

    # Resize frame for consistent processing
    img = cv2.resize(img, (1280, 720))
    frame = cv2.resize(frame, (1280, 720))

    # Run YOLO inference
    results = model(img, stream=True)

    # Prepare detections for SORT
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            current_class = class_names[cls]

            # Filter for vehicles
            if current_class in ['car', 'truck', 'motorcycle', 'bus'] and conf > 0.3:
                detections.append([x1, y1, x2, y2, conf])
                cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), l=9, t=2, colorR=(255, 0, 255))
                cvzone.putTextRect(frame, f'{current_class} {conf:.2f}', (max(0, x1), max(0, y1 - 20)), scale=1, thickness=1)

    # Update tracker
    if detections:
        tracked_objects = tracker.update(np.array(detections))
    else:
        tracked_objects = []

    # Process tracked objects
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {obj_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Count cars crossing the line
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            if obj_id not in totalCount:
                totalCount.append(obj_id)
                car_count += 1
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Draw counting line
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)
    cv2.putText(frame, f'Car Count: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Car Counter", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
