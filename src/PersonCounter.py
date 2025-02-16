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
model = YOLO('yolov8l.pt')

# Initialize video
cap = cv2.VideoCapture('static/istockphoto-897599120-640_adpp_is.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load mask
car_mask = cv2.imread('static/people_mask.png', cv2.IMREAD_COLOR)

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

# Define line for counting
line_pos = 600  # Adjust position based on frame size
limitsDown = (250, line_pos, 500, line_pos)
line_pos = 500  # Adjust position based on frame size
limitsUp = (650, line_pos, 1100, line_pos)

# Track people passing through the lines
up_count = 0
down_count = 0
totalCount = {}
totalCountUp  = totalCountDown = set()

while True:
    success, img = cap.read()
    if not success:
        break  # Handle the case when the video stream ends or cannot be read
    
    imgRegion = cv2.bitwise_and(img, car_mask)  # Apply mask if necessary

    # Run YOLO inference on the frame
    results = model(imgRegion, stream=True)
 
    detections = np.empty((0, 5))
 
    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
 
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence rounded to two decimal places
            cls = int(box.cls[0])
            currentClass = class_names[cls]
 
            if currentClass == "person" and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))  # Add detection for tracking
 
    # Update tracker with detections
    resultsTracker = tracker.update(detections)
 
    # Draw the crossing lines
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
 
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
 
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Circle at the center
 
        # Check if the object crosses the limits and update counts
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1] - 15 < cy < limitsUp[1] + 15:
            if id not in totalCountUp:
                totalCountUp.add(id)  # Use set for faster lookup
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
 
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1] - 15 < cy < limitsDown[1] + 15:
            if id not in totalCountDown:
                totalCountDown.add(id)  # Use set for faster lookup
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
 
    # Display counts
    cv2.putText(img, f'Up Count: {len(totalCountUp)}', (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, f'Down Count: {len(totalCountDown)}', (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)
 
    # Show the image with updates
    cv2.imshow("Image", img)
    
    # Optional: Display the region (optional line)
    # cv2.imshow("ImageRegion", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop when 'q' is pressed

cap.release()
cv2.destroyAllWindows()
