# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================
import cv2
import time
import numpy as np
import pose_module as pm  # Custom pose detection module

# Initialize Pose Detector
pose_detector = pm.poseDetector()
cap = cv2.VideoCapture(0)  # Open webcam (change index for external cameras)

# Variables for FPS
pTime = 0  # Previous time
cTime = 0  # Current time

# Counter and state for bicep curl counting
bicep_count = 0
direction = 0  # 0: Down, 1: Up

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to access the camera.")
        break
    
    img = pose_detector.findPose(img, False)  # Detect and annotate poses
    lmList = pose_detector.findPosition(img, False)

    if len(lmList) != 0:
        # Calculate left arm angle (change indices for right arm: 12, 14, 16)
        angle = pose_detector.findAngle(img, 11, 13, 15, False)


        # Bicep curl logic using the angle
        # Adjust thresholds based on your use case
        if angle > 160:  # Arm fully extended
            if direction == 1:
                bicep_count += 1
                direction = 0
        if angle < 60:  # Arm fully contracted
            direction = 1
        
        # Display bicep count
        cv2.putText(img, f'Biceps: {bicep_count}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        
    cv2.rectangle(img, (45, 445), (595, 485), (0, 255, 0), 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # Show the live feed
    cv2.imshow("Live Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
