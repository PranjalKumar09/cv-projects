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
cap = cv2.VideoCapture(0)  # Open webcam

# Variables for push-up counting and progress
push_up_count = 0
direction = 0  # 0: Down, 1: Up
percentage = 0
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Unable to access the camera.")
        break

    img = pose_detector.findPose(img, False)  # Detect and annotate poses
    lmList = pose_detector.findPosition(img, False)

    if len(lmList) != 0:
        # Calculate arm angle (adjust indices for your use case)
        angle = pose_detector.findAngle(img, 11, 13, 15)  # Left arm (shoulder, elbow, wrist)
        
        # Calculate percentage of push-up completion (linear interpolation)
        # Adjust the min_angle (fully down) and max_angle (fully up) thresholds as needed
        min_angle = 50   # Angle when fully down
        max_angle = 170  # Angle when fully up
        percentage = np.interp(angle, [min_angle, max_angle], [0, 100])

        # Push-up counting logic
        if percentage == 100:  # Fully up
            if direction == 0:
                push_up_count += 1
                direction = 1
        elif percentage == 0:  # Fully down
            direction = 0
        
        # Display push-up count
        cv2.putText(img, f'Push-ups: {push_up_count}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # Draw progress bar
    bar_width = int(np.interp(percentage, [0, 100], [45, 595]))  # Interpolate width of filled rectangle
    cv2.rectangle(img, (45, 445), (595, 485), (0, 255, 0), 3)  # Outer rectangle
    cv2.rectangle(img, (45, 445), (bar_width, 485), (0, 255, 0), cv2.FILLED)  # Filled progress bar
    
    # Display percentage on the progress bar
    cv2.putText(img, f'{int(percentage)}%', (260, 475), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    # Display the FPS
    cv2.putText(img, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0 , 0), 2)
    

    # Show the live feed
    cv2.imshow("Live Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
