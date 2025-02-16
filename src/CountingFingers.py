# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================
import cv2
import time
import numpy as np
import os
import hand_tracking_module as htm

################################################################
# Camera settings
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Path to hand gesture images
folderPath = "static/Hand"
myList = sorted(os.listdir(folderPath))  # Ensure sorted order
overlayList = []

# Load hand gesture images
for path in myList:
    image_path = os.path.join(folderPath, path)
    image = cv2.imread(image_path)
    if image is not None:
        overlayList.append(image)
    else:
        print(f"Warning: Unable to load image {image_path}")

# Hand detector initialization
detector = htm.handDetector(detectionConf=0.7)

# IDs of fingertip landmarks
tipIds = [4, 8, 12, 16, 20]

# Variables for FPS calculation
pTime = time.time()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from camera.")
        break

    # Detect hanCV/Code/hand_tracking_module.pyds and positions
    img = detector.findHands(img)
    lmList = detector.findPostion(img)

    # Overlay the default hand image (closed fist) at the top-left corner
    if len(overlayList) > 0:
        h, w, c = overlayList[0].shape
        img[0:h, 0:w] = overlayList[0]

    fingers = []
    if lmList:
        # Thumb (horizontal check)
        fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 2][1] else 0)

        # Other fingers (vertical check)
        for id in range(1, 5):
            fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

        # Count the number of fingers up
        fingersUp = fingers.count(1)

        # Display the corresponding hand image
        if fingersUp < len(overlayList):
            overlayImg = overlayList[fingersUp]
            h, w, c = overlayImg.shape
            img[0:h, 0:w] = overlayImg

        # Display the number of fingers detected
        cv2.putText(img, f'Fingers: {fingersUp}', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Display the video feed
    cv2.imshow('Hand Gesture Detection', img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
