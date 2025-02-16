# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================
import cv2
import numpy as np
import os
import hand_tracking_module as htm  # Custom hand tracking module

# Set up directory and load menu bar images
folderpath = "static/MenuBar"
myList = os.listdir(folderpath)  # List of images in the folder
img_array = [cv2.imread(folderpath + "/" + file) for file in sorted(myList)]

header = img_array[0]  # Default header (No color selected)
draw_color = (0, 0, 0)  # Default drawing color (No color selected)
brush_thickness = 50
eraser_thickness = 50

# Canvas for drawing
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

detector = htm.handDetector(detectionConf=0.85)

while True:
    # 1. Capture frame and flip it
    success, img = cap.read()
    if not success:
        print("Error: Unable to access the camera.")
        break
    img = cv2.flip(img, 1)  # Flip horizontally for natural interaction

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPostion(img, draw=False)

    # Variables for fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Tip of the index finger
        x2, y2 = lmList[12][1:]  # Tip of the middle finger

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. Selection Mode (Two fingers up)
        if fingers[1] and fingers[2]:
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv2.FILLED)
            # print("Selection Mode")

            # Check for selection in the menu bar
            if y1 < 200:
                if 0 < x1 < 250:
                    header = img_array[0]  # No color selected
                    draw_color = (0, 0, 0)  # Default (no drawing)
                elif 250 < x1 < 450:
                    header = img_array[1]  # Red brush
                    draw_color = (0, 0, 255)
                elif 450 < x1 < 650:
                    header = img_array[2]  # Blue brush
                    draw_color = (255, 0, 0)
                elif 650 < x1 < 850:
                    header = img_array[3]  # Green brush
                    draw_color = (0, 255, 0)
                elif 850 < x1 < 1050:
                    # header = img_array[4]  # Eraser
                    draw_color = (0, 0, 0)

        # 5. Drawing Mode (Index finger up)
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
            # print("Drawing Mode")
            if draw_color == (0, 0, 0):  # Eraser
                cv2.line(canvas, (x1, y1), (x1, y1), draw_color, eraser_thickness)
            else:
                cv2.line(canvas, (x1, y1), (x1, y1), draw_color, brush_thickness)

    # Merge canvas with live video
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) # convert to gray scale
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV) # inverse the image (means now black part is become white)
    # img_inv2 = img_inv ,, and colour part is black
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR) #  covert gray to rgb
    img = cv2.bitwise_and(img, img_inv) 
    
    img = cv2.bitwise_or(img, canvas)

    # Add menu bar to the live feed
    img[0:200, 0:1280] = header

    cv2.imshow("Live Feed", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
