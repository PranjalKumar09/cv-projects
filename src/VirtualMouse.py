# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================
import cv2
import numpy as np
import time
import pyautogui
import hand_tracking_module as htm  # Custom hand tracking module

# Canvas for drawing (optional)
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

detector = htm.handDetector(maxHands=1)  # Initialize the hand detector

pTime = 0  # Previous time for FPS calculation
smoothening_factor = 5  # Factor to smoothen the mouse movement
prev_x, prev_y = 0, 0  # Previous coordinates of the mouse

# Define the rectangular area (x, y, width, height)
rect_x, rect_y = 200, 100  # Top-left corner of the rectangle
rect_w, rect_h = 800, 500  # Width and height of the rectangle

while True:
    # 1. Find hand landmarks
    success, img = cap.read()
    if not success:
        print("Error: Unable to access the camera.")
        break
    img = cv2.flip(img, 1)  # Flip horizontally for natural interaction
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition2(img)

    # 2. Set tip of index & middle finger (for movement and clicking)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Tip of the index finger
        x2, y2 = lmList[12][1:]  # Tip of the middle finger

        # Draw circles at the finger tips
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), -1)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), -1)

    # 3. Check which fingers are up
    fingers = detector.fingersUp()
    
    if len(fingers) != 0:
        # 4. Only index finger up: Move the mouse (moving mode)
        if fingers[1] == 1 and fingers[2] == 0:  # Only index finger is up
            # 5. Convert Coordinates (to screen coordinates)
            # Ensure that the mouse is within the defined rectangle
            x = np.interp(x1, (rect_x, rect_x + rect_w), (0, pyautogui.size()[0]))  # Map X coordinate
            y = np.interp(y1, (rect_y, rect_y + rect_h), (0, pyautogui.size()[1]))  # Map Y coordinate

            # 6. Smoothen Values (to reduce jitter in the cursor)
            x = prev_x + (x - prev_x) / smoothening_factor
            y = prev_y + (y - prev_y) / smoothening_factor

            # 7. Move Mouse
            pyautogui.moveTo(x, y)
            prev_x, prev_y = x, y  # Update previous mouse position

        # 8. Both index & middle fingers up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:  # Both index and middle fingers up
            # 9. Find distance between index and middle fingers
            distance = detector.findDistance(8, 12, img)  # Index: 8, Middle: 12

            # 10. Check if distance is small enough to trigger a click
            if distance < 40:  # Adjust the threshold as needed
                pyautogui.click()

    # 11. Frame Rate (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    pTime = time.time()

    # 12. Draw the rectangle where the mouse is allowed to move
    cv2.rectangle(img, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

    # 13. Display the live feed
    cv2.imshow("Virtual Mouse Feed", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
