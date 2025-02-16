# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================

import cv2
import mediapipe as mp
import time
import numpy as np

class poseDetector():
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, detection=0.5, trackCon=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.detectionConf = detection
        self.trackCon = trackCon

        # Initialize Mediapipe Pose
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackCon
        )

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
        # Get frame dimensions
        # h, w, c = img.shape

        # Check if pose landmarks are detected
        
    def findPosition   (self, img, draw=True) -> list:
        self.lmList = []
        if self.results.pose_landmarks:
            # Iterate through landmarks for additional processing or debugging
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # Convert normalized coordinates to pixel values
                cx, cy = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                self.lmList.append([id, cx, cy])
                
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # drawing circle on each landmark
        return self.lmList
                
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the coordinates of the three points
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        # Calculate the angle using the dot product and arctangent
        angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
        
        # Ensure the angle is positive
        if angle < 0:
            angle += 360

        if draw:
            # Draw lines between the points
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 3)
            
            # Draw circles at the points
            cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 8, (0, 0, 255), cv2.FILLED)
            
            # Display the angle value near the middle point
            cv2.putText(img, f'{int(angle)} deg', (x2 - 50, y2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return angle










def main():
    cap = cv2.VideoCapture(0)

    pTime = time.time()

    detector = poseDetector()
    while True:
        success, img = cap.read()
        if not success: 
            print("Failed to capture image")
            break
        
        img = detector.findPose(img)
        # lmList = detector.findPosition(img)
        # print(lmList)
        lmList = detector.findPosition(img, False)
        # now to print any specific landmark (like landmark 5)
        detect_point = 23
        print(lmList[detect_point])

        cv2.circle(img, (lmList[detect_point][1],lmList[detect_point][2]), 15, (255, 0, 0), -1)
        
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        img = cv2.resize(img, (800, 600))  # Resize to 800x600 for better visibility

        cv2.imshow("Pose Estimation", img)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        
        

if __name__ == '__main__':
    main()