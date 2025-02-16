# ===============================
# Author: Pranjal Kumar Shukla
# GitHub: https://github.com/PranjalKumar09/cv-projects/
# ===============================
import cv2
import time
import mediapipe as mp
import math


class handDetector():
    def __init__(self, mode = False, maxHands=2, detectionConf=0.5,trackCon=0.5 ) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackCon = trackCon
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()  
        self.mp_draw = mp.solutions.drawing_utils\
            
        self.tipIds = [4,8,12,16,20]
        
        self.pTime = 0
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the RGB image to detect hands
        self.results = self.hands.process(imgRGB)

        # Draw hand landmarks on the original BGR image if hands are detected
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks: # running through each image
                
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS) 
        return img
        
    def findPostion(self, img, handNo=0, draw=True):
        self.lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):
                x, y, z = ln.x * img.shape[1], ln.y * img.shape[0], ln.z  # converting the ratio to actual pixel
                self.lmList.append([id, int(x), int(y)])  # append the landmark to the list
                # if (id==0): # for palm
                #     cv2.circle(img, (int(x), int(y)), 15, (255, 0, 255), -1) # drawing circle on each landmark
                if draw:
                    cv2.circle(img, (int(x), int(y)), 5, (255, 0, 255), -1)  # drawing circle on each landmark
                    
        return self.lmList
    
    def findPosition2(self, img, handNo=0, draw=True, draw_box=True):
        """
        Finds the position of hand landmarks and optionally draws them.
        Also computes and optionally draws a bounding box around the hand.

        Args:
            img: The input image where the hand is detected.
            handNo: The index of the hand to process.
            draw: Whether to draw circles on detected landmarks.
            draw_box: Whether to draw a bounding box around the detected hand.

        Returns:
            lmList: A list of landmarks for the detected hand.
            bbox: The bounding box around the hand (x_min, y_min, x_max, y_max).
        """
        self.lmList = []
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
        bbox = (0, 0, 0, 0)

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                x, y, z = int(lm.x * img.shape[1]), int(lm.y * img.shape[0]), lm.z
                self.lmList.append([id, x, y])  # Append the landmark to the list

                # Update the bounding box coordinates
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

                # Optionally draw the landmark points
                if draw:
                    cv2.circle(img, (x, y), 5, (255, 0, 255), -1)

            # Compute the bounding box
            bbox = (x_min, y_min, x_max, y_max)

            # Optionally draw the bounding box
            if draw_box:
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return self.lmList, bbox

    
    def fingersUp(self) -> list:
        fingers = []
        if self.lmList:
            # Thumb (horizontal check)
            fingers.append(1 if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 2][1] else 0)

            # Other fingers (vertical check)
            for id in range(1, 5):
                fingers.append(1 if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2] else 0)

            # Count the number of fingers up
            # fingersUp = fingers.count(1)
        return fingers


    def findDistance(self, p1, p2, img=None, draw = False):
        """
        Finds the distance between two points on the hand.
        
        Args:
            p1 (int): Landmark index for the first point (e.g., index finger).
            p2 (int): Landmark index for the second point (e.g., middle finger).
            img: Optional image to draw the distance line on.
            
        Returns:
            float: The distance between the two points.
        """
        # Get the coordinates of both points
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]

        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Optionally draw a line between the two points on the image
        if img is not None and draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), -1)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), -1)

        return distance
    
        
    









# Initialize video capture and MediaPipe Hands
# cap = cv2.VideoCapture(0)










def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, img = cap.read() 
        img = detector.findHands(img)
        lmList = detector.findPostion(img, handNo=0)
        
        if len(lmList)!=0:
            print(lmList[0]) # palm
        
        # now lets show frame also
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # put FPS on frame


    
    # Display the video frame with landmarks
        cv2.imshow('Video', img)

    # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()