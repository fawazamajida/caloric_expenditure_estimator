import cv2
import mediapipe as mp
import time
import calorific_estimation as calEst


class poseDetector():

    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        self.lmList = []
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return img, self.lmList

def main():
    cap = cv2.VideoCapture(0)
    detector = poseDetector()
    pTime = 0
    cal=0
    lmLen=0
    lmCount=0
    lmLong=[]
    while 1:
        success, img = cap.read()
        img, lmList = detector.findPose(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        lmLong.append(lmList)
        lmLen+=1
        if (detector.results.pose_landmarks):
            lmCount+=1
        else:
            lmCount=0
        if (lmCount>=3):
            currentCal=calEst.calculate(lmLong[lmLen-1],lmLong[lmLen-2],lmLong[lmLen-3],fps)
            cal+=currentCal/1000

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.putText(img, str(int(cal)), (200, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        # cv2.namedWindow("fullscreen", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("fullscreen", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow("fullscreen", img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()