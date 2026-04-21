import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Blad kamery")
    exit()

def nothing(x):
    pass

cv2.namedWindow("foreground image")
cv2.createTrackbar("Threshold", "foreground image", 25, 255, nothing)

previous_frame = None
current_frame = None
next_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    previous_frame = current_frame
    current_frame = next_frame
    next_frame = gray

    if previous_frame is not None and current_frame is not None and next_frame is not None:

        diff1 = cv2.absdiff(next_frame, current_frame)
        diff2 = cv2.absdiff(next_frame, previous_frame)

        motion = cv2.bitwise_and(diff1, diff2)

        t = cv2.getTrackbarPos("Threshold", "foreground image")
        _, motion = cv2.threshold(motion, t, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("foreground image", motion)

    cv2.imshow("current image", gray)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()